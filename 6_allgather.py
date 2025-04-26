from mpi4py import MPI
import torch
import triton
import triton.language as tl

from common    import *
from primitive import *
from topology  import *
from protocol  import *

# Now we're exploring allgather, the first rank-based collective algorithm.
# Allgather can be regarded as extended broadcast where each node has part of
# source according to their rank. We aims at having following API Example
#  buff = torch.zeros((world_size, size)) # first init all to 0
#  buff[rank] = buff[rank].random_()      # has input according to rank
#  allreduce(buff)                        # everyone's buff has gathered value

# The major difficulty here is now everyone is root. For every previous example
# we implemented, the program totally goes together, i.e., every 
# A simple solution here is to arange the buffer and SMs according to the rank
# that each rank's data has its own SM and buffer chunks.

# Kernel definition is similar to broadcast but now each node is src / dst
@triton.jit
def allgather_kernel_temporal(
    recvbuff,    # local  buffer to recv, src don't have
    sendbuff,    # remote buffer to send, dst don't have
    inputbuff,   # buffer for input, can be same as outputbuff
    outputbuff,  # buffer for output, can be same as inputbuff
    count,       # total no.element to send
    rank,        # rank of the caller
    dest,        # rank of the destination, i.e., next node
    world_size): # MPI World Size
    pid, tid = tl.program_id(0), thread_id(0)
    flag = ((pid * WORKGROUPS_PER_BLOCK + tid // WORKGROUP_SIZE) * ITERS_PER_SEND + 1)
    boundary = count // BYTES_LL128_RAW # // is integer division in Python
    sendbuff   = sendbuff.to(tl.uint64).to(tl.pointer_type(tl.uint8)) 
    recvbuff   = recvbuff.to(tl.uint64).to(tl.pointer_type(tl.uint8)) 
    inputbuff  = inputbuff.to(tl.uint64).to(tl.pointer_type(tl.uint8)) 
    outputbuff = outputbuff.to(tl.uint64).to(tl.pointer_type(tl.uint8)) 
    # Now we need flags_per_rank to distinguish who is the root of flag
    flags_per_rank = (count // BYTES_LL128_RAW) // world_size

    while flag < boundary:
        # Similarly, let's first do the synchronization
        offset_msg = (pid * WORKGROUPS_PER_BLOCK + tid // WORKGROUP_SIZE) * ITERS_PER_SEND * BYTES_LL128_MSG
        
        # NOTE there might be some redundant syncrhonization
        post_ll128(recvbuff + offset_msg)
        wait_ll128(sendbuff + offset_msg)

        for _ in range(WORKGROUP_ITERS): # see protocol.py for WORKGROUP_ITERS
            if flag < boundary: # need a more detailed boundary checking
                # offset_raw shall be more frequently managed
                offset_raw = (flag - 1) * BYTES_LL128_RAW # offset to raw data
                # NOTE only diff is to chck who is the root for current flag
                root_rank = flag // flags_per_rank
                root = rank == root_rank # bool if myself is root of this
                send = dest != root_rank # bool if next is root of this

                # Receiving phase
                if root: # root need to load and encode raw bytes to LL128
                    v0, v1, v2, v3 = load_ll128(inputbuff + offset_raw, flag)
                else:   
                    # Everyone not root just receive LL128 message
                    v0, v1, v2, v3 = recv_ll128(recvbuff + offset_msg, flag)
                    
                # Saving Phase
                if not root: # Everyone not root save it to GMEM
                    save_ll128(outputbuff + offset_raw, v0, v1, v2, v3) 

                # Sending Phase
                if send: # Avoid last one send to root again
                    send_ll128(sendbuff + offset_msg, v0, v1, v2, v3)
        
            flag += NUM_BLOCKS * WORKGROUPS_PER_BLOCK * ITERS_PER_SEND
            offset_msg += NUM_BLOCKS * WORKGROUPS_PER_BLOCK * ITERS_PER_SEND * BYTES_LL128_MSG

@triton.jit
def allgather_kernel_spatial(
    recvbuff,    # local  buffer to recv, src don't have
    sendbuff,    # remote buffer to send, dst don't have
    inputbuff,   # buffer for input, can be same as outputbuff
    outputbuff,  # buffer for output, can be same as inputbuff
    count,       # total no.element to send
    rank,        # rank of the caller
    dest,        # rank of the destination, i.e., next node
    world_size): # MPI World Size
    # Okay now we can fix this at the beginning
    pid, tid = tl.program_id(0), thread_id(0)
    blocks_rank = tl.num_programs(0) // world_size # NOTE assume divisible
    root_rank = pid // blocks_rank # fixed for 
    root = root_rank == rank # bool indicates if serving as rank
    send = root_rank != dest # bool indicates if need to send
    # But now we shall manage the flag differently! Remember our flag intends
    # to be unique for each chunk of data and our previous thread arangement
    # is continuous to data but now it's not as blocks handles different chunks
    flags_per_rank = (count // BYTES_LL128_RAW) // world_size
    flag = flags_per_rank * root_rank + (pid % blocks_rank * WORKGROUPS_PER_BLOCK
            + tid // WORKGROUP_SIZE) * ITERS_PER_SEND + 1
    # and boundary is also per rank!!!
    boundary = flags_per_rank * (root_rank + 1)
    sendbuff   = sendbuff.to(tl.uint64).to(tl.pointer_type(tl.uint8)) 
    recvbuff   = recvbuff.to(tl.uint64).to(tl.pointer_type(tl.uint8)) 
    inputbuff  = inputbuff.to(tl.uint64).to(tl.pointer_type(tl.uint8)) 
    outputbuff = outputbuff.to(tl.uint64).to(tl.pointer_type(tl.uint8)) 

    # Formulating like this the loops will be nearly the same as broadcast.py
    while flag < boundary: # local boundary
        # Keep send and recv buff the same, but the semantic will change automaticallyh 
        offset_msg = (pid * WORKGROUPS_PER_BLOCK + tid // WORKGROUP_SIZE) * ITERS_PER_SEND * BYTES_LL128_MSG

        if not root: # Everyone not root finished recv so post
            post_ll128(recvbuff + offset_msg)
        if send: # Everyone send shall wait to send
            wait_ll128(sendbuff + offset_msg)

        for _ in range(WORKGROUP_ITERS): # see protocol.py for WORKGROUP_ITERS
            if flag < boundary: # need a more detailed boundary checking
                # offset_raw shall be more frequently managed
                offset_raw = (flag - 1) * BYTES_LL128_RAW # offset to raw data 

                # Receiving Phase
                if root: # root need to load and encode raw bytes to LL128
                    v0, v1, v2, v3 = load_ll128(inputbuff + offset_raw, flag)
                else:   # Everyone not root just receive LL128 message
                    v0, v1, v2, v3 = recv_ll128(recvbuff + offset_msg, flag)
                
                # Saving Phase
                if not root: # Everyone not root save it to GMEM
                    save_ll128(outputbuff + offset_raw, v0, v1, v2, v3) 

                # Sending Phase
                if send: # Avoid last one send to root again
                    send_ll128(sendbuff + offset_msg, v0, v1, v2, v3)
        
            flag += blocks_rank * WORKGROUPS_PER_BLOCK * ITERS_PER_SEND
            offset_msg += NUM_BLOCKS * WORKGROUPS_PER_BLOCK * ITERS_PER_SEND * BYTES_LL128_MSG


def allgather(buff: torch.Tensor):
    count = buff.numel() * buff.itemsize # we don't care about layout
    comm = MPI.COMM_WORLD
    rank, world_size = comm.Get_rank(), comm.Get_size()
    ring: Route = None # declare here only for Python scope
    if rank == 0:
        ring = find_ring() # construct the ring from topology
        comm.bcast(ring, root=0)
    else: # others wait until root send out the ring
        ring = comm.bcast(None, root=0)
    # exchange recv and next ptr
    pos = ring.index(rank) # find one's position inside the ring
    recvbuff = torch.empty((BUFF_SIZE,), dtype=torch.uint8).data_ptr()
    send_ptr(recvbuff, ring[(pos - 1) % world_size])
    sendbuff = recv_ptr(ring[(pos + 1) % world_size])
    inputbuff, outputbuff = buff.data_ptr(), buff.data_ptr()
    allgather_kernel_temporal[(NUM_BLOCKS.value, )](recvbuff, sendbuff, inputbuff, outputbuff,
        count, rank, ring[(pos + 1) % world_size], world_size, num_warps=NUM_WARPS.value)
    

if __name__ == "__main__":
    # NOTE launch with V100/A100/H100/B100 x2/x4/x8 Machine
    rank, world_size = MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()
    set_gpu(rank)
    # Here everyone shall be zeros and fill buff[rank] with random value!
    buff = torch.zeros((world_size, TEST_COUNT // world_size))
    buff[rank] = buff[rank].random_()

    allgather(buff) # call our API
    
    if TEST_MODE:
        import torch.distributed as dist
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend="nccl", rank=rank, 
                                world_size=MPI.COMM_WORLD.Get_size())
        nccl_buff = torch.empty_like(buff)
        dist.all_gather_into_tensor(nccl_buff, buff[rank]) # buff[rank] is same
        print(f"# rank {rank}, equal: {torch.equal(buff, nccl_buff)}")