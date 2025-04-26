from mpi4py import MPI
import torch
import triton
import triton.language as tl

from common    import *
from primitive import *
from topology  import *
from protocol  import *

# Our treatment of reducescatter is roughly the same as allgather ~ broadcast
# Based on our measurement 
@triton.jit
def reducescatter_kernel(
    recvbuff,    # local  buffer to recv, src don't have
    sendbuff,    # remote buffer to send, dst don't have
    inputbuff,   # buffer for input, can be same as outputbuff
    outputbuff,  # buffer for output, can be same as inputbuff
    count,       # total no.element to send
    rank,        # rank of the caller
    world_size): # MPI World Size

    blocks_rank = tl.num_programs(0) // world_size # NOTE assume divisible
    root_rank = tl.program_id(0) // blocks_rank # fixed for 
    root = root_rank == rank # bool indicates if serving as rank
    if root: round_ = 0
    # But now we shall manage the flag differently! Remember our flag intends
    # to be unique for each chunk of data and our previous thread arangement
    # is continuous to data but now it's not as blocks handles different chunks
    flags_per_rank = (count // BYTES_LL128_RAW) // world_size
    flag = flags_per_rank * root_rank + (tl.program_id(0) % blocks_rank * WORKGROUPS_PER_BLOCK
            + thread_id(0) // WORKGROUP_SIZE) * ITERS_PER_SEND + 1
    # and boundary is also per rank!!!
    boundary = flags_per_rank * (root_rank + 1)
    sendbuff   = sendbuff.to(tl.uint64).to(tl.pointer_type(tl.uint8)) 
    recvbuff   = recvbuff.to(tl.uint64).to(tl.pointer_type(tl.uint8)) 
    inputbuff  = inputbuff.to(tl.uint64).to(tl.pointer_type(tl.uint8)) 
    outputbuff = outputbuff.to(tl.uint64).to(tl.pointer_type(tl.uint8)) 

    # Formulating like this the loops will be nearly the same as broadcast.py
    while flag < boundary: # local boundary
        # Keep send and recv buff the same, but the semantic will change automaticallyh 
        offset_msg = (tl.program_id(0) * WORKGROUPS_PER_BLOCK + 
            thread_id(0) // WORKGROUP_SIZE) * ITERS_PER_SEND * BYTES_LL128_MSG

        post_ll128(recvbuff + offset_msg)
        wait_ll128(sendbuff + offset_msg)

        for _ in range(WORKGROUP_ITERS): # see protocol.py for WORKGROUP_ITERS
            if flag < boundary: # need a more detailed boundary checking
                # offset_raw shall be more frequently managed
                offset_raw = (flag - 1) * BYTES_LL128_RAW # offset to raw data 

                v0, v1, v2, v3 = load_ll128(inputbuff + offset_raw, flag)

                # Receiving Phase
                if root: # root need to load and encode raw bytes to LL128
                    # NOTE after world_size packet, root start receiving packet
                    # to avoid deadlock (because world_size edges, set this
                    # also maintain every edge busy) And root has diverged flag
                    # for receiving, phase diff is exactly `world_size` :)
                    if round_ >= world_size:
                        flag_ = flag - blocks_rank * WORKGROUPS_PER_BLOCK * \
                            ITERS_PER_SEND * WORKGROUP_ITERS * world_size
                        offset_ = flag_ * BYTES_LL128_RAW
                        r0, r1, r2, r3 = recv_ll128(recvbuff + offset_msg, flag_)
                        save_ll128(outputbuff + offset_, r0, r1, r2, r3) 
                else: # not root then receive and compute, only one phase
                    r0, r1, r2, r3 = recv_ll128(recvbuff + offset_msg, flag)
                    v0, v1, v2, v3 = comp_ll128(v0, v1, v2, v3, r0, r1, r2, r3)
                
                # Sending Phase, root send raw and else send computed
                send_ll128(sendbuff + offset_msg, v0, v1, v2, v3)
        
                offset_msg += NUM_BLOCKS * WORKGROUPS_PER_BLOCK * ITERS_PER_SEND * BYTES_LL128_MSG
                # NOTE flag management is different! Let's use BLOCKS_RANK
                flag += blocks_rank * WORKGROUPS_PER_BLOCK * ITERS_PER_SEND
        
        if root: round_ += 1 # round_ only applicable to root

def reducescatter(buff: torch.Tensor):
    count = buff.numel() * buff.itemsize
    comm = MPI.COMM_WORLD
    rank, world_size = comm.Get_rank(), comm.Get_size()
    ring: Route = None
    if rank == 0:
        ring = find_ring() # construct the ring from topology
        comm.bcast(ring, root=0)
    else: # others wait until root send out the ring
        ring = comm.bcast(None, root=0)
    pos = ring.index(rank) # find one's position inside the ring
    # In reduce, everyone have a recvbuff, including root!
    recvbuff = torch.empty((BUFF_SIZE,), dtype=torch.uint8).data_ptr()
    # Send recv buffer to prev and prev receive this buffer
    send_ptr(recvbuff, ring[(pos - 1) % world_size])
    sendbuff = recv_ptr(ring[(pos + 1) % world_size])
    reducescatter_kernel[(NUM_BLOCKS.value, )](recvbuff, sendbuff, buff, buff,
        count, rank, world_size, num_warps=NUM_WARPS.value)
    
if __name__ == "__main__":
    import os
    rank, world_size = MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()
    set_gpu(rank)
    # Here we use one env var to easily configure sender and receiver
    buff = torch.randn((world_size, TEST_COUNT // world_size), dtype=torch.float16)
    if TEST_MODE: # make a copy of root as it will be overwritten :)
        nccl_input = buff.copy_()
    reducescatter(buff)
    if TEST_MODE:
        import torch.distributed as dist
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend="nccl", rank=rank, 
                                world_size=MPI.COMM_WORLD.Get_size())
        nccl_output = torch.empty(TEST_COUNT // world_size, dtype=torch.float16)
        dist.reduce_scatter_tensor(nccl_output, nccl_input)
        print(f"# rank {rank}, equal: {torch.equal(buff[rank], nccl_output)}")