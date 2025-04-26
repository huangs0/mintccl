import os
from mpi4py import MPI
import torch
import triton
import triton.language as tl

from common    import *
from primitive import *
from topology  import *
from protocol  import *

# Now we're exploring the simplest MPI Collectives, broadcast, which will
# copies `count` elements from `buff` on the `root` rank to all ranks’ `buff`
# A simple API for demonstration is:
#   buff = torch.randn(count) if rank == root else torch.empty(count)
#   broadcast(buff, root=root) # then everyone has rank 0's buff

# Here we will implement the default Ring-Based Algorithm for broadcasting,
# Ring is basically a circle based on topology like 0 <-> 1 <-> 2 <-> 3 <-> 0
# will have a ring formed by 0 -> 1 -> 2 -> 3 -> 0, and in broadcast, we ignore
# the last connection (3 -> 0) as 0 (as root) already have elements
# Algorithm to find a ring is a fundamental graph algorithm :)
# NOTE check details of find_ring is in topology.py

# Now it's for kernel, it's based on the sendrecv_buff_kernel we implemented
# and the only differences is that everyone except root has outputbuff != NULL
@triton.jit
def broadcast_kernel(recvbuff,  # local  buffer to recv, src don't have
                     sendbuff,  # remote buffer to send, dst don't have
                     inputbuff, # buffer for input, only applicable to src
                     outputbuff,# buffer for output,only applicable to dst
                     count):    # no.bytes to send, count >> BUFF_SIZE
    # Similarly let's do the configuration and initialization
    pid, tid = tl.program_id(0), thread_id(0)
    flag = ((pid * WORKGROUPS_PER_BLOCK + tid // WORKGROUP_SIZE) * ITERS_PER_SEND + 1).to(tl.uint64)
    boundary = (count // BYTES_LL128_RAW).to(tl.uint64)
    # here we modify src and dst to root and send
    root = inputbuff != NULL # or outputbuff == NULL
    send = sendbuff  != NULL # the last one on ring don't need to send :)
    sendbuff   = sendbuff.to(tl.uint64).to(tl.pointer_type(tl.uint8)) 
    recvbuff   = recvbuff.to(tl.uint64).to(tl.pointer_type(tl.uint8)) 
    inputbuff  = inputbuff.to(tl.uint64).to(tl.pointer_type(tl.uint8)) 
    outputbuff = outputbuff.to(tl.uint64).to(tl.pointer_type(tl.uint8)) 

    while flag < boundary:
        # Similarly, let's first do the synchronization
        offset_msg = (pid * WORKGROUPS_PER_BLOCK + tid // WORKGROUP_SIZE) * ITERS_PER_SEND * BYTES_LL128_MSG

        if not root: # Everyone not root finished recv so post
            tmp = post_ll128(recvbuff + offset_msg)
        if send: # Everyone send shall wait to send
            tmp = wait_ll128(sendbuff + offset_msg)

        # Let's replace original one pass with a loop whose length ~ BUFF_SIZE
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
        
                # advance flags and offset_msg
            flag += NUM_BLOCKS * WORKGROUPS_PER_BLOCK * ITERS_PER_SEND
            offset_msg += NUM_BLOCKS * WORKGROUPS_PER_BLOCK * ITERS_PER_SEND * BYTES_LL128_MSG


def broadcast(buff: torch.Tensor, root: int):
    count = buff.numel() * buff.itemsize
    comm = MPI.COMM_WORLD
    rank, world_size = comm.Get_rank(), comm.Get_size()
    ring: Route # declare here to avoid scope limitation
    if rank == root:
        # ring = find_ring() # construct the ring from topology
        ring = [0, 1, 0]
        comm.bcast(ring, root=root)
    else: # others wait until root send out the ring
        ring = comm.bcast(None, root=root)
    pos = ring.index(rank) # find one's position inside the ring
    sendbuff, recvbuff = NULL.value, NULL.value
    if pos != 0: # everyone not the first init buffer and send metadata to prev
        recvbuff = torch.zeros(BUFF_SIZE, dtype=torch.uint8).data_ptr()
        send_ptr(recvbuff, ring[(pos - 1) % world_size])
    if pos != len(ring) - 2: # everyone not the last receive the sendbuff, 2 as root is repeated
        sendbuff = recv_ptr(ring[(pos + 1) % world_size])
    inputbuff  = buff.data_ptr() if rank == root else NULL.value # root  has input
    outputbuff = buff.data_ptr() if rank != root else NULL.value # other has output
    # Now launch the kernel for communication, well done!
    broadcast_kernel[(NUM_BLOCKS.value, )](recvbuff, sendbuff, 
        inputbuff, outputbuff, count, num_warps=NUM_WARPS.value)
    
if __name__ == "__main__":
    # NOTE launch with V100/A100/H100/B100 x4/x8 Machine
    rank = MPI.COMM_WORLD.Get_rank()
    set_gpu(rank)
    # Here we use one env var to easily configure sender and receiver
    root = int(os.getenv("ROOT_TEMP", 0))
    buff = torch.randn(TEST_COUNT) if rank == root else torch.empty(TEST_COUNT)
    broadcast(buff, root)
    if TEST_MODE:
        import torch.distributed as dist
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend="nccl", rank=rank, 
                                world_size=MPI.COMM_WORLD.Get_size())
        nccl_buff = buff if rank == root else torch.empty_like(buff)
        dist.broadcast(nccl_buff, root) # torch.dist's src is our root
        print(f"# rank {rank}, equal: {torch.equal(buff, nccl_buff)}")
        if rank != root: # root has no meaning
            print("# head")
            print(buff[:16])
            print(nccl_buff[:16])
            print("# tail")
            print(buff[-16:])
            print(nccl_buff[-16:])
