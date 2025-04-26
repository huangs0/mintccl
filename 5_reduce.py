from mpi4py import MPI
import torch
import triton
import triton.language as tl

from common    import *
from primitive import *
from topology  import *
from protocol  import *

# Now we're exploring the first MPI Collective with Computation, the reduce
# that sum/prod/max/min the buffer on different devices in elementwise manner
# A simple API for demonstration is simply as:
#   buff = torch.randn(count)  # Now everyone is random number
#   reduce(buff, root=root, dtype=tl.float16, op=Sum) # root has Sum of all

# Our algorithm is still the Ring Algorithm used in broadcast.py, the only 
# difference is the last one also need to send the summed output to root
# This means we finally have the Circular Ring so DEADLOCK can happen :(
# Thus we need to more carefully design the synchronization to avoid it...

# Let's started with an example 0 -> 1 -> 2 -> 3 -> 0, take 0 as root. 
# If like broadcast, everyone first wait for sending, it's a deadlock :)
# If everyone except root wait for sending, then after 4 round, still deadlock
# because 3 can not wait for 0 to accept its data...
# Bingo! A correct solution is 0 don't wait until `world_size` packets sent
#        and start wait since `world_size`, with necessary post processing

# Another difficulty is computation or Reduce Operator, i.e., elementwise sum/
# prod/min/max. This sounds simple for Triton, but with LL128 we have the flag
# and we shall not reduce the flag or protocol corrupted. Thus, we implement a
# comp_ll128() in protocol.py for such a need.

# Now it's for kernel, this is not an easy job and current impl still have
# many space left for performance tuning, but a readable good start
@triton.jit
def reduce_kernel(recvbuff,   # local  buffer to recv, src don't have
                  sendbuff,   # remote buffer to send, dst don't have
                  inputbuff,  # buffer for input, only applicable to src
                  outputbuff, # buffer for output,only applicable to dst
                  count,      # total number of bytes to send
                  # RedOp: triton.JITFunction, # Reduce Operator
                  world_size: tl.constexpr, # MPI WORLD_SIZE
                  # dtype: tl.constexpr,  # dtype of inputbuff as computed
                  root:  tl.constexpr): # root constexpr as code too different
    # Similarly let's do the configuration and initialization
    pid, tid = tl.program_id(0), thread_id(0)
    flag = ((pid * WORKGROUPS_PER_BLOCK + tid // WORKGROUP_SIZE) * ITERS_PER_SEND + 1)
    boundary = count // BYTES_LL128_RAW # // is integer division in Python
    sendbuff   = sendbuff.to(tl.uint64).to(tl.pointer_type(tl.uint8)) 
    recvbuff   = recvbuff.to(tl.uint64).to(tl.pointer_type(tl.uint8)) 
    inputbuff  = inputbuff.to(tl.uint64).to(tl.pointer_type(tl.uint8))
    outputbuff = outputbuff.to(tl.uint64).to(tl.pointer_type(tl.uint8))
    if root: round_ = 1 # round_ only applicable to root

    while flag < boundary:
        # Similarly, let's first do the synchronization
        offset_msg = (pid * WORKGROUPS_PER_BLOCK + tid // WORKGROUP_SIZE) * ITERS_PER_SEND * BYTES_LL128_MSG

        post_ll128(recvbuff + offset_msg) # everyone recv, so everyone post
        wait_ll128(sendbuff + offset_msg) # everyone send, so everyone wait

        # Let's replace original one pass with a loop whose length ~ BUFF_SIZE
        for _ in range(WORKGROUP_ITERS): # see protocol.py for WORKGROUP_ITERS
            if flag < boundary: # need a more detailed boundary checking
                # offset_raw shall be more frequently managed
                offset_raw = (flag - 1) * BYTES_LL128_RAW # offset to raw data 

                # Loading Phase, now everyone need to load from buffer!
                v0, v1, v2, v3 = load_ll128(inputbuff + offset_raw, flag)

                if not root: # not root then receive and compute, only one phase
                    r0, r1, r2, r3 = recv_ll128(recvbuff + offset_msg, flag)
                    v0, v1, v2, v3 = comp_ll128(v0, v1, v2, v3, r0, r1, r2, r3)
                
                # Sending Phase, root send raw and else send computed
                send_ll128(sendbuff + offset_msg, v0, v1, v2, v3)

                if root:
                    if round_ >= world_size - 1:
                    # NOTE after world_size packet, root start receiving packet
                    # to avoid deadlock (because world_size edges, set this
                    # also maintain every edge busy) And root has diverged flag
                    # for receiving, phase diff is exactly `world_size` :)
                        flag_ = flag - NUM_BLOCKS * WORKGROUPS_PER_BLOCK * \
                        ITERS_PER_SEND * WORKGROUP_ITERS * (world_size - 2)
                        offset_ = (flag_ - 1) * BYTES_LL128_RAW
                        r0, r1, r2, r3 = recv_ll128(recvbuff + offset_msg, flag_)
                        save_ll128(outputbuff + offset_, r0, r1, r2, r3)

            # advance flags and offset_msg
            flag += NUM_BLOCKS * WORKGROUPS_PER_BLOCK * ITERS_PER_SEND
            offset_msg += NUM_BLOCKS * WORKGROUPS_PER_BLOCK * ITERS_PER_SEND * BYTES_LL128_MSG
        
        if root: round_ += 1 # round_ only applicable to root


# Now it's time to manage the metadata, it's similar to broadcast
def reduce(buff: torch.Tensor, root: int):
    count = buff.numel() * buff.itemsize
    comm = MPI.COMM_WORLD
    rank, world_size = comm.Get_rank(), comm.Get_size()
    ring: Route = None
    if rank == root:
        # ring = find_ring() # construct the ring from topology
        ring = [0, 1]
        comm.bcast(ring, root=root)
    else: # others wait until root send out the ring
        ring = comm.bcast(None, root=root)
    pos = ring.index(rank) # find one's position inside the ring
    # In reduce, everyone have a recvbuff, including root!
    recvbuff = torch.empty((BUFF_SIZE,), dtype=torch.uint8).data_ptr()
    # Send recv buffer to prev and prev receive this buffer
    send_ptr(recvbuff, ring[(pos - 1) % world_size])
    sendbuff = recv_ptr(ring[(pos + 1) % world_size])
    # inputbuff and outputbuff on root is the same
    inputbuff  = buff.data_ptr()
    outputbuff = buff.data_ptr() if rank == root else NULL.value
    # Now launch the kernel for communication, well done!
    reduce_kernel[(NUM_BLOCKS.value, )](recvbuff, sendbuff, inputbuff, outputbuff, 
        count, world_size, root=(rank==root), num_warps=NUM_WARPS.value)


if __name__ == "__main__":
    # NOTE Tested on V100x8
    import os
    rank = MPI.COMM_WORLD.Get_rank()
    set_gpu(rank)
    # Here we use one env var to easily configure sender and receiver
    root = int(os.getenv("ROOT_TEMP", 0))
    buff = torch.randn(TEST_COUNT, dtype=torch.float16)
    if TEST_MODE: # make a copy of root as it will be overwritten :)
        nccl_buff = buff.clone().detach()
    
    reduce(buff, root) # call the reduce as proposed
    
    if TEST_MODE:
        import torch.distributed as dist
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend="nccl", rank=rank, 
                                world_size=MPI.COMM_WORLD.Get_size())
        dist.reduce(nccl_buff, root, op=dist.ReduceOp.SUM)
        if rank == root: # only root has the result!
            print(f"# rank {rank}, equal: {torch.equal(buff, nccl_buff)}")
            if not torch.equal(buff, nccl_buff):
                print("# head")
                print(buff[:16])
                print(nccl_buff[:16])
                print("# tail")
                print(buff[-16:])
                print(nccl_buff[-16:])
                eq = torch.eq(buff, nccl_buff)
                mis = torch.nonzero(~eq, as_tuple=False)
                print(mis[0].item())