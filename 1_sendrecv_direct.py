import torch
import triton
import triton.language as tl
from mpi4py import MPI # Used to create the shared memory for metadata

from common    import *
from primitive import *

# Let's started from a simple example: Two GPU 0 and 1 where GPU 0 want to send
# a tensor to GPU 1, and GPU 1 need an empty tensor of same shape as buffer.
# As Python has GIL, let's use multi-process with MPI(https://mpitutorial.com)
# Our target is to implement send() & recv() for following one-line interface:
#   send(buff) if MPI.COMM_WORLD.Get_rank() == 0 else recv(buff) # rank == 1

# Here we assume GPU 0 and GPU 1 has peer-to-peer unified memory access, 
# which means access other GPU's memory via NvLink can be simplified as 
# the standard ld.volatile.global and st.volatile.global PTX primitive
# we just need to pass in correct address to remote tensor

# Based on this assumption, there's two way to handle the movement:
# 1. Sender   use st.volatile.global to save data to remote
# 2. Receiver use ld.volatile.global to read data from remote
# An important factor in choosing these two is SYNChronization, i.e., 
#   how and when receiver knows data is arrived
# Choosing 1. means only the kernel is finished on sender side, and receiver
#   actually don't know (need MPI to help broadcast message
# Choosing 2. means when each `ld` is finished inside kernel on receiver side
#   because ld is a blocking call. Intra-kernel also facilitate computations
# Thus, let's implement the 2nd approach, with only receiver kernel not sender

@triton.jit
def recv_kernel(sendbuff, recvbuff, nbyte, BLOCK_BYTE: tl.constexpr):
    # Here we use a trick called Persistent Kernel, which fix the total blocks
    # launched and manually simulates the main loop (blockIdx) via following 
    # main loop. This enables more controllable performance tuning!

    # Moreover, instead of using standard torch.Tensor, here we use pointers 
    # as input and manage them manually. This is because in communication, 
    # tensors are remotely other than locally (so only ptrs are available).
    sendbuff = sendbuff.to(tl.uint64).to(tl.pointer_type(tl.uint8)) # uint8 := byte
    recvbuff = recvbuff.to(tl.uint64).to(tl.pointer_type(tl.uint8)) # uint8 := byte
    block_start = tl.program_id(0) * BLOCK_BYTE
    while block_start < nbyte:
        # First create offsets and masks according to Triton Tutorial
        offsets = block_start + thread_id(0) * VECTORIZED_BYTES
        if offsets < nbyte:
            # For throughtput, we use 128bit instruction (largest for CUDA)
            # to load data from sendbuff (remote) to 2 u64 registers v0, v1 
            # NOTE communication through NvLink happens here
            (v0, v1) = load128(sendbuff + offsets)
            # Store the data from registers to local recvbuff on DRAM
            store128(recvbuff + offsets, v0, v1)
        # Advance block_start for the next loop
        block_start += tl.num_programs(0) * BLOCK_BYTE

# Other than kernel, we have some metadata to manage. For example, sendbuff
# is a tensor.data_ptr() in sender process, which is not visible to receiver.
# For simplicity, let's use MPI Collectives to help us manage these.

def send(buff: torch.Tensor, peer: int):
    # As we choose a receiver-oriented approach
    send_ptr(buff.data_ptr(), peer)
    # don't need to launch kernel, just leave!

def recv(buff: torch.Tensor, peer: int, profile = True):
    # try receive the sendbuff from sender process
    sendbuff: int = recv_ptr(peer) # only data_ptr received
    # Now it's time for performance tuning! NOTE see common.py for details
    grid = (NUM_BLOCKS.value, )
    BLOCK_BYTE = (NUM_WARPS * WARP_SIZE * VECTORIZED_BYTES).value
    # Finally launch the kernel to finish the p2p communication
    count = buff.numel() * buff.itemsize
    if profile: 
        torch.cuda.cudart().cudaProfilerStart()
    recv_kernel[grid](sendbuff, buff.data_ptr(), count, BLOCK_BYTE, 
        num_warps=NUM_WARPS.value) # num_warps is a Triton intrinsic
    if profile: 
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()


# finally let's do a testing, won't be executed if you import above function
if __name__ == "__main__":
    rank = MPI.COMM_WORLD.Get_rank()
    # torch.set_default_device(f"cuda:{get_gpu(rank)}") # set default device
    set_gpu(rank) 
    buff = torch.empty((TEST_COUNT, ), dtype=torch.float16) 
    # use rank = 0 as sender
    if rank == 0: buff.random_() # fill some random data for testing
    # run proposed online interface 
    send(buff, 1) if rank == 0 else recv(buff, 0)
    # move data to CPU via MPI for comparison
    if TEST_MODE:
        import torch.distributed as dist
        # These two are just required for torch.distribtued
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend="nccl", rank=rank, 
                                world_size=MPI.COMM_WORLD.Get_size())
        if rank == 0:
            dist.send(buff, 1)
        elif rank == 1:
            test_buff = torch.empty_like(buff) # need another buffer to test
            dist.recv(test_buff, 0)
            print(f"# rank {rank}, equal: {torch.equal(buff, test_buff)}")
            print(buff)
            print(test_buff)

# Final Notes: We ignore many topics in networking like sync/correct/order, as
# 1. NvLink will help us maintain most of correctness and atomicity
# 2. We have NO WAY to do so, this is GPU with little OS and NIC support
# What's Next? What if there's no p2p access and we need some GPU as jumphost?
#   See protocol.py and 2_sendrecv_indirect.py for these more general cases!