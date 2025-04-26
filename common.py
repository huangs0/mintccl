"""Common Definition and Configuration"""

import os
import time
import torch
import triton.language as tl
import cuda.bindings.runtime as cudart
from mpi4py import MPI

def set_gpu(rank: int):
    """Set Default Device by MPI Rank and optionally CUDA_VISIBLE_DEVICES"""
    if os.getenv("CUDA_VISIBLE_DEVICES"):
        gpus = os.getenv("CUDA_VISIBLE_DEVICES").split(",")
        torch.set_default_device(f"cuda:{gpus[rank]}")
    else: # by default, just try the default mapping
        assert rank < torch.cuda.device_count(), f"rank {rank} > available gpu"
        torch.set_default_device(f"cuda:{rank}")

# For safely exchange pointers between processes (torchrun or mpirun)
# NOTE ONLY exchange between consecutive (p2p) devices for peer access

def send_ptr(ptr: int, peer: int):
    """Safely Send a PyTorch Tensor Pointer to Peer"""
    err, handle = cudart.cudaIpcGetMemHandle(ptr)
    assert err == cudart.cudaError_t.cudaSuccess, "cudaIpcMemHandle failed"
    MPI.COMM_WORLD.send(handle.reserved, dest=peer)


def recv_ptr(peer: int) -> int:
    """Safely Recv a PyTorch Tensor Pointer to Peer
    NOTE will also open peer access between rank and peer"""
    handle_bytes = MPI.COMM_WORLD.recv(source=peer)
    handle = cudart.cudaIpcMemHandle_t()
    handle.reserved = handle_bytes
    err, ptr = cudart.cudaIpcOpenMemHandle(
        handle, cudart.cudaIpcMemLazyEnablePeerAccess)
    assert err == cudart.cudaError_t.cudaSuccess, "cudaIpcMemHandle failed"
    return ptr


class Timer:
    """A simple Timer supporting CPU and GPU timer and Python with statement"""
    def __init__(self, device: bool = False, rank: int = None) -> None:
        self.rank  = rank
        self.device = device

    def __enter__(self):
        if self.device:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
        else:
            self.start = time.perf_counter_ns()

    def __exit__(self, exc_type, exc_value, exc_traceback): 
        elapsed: float = 0.0
        if self.device:
            self.end.record()
            torch.cuda.synchronize() # wait until event finished
            elapsed = self.start.elapsed_time(self.end) / 1e3
        else:
            self.end = time.perf_counter_ns()
            elapsed = (self.end - self.start) / 1e9
        if self.rank is not None:
            print(f"# rank: {self.rank} time: {elapsed}s")
        else:
            print(f"# time {elapsed}s")


# Macros for convenience
NULL = tl.constexpr(0) # C NULL
properties = torch.cuda.get_device_properties(torch.cuda.current_device())
CUDA_ARCH = tl.constexpr(properties.major * 100 + properties.minor * 10) 


# PERFORMANCE TUNING
# NOTE set all these to tl.constexpr as they might be referenced within kernel
#      use .value to dereference them to int

# First is Block-Level Parallelism. As we use the persistent kernel,
# no.parallel copies is exactly no.kernel launchs, i.e., gird! And this 
# value shall be bounded by the bandwidth of connectviity like NvLink
# A larger value cannot accelerate the kernel as bound is on bandwidth
# A common estimation is 1 block for 5GB/s
NUM_BLOCKS = tl.constexpr(int(os.getenv("MINTCCL_NUM_BLOCKS", 16)))

# Second is Instruction-Level Parallelism. First, let's use vectorizead 
# instruction (SIMD), like 128bit, so every thread can read 16/itemsize 
# elements in one time. And as reading remote memory is time consuming, 
# let's launch a larger block for higher instruction-level parallelism.
# NOTE must be power of 2 or Triton complains
NUM_WARPS = tl.constexpr(int(os.getenv("MINTCCL_NUM_WARPS", 16)))

# Third is the buffer size, which determines no.elements send per loop.
# Larger buffer GENERALLY leads to less memory access and better performance.
# Default value is 4194304 (4MB, same as NCCL), must be power of 2
# REF https://github.com/NVIDIA/nccl/issues/1252
# REF https://github.com/NVIDIA/nccl/issues/559
# REF https://github.com/NVIDIA/nccl/issues/157
# REF https://github.com/NVIDIA/nccl/issues/353
BUFF_SIZE = tl.constexpr(int(os.getenv("MINTCCL_BUFFSIZE", 4194304)))

# and some systematic config that can not be changed
WARP_SIZE = tl.constexpr(32)        # hardware setup, don't modify it
VECTORIZED_BYTES = tl.constexpr(16) # 16 Bytes := 128 bits := .v4.b32/.v2.b64
SPINS_BEFORE_ABORT = tl.constexpr(1000000) # see NCCL_SPINS_BEFORE_CHECK_ABORT

# TESTING AND DEBUGGING
 
# Testing mode will compare the result with NCCL backend (`torch.distributed`)
# REF https://pytorch.org/docs/stable/distributed.html
# Details of testing is in each collective algorithm
TEST_MODE = bool(os.getenv("MINTCCL_TEST_MODE", 0))
# test count, by default is 64MB size and not typed, NOTE this is not typed
TEST_COUNT = int(os.getenv("MINTCCL_TEST_COUNT", 67108864))

# An env var to enable more informative message printing for loggin/debugging
VERBOSE = bool(os.getenv("MINTCCL_TEST_VERBOSE", False))