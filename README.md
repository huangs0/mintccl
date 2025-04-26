# 🌱 MinTCCL: MINimalistic Triton Communication Collective Library

MinTCCL is an attempt on using pure Python (Triton + PyTorch) to implement a minimalistic [NCCL](https://github.com/NVIDIA/nccl) in ~1,200 line of Python. 
The author envision it as:
* a development repo that you can easily hack, deploy and fuse the collectives with other Triton kernels.
* an educational tutorial that you can read through (clean codes and comments) to learn how we gradually implement advanced AlltoAll starting from simplest SendRecv.

## Structure and How to use
The structure is too simple to use folders, the core lies in four files:
1. `common.py` implements common utilities and configurations via envariables.
2. `primitive.py` implements lacked device built-ins (like `syncwarp`) and 128-bit Ops.
3. `protocol.py` implements NCCL's LL128 protocol primitives, relies on `primitive.py`.
4. `topology.py` implements topology and routing algorithms, e.g., ring.

Based on these four modules, we can build MPI/NCCL Communication Collectives:
- [x] `1_sendrecv_direct.py`: simplest load/store for send/recv in direct link
- [x] `2_sendrecv_indirect.py`: use LL128 protocol to support indirect link (multi-hop)
- [x] `3_sendrecv_buff.py`: use buffer management to save memory usage
- [x] `4_broadcast.py`: 1st collective using ring algorithm
- [x] `5_reduce.py`: 1st computation and deadlocks preventions!
- [x] `6_allgather.py`: 1st rank-divided algorihtm, supporting spatial/temporal
- [x] `7_reducescatter.py`: 
- [ ] `8_allreduce.py`
- [ ] `9_alltoall.py` 

Each above collective can be used with `mpirun -np 2/4/8 python ...`.
To integrate into your project, simply copy the four core module and the colletivce you interested into your project, no installation need!

The author orders these collectives into a learning curve that each collective gradually introduces some new (but not too much) knowledge.

## How it differs from ...
Compared with NCCL, it cut off:
1. Inter-Node Communication with RDMA
2. Intra-Node Communication with PCIe (currently NvLink ONLY for enabling P2P)

Compared with other attempts in Triton + Communication, e.g., [`SymmetricMemory`(PyTorch)](https://dev-discuss.pytorch.org/t/pytorch-symmetricmemory-harnessing-nvlink-programmability-with-ease/2798), it add more:
1. Protocol and Topology support: with full LL128 protocol, indirect link is supported and you don't need NvSwitch!
2. Usage support: only four dependencies, [`cuda-python`](https://nvidia.github.io/cuda-python/cuda-bindings/latest/install.html)/[`mpi4py`](https://mpi4py.readthedocs.io/en/stable/)/`torch`/`triton`, no custom build or bindings!

## How it works
The fundamental of (NCCL and MinTCCL) is that NvLink can be programmed via basic `ld`/`st` primitive (of PTX)
on machines with [Unified Memory](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/) 
and [GPU P2P Access](https://developer.nvidia.com/gpudirect)
support that can map GMEM address of the GPU to another. 
As there's no difference between reading local GMEM, Triton, theoretically, can also be used for communications and MinTCCL tries to prove this. 

## Roadmap
* Collectives
  - [ ] AllReduce
  - [ ] AlltoAll
  - [ ] Grouped P2P
* Topology
  - [x] Ring
  - [ ] Tree / Binary Tree
  - [ ] Hypercube
* Performance Tuning
  - [ ] Fine-grained control of SM Layout like NCCL
* Arithemetics Types for ReduceOps
  - [x] FP16 
  - [ ] Generic Support of FP32/BF16/FP8

## Known Issues
1. On 8-GPU machines, MinTCCL will fail into deadlock, seems a problem of flag maangements, still debugging.
