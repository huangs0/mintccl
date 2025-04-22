"""NCCL Protocol Re-Implementation

Protocol enables high efficient point-to-point communication (send/recv)
with enough synchronization and integrity check (if applicable)

Currently only LL128 protocol is supported:
* LL128:  A faster protocol using packet checkbit for synchronisation

Working on supporting:
* Simple: A protocol using system-level barrier like threadfence_system()

WARNING: Protocol-layer send/recv is topology-diagonistic, i.e., it requires
p2p connection between two GPU with NvLink/PCIe. On non-NvSwitch system, this
is likely to fail so for application, use higher-level API in p2p.py
"""

import triton 
import triton.language as tl 
from common    import * 
from primitive import * 

"""NCCL LL128 Protocol
There's no official documentation but following discussions and codes
* https://github.com/NVIDIA/nccl/issues/281
* https://github.com/NVIDIA/nccl/blob/master/src/device/prims_ll128.h

LL stands for Low Latency and 128 stands for 128Byte (32x4)
where it has 120Byte payload and 8Byte flag used for synchronization such that
no costly system-level memory barrier syncrhonization is need, so Low Latency

The basic mechanism of LL is: ```LL achieves lower latency than Simple for 
small messages by sending the flag along with the data. When we see the flag, 
the data is valid, so that we can avoid expensive memory barriers.```

NOTE this is applicable ONLY for NvLink with Memory Operation Ordered 128Byte
NOTE There're some differences between MintCCL's impl and NCCL's impl:
1. NCCL send 4 message per loop (8 regs) but we only send 2 message per loop
2. NCCL have more complicated flag initial value
"""

# Basic Protocol Constants
PAYLOAD_WORK_THREAD = tl.constexpr(16) # 16 bytes = 128 bit
PAYLOAD_FLAG_THREAD = tl.constexpr(8)  # rest 8 bytes reserved for flag
WORKGROUP_SIZE      = tl.constexpr(8)  # 8 threads, each 16 bytes so 128bytes
ITERS_PER_SEND      = tl.constexpr(2)  # NOTE Original NCCL Impl is 4
WORKGROUPS_PER_BLOCK = NUM_WARPS * WARP_SIZE // WORKGROUP_SIZE
BYTES_LL128_MSG = PAYLOAD_WORK_THREAD * WORKGROUP_SIZE
BYTES_LL128_RAW = PAYLOAD_WORK_THREAD * (WORKGROUP_SIZE - 1) + PAYLOAD_FLAG_THREAD

# Buffer is managed hierarchically block -> workgroup -> iterative
WORKGROUP_BUFFSIZE = BUFF_SIZE // (NUM_BLOCKS * WORKGROUPS_PER_BLOCK)      
WORKGROUP_ITERS = WORKGROUP_BUFFSIZE // (BYTES_LL128_MSG * ITERS_PER_SEND)

# NOTE all buffers used below, inputbuff/outputbuff/recvbuff/sendbuff, 
#      are pointers to be aligned to the workgroup!

@triton.jit
def load_ll128(inputbuff, flag):
    flag = flag.to(tl.uint64)
    # First 128Byte := 16 * 8, so let's order in 8 threads as a group, which
    # fit into the standard warp system, and we specially take the last thread 
    # of 8 as the flagThread responsible for adding Flag (8 Byte) and checking
    groupLane  = thread_id(0) % WORKGROUP_SIZE
    flagThread = groupLane == (WORKGROUP_SIZE - 1)
    # NOTE refer to func loadRegsBegin in nccl/src/device/prims_ll128.h
    if not flagThread:
        # for first 7 threads, let's load 128bit data into register
        # NOTE we use PTX Assembly u64 and we don't care about the real data 
        #      type as uint is raw bits without conversion
        v0, v1 = load128(inputbuff + groupLane * VECTORIZED_BYTES)
        # We also use BYTES_LL128_MSG because everyone include flagThread loads
        v2, v3 = load128(inputbuff + BYTES_LL128_MSG + groupLane * VECTORIZED_BYTES)
    else:
        # for the last thread, let's load 64bit data into register, but let's 
        # use the same load128 and manually ignore v1, this can avoid the 
        # thread instruction difference inside warp and stall cycle wasted.
        # NOTE NCCL's original impl is more complicated to shuffle two load
        #      but let's do it naively and hope Triton can optimize it :)
        v0, v1 = load128(inputbuff + groupLane * VECTORIZED_BYTES)
        # v2, v3 = ld_global_128() don't load because v2 and v3 are for flags
        # NOTE refer to loadRegsFinish in nccl/src/device/prims_ll128.h
        v2 = v1       # do some swapping to make v2 as message
        v1 = flag     # and make v1 as the 1st flag flag 
        v3 = flag + 1 # make v3 as the 2nd flag, register operation is fast!
    return v0, v1, v2, v3

@triton.jit
def save_ll128(outputbuff, v0, v1, v2, v3):
    groupLane  = thread_id(0) % WORKGROUP_SIZE
    flagThread = groupLane == (WORKGROUP_SIZE - 1)
    # NOTE refer to storeRegs in nccl/src/device/prims_ll128.h
    if not flagThread:
        store128(outputbuff + groupLane * VECTORIZED_BYTES, v0, v1)
        # We also use BYTES_LL128_MSG because everyone include flagThread saves
        store128(outputbuff + BYTES_LL128_MSG + groupLane * VECTORIZED_BYTES, v2, v3)
    else:
        # NOTE v1 = v2 is register operation, stall cycle shall be neglectable
        v1 = v2 # ignore original v1 value as it's flag not real data to save
        store128(outputbuff + groupLane * VECTORIZED_BYTES, v0, v1)
        # flag thread has half payload so no 2nd saving

@triton.jit
def recv_ll128(recvbuff, flag):
    # NOTE some modification here, NCCL check reload for both 
    # For receiving, let's also issue a flagThread
    groupLane  = thread_id(0) % WORKGROUP_SIZE
    flagThread = groupLane == (WORKGROUP_SIZE - 1)
    # No matter whom, just load the data from buffer
    v0, v1 = load128(recvbuff + groupLane * VECTORIZED_BYTES)
    # and flagThread need to check the flag consistency
    needReload = flagThread and (v1 != flag)
    # then we need to broadcast needReload to all threads in warp via any_sync
    # If not pass, just redo until it pass, a naive but effective spinlock!
    # NOTE here all threads under same warp must go together, limited by CUDA
    while any_sync(needReload):
        v0, v1 = load128(recvbuff + groupLane * VECTORIZED_BYTES)
        needReload = flagThread and (v1 != flag) # check again

    # And there's onething special, let's don't write registers back to any 
    # memory because recv_ll128 might be fuse with reduce operators, let's 
    # maintain another save_ll128 to write back the result :)
    
    # NOTE for consistency (operate on four value v0, v1, v2, v3), 
    # let's redo all the above to also read and return v2 and v3
    v2, v3 = load128(recvbuff + BYTES_LL128_MSG + groupLane * VECTORIZED_BYTES)
    needReload = flagThread and (v3 != flag + 1)
    while any_sync(needReload):
        v2, v3 = load128(recvbuff + BYTES_LL128_MSG + groupLane * VECTORIZED_BYTES)
        needReload = flagThread and (v3 != flag + 1) # check again
    return v0, v1, v2, v3

@triton.jit
def send_ll128(sendbuff, v0, v1, v2, v3):
    groupLane  = thread_id(0) % WORKGROUP_SIZE
    # Sending is basically a store operation, and we do twice for v0 to v3
    store128(sendbuff + groupLane * VECTORIZED_BYTES, v0, v1)
    store128(sendbuff + BYTES_LL128_MSG + groupLane * VECTORIZED_BYTES, v2, v3)

# NOTE comp_ll128 are used in reduce operation (reduce, reducescatter, allreduce)

@triton.jit
def comp_ll128(v0, v1, v2, v3, r0, r1, r2, r3): # all are uint64
    # NOTE for flag thread, we shall not add v1 and v3 as they're flags...
    # NOTE this is extended from `applyReduce` in nccl/src/device/prims_ll128.h
    groupLane  = thread_id(0) % WORKGROUP_SIZE
    flagThread = groupLane == (WORKGROUP_SIZE - 1)
    # There might be some diverged computation and stall cycle, but as all are
    # register operation, it shall be pretty fast
    if flagThread:
        v0, v1, v2, v3 = (SumFP16(v0, r0), r1, # v1 and r1 is flag, don't add
                          SumFP16(v2, r2), r3) # v3 and r3 is flag, don't add
    else:
        v0, v1, v2, v3 = (SumFP16(v0, r0), SumFP16(v1, r1), 
                          SumFP16(v2, r2), SumFP16(v3, r3))
    return v0, v1, v2, v3

# NOTE Following wait_ll128 and post_ll128 are used in buffered implementation 
#      name of post and wait are borrowed from semaprhore by Dijkstra, respect

@triton.jit
def wait_ll128(sendbuff): 
    # NOTE refer to waitSend and checkAbort in nccl/src/device/prims_ll128.h
    # A little bit tricky here, sendbuff is remote so load64 is through nvlink
    # This is costly but when waiting for result, NvLink is busy so for free
    groupLane = thread_id(0) % WORKGROUP_SIZE
    flagThread = groupLane == (WORKGROUP_SIZE - 1)
    offset = groupLane * VECTORIZED_BYTES + PAYLOAD_FLAG_THREAD
    flag = load64(sendbuff + offset)
    # Here is a tricky point, we want a value impossible(here) but predictable
    # If not ready, load(sendbuff) shall be flag of prev round and it shall 
    # never be set to correct "flag" or next recv_ll128 loss sync directly
    # for simplicity, we can use 0, the simple but impossible value, which:
    # 1. solve the problem of init, every buffer start to be clear (zeros)
    # 2. 0 is not a valid flag so won't appear normally!
    needWait = flagThread and flag != 0
    while any_sync(needWait):
        flag = load64(sendbuff + offset)
        needWait = flagThread and flag != 0
    return 0 # meaningless return for a Triton bug that jit must return sth...

@triton.jit
def post_ll128(recvbuff): 
    # NOTE refer to postRecv in nccl/src/device/prims_ll128.h
    groupLane = thread_id(0) % WORKGROUP_SIZE
    # flagThread = groupLane == (WORKGROUP_SIZE - 1)
    # actually only flag is fine but Triton has more optimization if full
    store64(recvbuff + groupLane * VECTORIZED_BYTES + PAYLOAD_FLAG_THREAD, 0)
    return 0 # meaningless return for a Triton bug that jit must return sth...