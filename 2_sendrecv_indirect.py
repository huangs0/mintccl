import os
from mpi4py import MPI
import torch
import triton
import triton.language as tl

from common    import *
from primitive import *
from topology  import *
from protocol  import *

# Now we're exploring a more generic setup: What if no direct P2P Access?
# Unfortunately it's a common setup for most 4/8 GPUs cluster without NvSwitch
# there would be some topology (use `nvidia-smi topo -m` to check
# Let's started by the simplest topology 0 <--> 1 <--> 2. In such a case, when
# GPU 0 want to send a tensor to GPU 2, we need to use GPU 1 as the jump host.

# Now let's rethink our previous solution (p2p.py). It's clear that now we need
# to launch kernel twice (0->1 and 1->2) with a syncrhonization in between.
# But this time the solution is not perfect as there's not overlapping between
# 0->1 and 1->2. For example, if 0->1 finish some part (like 1%), then 1->2 can
# start working on these 1% to save the time. We need another solution!

# A straightforward solution can be launching the kernel simutaneously for both
# 0->1 and 1->2, so when GPU 1 reads from0, it writes to DRAM visible to GPU 2
# so GPU 2 can reads from GPU 1. However, this raise another syncrhonization 
# issue: GPU 1's DRAM is always visible to GPU 2, how GPU 2 knows whether GPU 1
# has read GPU 0 and writes to DRAM or it's original dirty bits?
# Network Community's answer is protocol, i.e., adding headers (check bits) to
# identify the datagram, such as IP. Similarly, NCCL also do this (LL128) that
# adding a flag (sequence number) per 128 bytes send (128byte is NvLink unit).

# So the solution sounds simple as every GPU except src continuously read the 
# DRAM of ancestor in the chain (recv read send) and check if bits is correct
# However, this will create significant traffic on NvLink because everyone is
# actively quering if its bit is ready! In such a case, a better solution is 
# to use the sending (the one we give up in p2p), i.e., every sender send bits 
# to receiver and receiver check the bits on the local DRAM but not NvLink

# Now let's write a kernel to implement these ideas. 

# NOTE for details in LL128, see protocol.py. Buffers must be 240bytes aligned
# here we fuse send and recv in one kernel as only one can be launched
@triton.jit
def sendrecv_kernel(recvbuff,   # buffer to recv, local,  src has no recvbuff
                    sendbuff,   # buffer to send, remote, dst has no sendbuff
                    inputbuff,  # buffer for input, only applicable to src
                    outputbuff, # buffer for output, only applicable to dst
                    count):      # total number of bytes to send
    # Let's started with `flag`, an unique id of the message used to indicate
    # the arrival of a message (synchronization). Here let's use a naive soln
    # , a simple auto-incremental index. This is only item of the header!
    # The flag (8 byte) will be appended at the end of 120 byte to form LL128
    # And these 128 bytes is issued by 8 thread (work_group) so each thread
    # do 16 byte (st.global.v2.u64) and only last thread manage the flag.
    # In total, LL128 enjoys 93.75% bandwidth utilization without system sync.
    pid, tid = tl.program_id(0), thread_id(0)
    # Here we flattenly arange flag per workgroup, and 
    # NOTE flag can not be 0 as mostly we init everything to 0, set to 0 will
    #      cause 0th group fail. (And we left 0 for another usage :)
    flag = ((pid * WORKGROUPS_PER_BLOCK + tid // WORKGROUP_SIZE) * ITERS_PER_SEND + 1).to(tl.uint64)
    # And here we use a performance trick to send 2 message per thread in each
    # loop, so flagThread also use ld128 instead of ld64 (avoid stall cycles)
    # This is why there's v0, v1, v2, v3 four u64 register here (2 per loop)
    boundary = (count // BYTES_LL128_RAW).to(tl.uint64) # total number of flag
    src = inputbuff  != NULL
    dst = outputbuff != NULL 
    # Just to convince Triton that it's a ptr, no computation need actually
    sendbuff   = sendbuff.to(tl.uint64).to(tl.pointer_type(tl.uint8)) 
    recvbuff   = recvbuff.to(tl.uint64).to(tl.pointer_type(tl.uint8)) 
    inputbuff  = inputbuff.to(tl.uint64).to(tl.pointer_type(tl.uint8))
    outputbuff = outputbuff.to(tl.uint64).to(tl.pointer_type(tl.uint8)) 

    while flag < boundary:
        # As header takes 8 bytes, we maintain two offset aligned to work group
        offset_raw = (flag - 1) * BYTES_LL128_RAW # offset to raw data without header
        offset_msg = (flag - 1) * BYTES_LL128_MSG # offset to messages with header

        # Receiving Phase
        if src: # src need to load and encode raw bytes to LL128, so offset_raw
            v0, v1, v2, v3 = load_ll128(inputbuff + offset_raw, flag)
        else:   # Everyone not src just receive LL128 message, so offset_msg
            # NOTE Synchronization details in recv_ll128!!!
            v0, v1, v2, v3 = recv_ll128(recvbuff + offset_msg, flag)
        
        # Sending Phase
        if dst: # dst need to decode and save LL128 to raw, so offset_raw
            save_ll128(outputbuff + offset_raw, v0, v1, v2, v3) 
        else:   # others only need to send LL128 to next, so offset_msg
            send_ll128(sendbuff + offset_msg, v0, v1, v2, v3)
        
        # advance flag with total num_blocks * 2 (iters per loop)
        flag += NUM_BLOCKS * WORKGROUPS_PER_BLOCK * ITERS_PER_SEND


def send(buff: torch.Tensor, peer: int):
    # Similar to most distributed systems, let's choose a master node like 0
    comm = MPI.COMM_WORLD
    count = buff.numel() * buff.itemsize
    comm.send({'data_ptr': buff.data_ptr(), 'count': count, 'dst': peer}, dest=0, tag=0)


def recv(buff: torch.Tensor, peer: int):
    # Similarly let's also send these metadata instead of directly launch kernel
    # this is because we have more steps to do than simply issue the buffer
    comm = MPI.COMM_WORLD
    count = buff.numel() * buff.itemsize
    comm.send({'data_ptr': buff.data_ptr(), 'count': count, 'src': peer}, dest=0, tag=1)


# Here we encountered a problem that there might be GPU without calling send()
# / recv() need to launch kernel as jumphost, so we formulate a sync() as 
# a function that must be called by all like cuDeviceSyncrhonize()
def sync():
    comm = MPI.COMM_WORLD
    comm.barrier() # Make sure messages from send and recv is sent
    rank = comm.Get_rank()
    # now try to find a path from src to dst 
    if rank == 0:
        # first try to listen for send and recv message
        sendStatus = MPI.Status()
        sendInfo = comm.recv(source=MPI.ANY_SOURCE, status=sendStatus, tag=0)
        recvStatus = MPI.Status()
        recvInfo = comm.recv(source=MPI.ANY_SOURCE, status=recvStatus, tag=1)
        # do checking to assure src and dst matches and size mathces
        if VERBOSE: print(sendInfo, recvInfo)
        assert sendInfo['dst']   == recvStatus.Get_source() # rank of src
        assert recvInfo['src']   == sendStatus.Get_source() # rank of dst
        assert sendInfo['count'] == recvInfo['count']
        # then try to build the route
        # route = find_route(recvInfo['src'], sendInfo['dst'])
        route = [0, 1]
        msg = {'src': recvInfo['src'], 'dst': sendInfo['dst'],
               'src_ptr': sendInfo['data_ptr'], 'dst_ptr': recvInfo['data_ptr'],
               'route': route, 'count': sendInfo['count']}
        comm.bcast(msg, root=0)
    else:
        # listen to master broadcasting the route 
        msg = comm.bcast(None, root=0)
        route: Route = msg['route']
        count:  int  = msg['count']
    # now all nodes on the link prepare message buffer and 
    recvbuff, sendbuff = NULL.value, NULL.value # for scope and default value
    for i in range(1, len(route)):
        if rank == route[i]: # selected as jumphost, prepare the buffer!
            # NOTE message buffer is larger due to header size
            recvbuff = torch.zeros(
                (count // BYTES_LL128_RAW * BYTES_LL128_MSG,), 
                dtype=torch.uint8).data_ptr() # want ptr only
            # then broadcast message to the ancestor 
            send_ptr(recvbuff, route[i - 1])
    for i in range(0, len(route) - 1):
        if rank == route[i]:
            sendbuff = recv_ptr(route[i + 1])
    # wait until exchanging intermediate buffer finished
    comm.Barrier()
    if rank in route: # now all nodes in route launch kernel for communication 
        # if src, use raw buffer as recvbuff, else use self buffer instead
        inputbuff  = msg['src_ptr'] if rank == msg['src'] else NULL.value
        outputbuff = msg['dst_ptr'] if rank == msg['dst'] else NULL.value
        if VERBOSE: print(rank, recvbuff, sendbuff, inputbuff, outputbuff)
        count = msg['count']
        # Launch the kernel to transmit the data
        # torch.cuda.cudart().cudaProfilerStart()
        sendrecv_kernel[(NUM_BLOCKS.value, )](recvbuff, sendbuff, 
            inputbuff, outputbuff, count, num_warps=NUM_WARPS.value)
        # torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__": 
    # NOTE launch with V100/A100/H100/B100 x4/x8 Machine
    rank = MPI.COMM_WORLD.Get_rank()
    set_gpu(rank)
    # Here we use two env var to easily configure sender and receiver
    src, dst = int(os.getenv("SRC_TEMP", 0)), int(os.getenv("DST_TEMP", 1))
    # buff = torch.empty((TEST_COUNT, ))
    if rank == src:
        buff = torch.randn((TEST_COUNT, ))
        send(buff, dst)
    elif rank == dst:
        buff = torch.empty((TEST_COUNT, ))
        recv(buff, src)
    # NOTE everyone must sync() even not send/recv, communication happens here
    # with Timer(rank=rank) as t:
    sync()
    
    if TEST_MODE:
        import torch.distributed as dist
        # These two are just required for torch.distribtued
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend="nccl", rank=rank, 
                                world_size=MPI.COMM_WORLD.Get_size())
        if rank == src:
            dist.send(buff, dst)
        elif rank == dst:
            nccl_buff = torch.empty_like(buff) # need another buffer to test
            dist.recv(nccl_buff, src)
            print(f"# rank {rank}, equal: {torch.equal(buff, nccl_buff)}")
            print("# head")
            print(buff[:16])
            print(nccl_buff[:16])
            print("# tail")
            print(buff[-16:])
            print(nccl_buff[-16:])