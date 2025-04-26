import os
from mpi4py import MPI
import torch
import triton
import triton.language as tl

from common    import *
from primitive import *
from topology  import *
from protocol  import *

# Our previous version of sendrecv is has an dramatic problem, it requires a
# very huge buffer (size // BYTES_LL128_RAW * BYTES_LL128_MSG,) for each jump
# And our goal is to reduce the buffer before we move to complex collectives.

# An simple but robust solution is to use a constant size buffer per node, and
# each node write next one's buffer till full, and then WAIT until next node
# processing the buffer before continue writing. This idea is very similar to
# batch_size used in deep learning i.e., BUFF_SIZE (configured in common.py)
# defines the parallelism level and shall be set according to system capacity.

# Kernel defn is unchanged, recvbuff and sendbuff now having const BUFF_SIZE
@triton.jit
def sendrecv_buff_kernel(recvbuff,  # local  buffer to recv, src don't have
                         sendbuff,  # remote buffer to send, dst don't have
                         inputbuff, # buffer for input, only applicable to src
                         outputbuff,# buffer for output,only applicable to dst
                         count):    # no.bytes to send, count >> BUFF_SIZE
    # Similarly let's have the flag definition as non-buffer version, because
    # we want flag as unique identifier of data chunk in inputbuff/outputbuff
    pid, tid = tl.program_id(0), thread_id(0)
    flag = ((pid * WORKGROUPS_PER_BLOCK + tid // WORKGROUP_SIZE) 
                * ITERS_PER_SEND + 1).to(tl.uint64)
    boundary = (count // BYTES_LL128_RAW).to(tl.uint64)
    # Some basic bool, need to save it before converted to ptr or Triton fails
    src = inputbuff  != NULL # or recvbuff == NULL
    dst = outputbuff != NULL # or sendbuff == NULL
    # Just to convince Triton that it's a ptr, no computation need actually
    sendbuff   = sendbuff.to(tl.uint64).to(tl.pointer_type(tl.uint8)) 
    recvbuff   = recvbuff.to(tl.uint64).to(tl.pointer_type(tl.uint8)) 
    inputbuff  = inputbuff.to(tl.uint64).to(tl.pointer_type(tl.uint8))
    outputbuff = outputbuff.to(tl.uint64).to(tl.pointer_type(tl.uint8)) 
    # Outer loop is unchanged as total communication is the same
    while flag <= boundary: # = because flag starts with 1
        # Now offset_msg is aligned with buffer and initialize per loop
        offset_msg = ((pid * WORKGROUPS_PER_BLOCK + tid // WORKGROUP_SIZE) 
                        * ITERS_PER_SEND * BYTES_LL128_MSG).to(tl.uint64)
        # Now with buffer, we need more sync, first wait till buffer clear
        if not src: # Every recv node inform previous it's ready to receive
            # NOTE fix: https://github.com/triton-lang/triton/issues/5768
            tmp = post_ll128(recvbuff + offset_msg) 
        if not dst: # Every send node shall wait for next node till ready
            tmp = wait_ll128(sendbuff + offset_msg)
        
        # Let's replace original one pass with a loop whose length ~ BUFF_SIZE
        for _ in range(WORKGROUP_ITERS): # see protocol.py for WORKGROUP_ITERS
            if flag <= boundary: # need a more detailed boundary checking
                # offset_raw is still the same
                offset_raw = (flag - 1) * BYTES_LL128_RAW # offset to raw data 
                
                # offset_msg = (flag_msg - 1) * BYTES_LL128_MSG
                if src: # load use offset_raw
                    v0, v1, v2, v3 = load_ll128(inputbuff + offset_raw, flag)
                else:   # recv use offset_msg
                    v0, v1, v2, v3 = recv_ll128(recvbuff  + offset_msg, flag)
                
                # Sending Phase
                if dst: # save use offset_raw
                    save_ll128(outputbuff + offset_raw, v0, v1, v2, v3) 
                else:   # send use offset_msg
                    send_ll128(sendbuff   + offset_msg, v0, v1, v2, v3)
            # advance flag with total num_blocks * 2 (iters per loop)
            flag += NUM_BLOCKS * WORKGROUPS_PER_BLOCK * ITERS_PER_SEND
            offset_msg += NUM_BLOCKS * WORKGROUPS_PER_BLOCK * ITERS_PER_SEND * BYTES_LL128_MSG


# Following code is the same as sendrecv.py, other than recvbuff's creation
def send(buff: torch.Tensor, peer: int):
    # Similar to most distributed systems, let's choose a master node like 0
    comm = MPI.COMM_WORLD
    count = buff.numel() * buff.itemsize
    if VERBOSE: print(buff.device, buff.data_ptr())
    comm.send({'data_ptr': buff.data_ptr(), 'count': count, 'dst': peer}, dest=0, tag=0)


def recv(buff: torch.Tensor, peer: int):
    # Similarly let's also send these metadata instead of directly launch kernel
    # this is because we have more steps to do than simply issue the buffer
    comm = MPI.COMM_WORLD
    count = buff.numel() * buff.itemsize
    if VERBOSE: print(buff.device, buff.data_ptr())
    comm.send({'data_ptr': buff.data_ptr(), 'count': count, 'src': peer}, dest=0, tag=1)


# Here we encountered a problem that there might be GPU without calling send()
# / recv() need to launch kernel as jumphost, so we formulate a sync() as 
# a function that must be called by all like cuDeviceSyncrhonize()
def sync():
    comm = MPI.COMM_WORLD
    comm.barrier() # Make sure messages from send and recv is sent
    rank, world_size = comm.Get_rank(), comm.Get_size()
    # now try to find a path from src to dst 
    if rank == 0:
        # first try to listen for send and recv message
        sendStatus = MPI.Status()
        sendInfo = comm.recv(source=MPI.ANY_SOURCE, status=sendStatus, tag=0)
        recvStatus = MPI.Status()
        recvInfo = comm.recv(source=MPI.ANY_SOURCE, status=recvStatus, tag=1)
        if VERBOSE: print(sendInfo, recvInfo) # debugging
        # assure src and dst matches and size mathces
        assert sendInfo['dst']   == recvStatus.Get_source() # rank of src
        assert recvInfo['src']   == sendStatus.Get_source() # rank of dst
        assert sendInfo['count'] == recvInfo['count']
        # then try to build the route
        # route = find_route(recvInfo['src'], sendInfo['dst'])
        route = [0, 1]
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
    # now all nodes on the link prepare message buffer and 
    recvbuff, sendbuff = NULL.value, NULL.value
    for i in range(1, len(route)):
        if rank == route[i]: # selected as jumphost, prepare the buffer!
            # NOTE change buffer to BUFF_SIZE configured in common.py
            recvbuff = torch.zeros(BUFF_SIZE, dtype=torch.uint8)
            recvbuff = recvbuff.data_ptr()
            send_ptr(recvbuff, route[i - 1])
    for i in range(0, len(route) - 1):
        if rank == route[i]:
            sendbuff = recv_ptr(route[i + 1])
    # wait until exchanging intermediate buffer finished
    comm.Barrier()
    if rank in route: # now all nodes in route launch kernel for communicatio
        src = rank == msg['src']
        dst = rank == msg['dst']
        inputbuff  = msg['src_ptr'] if src else NULL.value
        outputbuff = msg['dst_ptr'] if dst else NULL.value
        count: int = msg['count']
        if VERBOSE: print(rank, recvbuff, sendbuff, inputbuff, outputbuff)
        # Launch the kernel to transmit the data
        sendrecv_buff_kernel[(NUM_BLOCKS.value, )](recvbuff, sendbuff, 
            inputbuff, outputbuff, count, num_warps=NUM_WARPS.value)

if __name__ == "__main__": 
    # NOTE launch with V100/A100/H100/B100 x4/x8 Machine
    rank = MPI.COMM_WORLD.Get_rank()
    set_gpu(rank)
    # Here we use two env var to easily configure sender and receiver
    src, dst = int(os.getenv("SRC_TEMP", 0)), int(os.getenv("DST_TEMP", 1))
    if rank == src:
        buff = torch.randn((TEST_COUNT, ))
        send(buff, dst)
    elif rank == dst:
        buff = torch.empty((TEST_COUNT, ))
        recv(buff, src)
    # NOTE everyone must call sync(), communication happens here
    sync()
    # NOTE testing requries cupy and cupyx.distribtued
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
