"""NCCL Topology and Routing Re-Implemented

Topology is a graph describing the connectivity of your cluster using the 
fundamental P2P Physical Link. We use cuda driver API (cudaDeviceCanAccessPeer
and cudaMemcpyPeerAsync) to profile and obtained the Topology, inspired by NCCL

Based on Topology, we can locate the "optimal" data paths for MPI Collective, 
including Ring and Tree, basically a series of Graph Algorihtm practices :)

TODO Tree https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4
NOTE All functions inside except find_plan() is costly and shall be called
     ONLY by the root/master and broadcasted to rests via MPI!
ACKNOWLEDGEMENT: Thanks DeepSeek-R1 for these algorithms"""

import heapq # helpful for graph based algorithm like Dijkstra, A*
import cuda.bindings.runtime as cudart # CUDA Runtime API Binding

Topology = dict[int, list[tuple[int, float]]] # src, (dst, bandwidth), sorted 
Route = list[int] 
Plan = tuple[int, int] # prev, next


def find_route(src: int, dst: int, topo: Topology = None, shortest: bool = False) -> Route:
    """Find the optimal route (i.e., shortest and widest) for src -> dst

    By default, we recognize widest (largest bandwidth) path as optimal route,
    except direct path exists. Use `shortest=True` to select shortest path"""

    if topo is None: topo = get_topology() # Search topology if not given
    if dst in topo[src]: return [src, dst] # First check if direct link exists
    # Widest Bandwidth Problem like BFS, Basically a Leetcode :(
    max_bw = {n: 0 for n in topo.keys()}  # maximum bandwidth till here
    prev = {n: None for n in topo.keys()} # last node of maximum bw arrive n
    max_bw[src] = float('inf')   # start condition, local has infinite bw
    heap = [(src, -max_bw[src])] # the max heap started 
    while heap:
        u, neg = heapq.heappop(heap)    # heapq is optimized than manual
        cur_bw = -neg
        if u == dst: break              # reach the end, finished
        if cur_bw < max_bw[u]: continue # won't select a slower link
        for v, bw in topo[u]:
            new_bw = min(cur_bw, bw)    # state transition is taking min
            if new_bw > max_bw[v]:      # select new route if bw is larger
                max_bw[v] = new_bw
                prev[v] = u
                heapq.heappush(heap, (v, -new_bw))
    path = [] # holder for path
    cur = dst # started from dst to traverse back
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    # if reached src, return the path, else return [] (means not searched
    return path[::-1] if path and path[-1] == src else []


def find_ring(topo: Topology = None) -> Route:
    """Find the BEST ring for collective communication algorithm
    
    Here we use a priority-based BFS algorithm to explore graph and compare
    every valid ring (all nodes visited + return to start) by min bandwidth
    of all the edges forming the link
    
    TODO There might be two independet ring in V100/A100/H100x8 cluster"""
    
    if topo is None: topo = get_topology() # get topology if not given
    # State Tuple: current_node, path, min_bw
    heap: list[tuple[int, tuple[int, list[int], float]]] = []
    # start with 0, started from any node is the same
    heapq.heappush(heap, (0, (0, [0], float('inf'))))
    best_cycle, best_bw = [], 0

    while heap:
        _, (u, path, min_bw) = heapq.heappop(heap)

        if len(path) == len(topo):  # Terminating if cover all
            for (v, bw) in topo[u]: # and
                if v == path[0]:    # return to the start
                    final_bw = min(min_bw, bw)
                    if final_bw > best_bw:
                        best_bw = final_bw
                        best_cycle = path + [path[0]]
            continue # anyway, get away, ignore this

        for v, edge_bw in topo[u]:
            new_min = min(min_bw, edge_bw)
            if len(path) >= 2 and v == path[-2]: continue # avoid turn over
            if v in path: continue # avoid not expanding paths
            new_path = path + [v]
            # Priority is Coverage First + Bandwidth Priority
            priority = len(new_path) * 10 + new_min
            heapq.heappush(heap, (-priority, (v, new_path, new_min)))
    # NOTE root appears twice, 1st and last, in best_cycle
    return best_cycle if best_cycle else []


def get_p2p_speed(src, dst, size=33554432) -> float: # default size 64MB
    """Measure the p2p copy speed (GB/s) based on cudaMemcpyPeerAsync
    
    NOTE Based on the following example recommended by NCCL developers:
    https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/p2pBandwidthLatencyTest"""
    # enable p2p access
    cudart.cudaSetDevice(src)
    cudart.cudaDeviceEnablePeerAccess(dst, 0)
    # allocate memory 
    _, src_ptr = cudart.cudaMalloc(size)
    cudart.cudaSetDevice(dst)
    _, dst_ptr = cudart.cudaMalloc(size)
    # create event
    _, stream = cudart.cudaStreamCreate()
    _, start_event = cudart.cudaEventCreate()
    _, stop_event  = cudart.cudaEventCreate()
    # benchmark p2p link speed
    cudart.cudaEventRecord(start_event, stream)
    _ = cudart.cudaMemcpyPeerAsync(dst_ptr, dst, src_ptr, src, size, stream)[0]
    cudart.cudaEventRecord(stop_event, stream)
    cudart.cudaStreamSynchronize(stream)
    _, elapsed_ms = cudart.cudaEventElapsedTime(start_event, stop_event)
    # destroy everything
    cudart.cudaFree(src_ptr)
    cudart.cudaFree(dst_ptr)
    cudart.cudaStreamDestroy(stream)
    cudart.cudaEventDestroy(start_event)
    cudart.cudaEventDestroy(stop_event)
    # return throughput in GB/s
    return (size / (elapsed_ms / 1000.0)) / 1e9


def get_topology(topo: str = "topo.json") -> Topology:
    """Load or Explore the Current Intra-Node Point-to-Point Topology"""
    import os, json # to persist topology instead of searching again and again
    if os.path.exists(topo):
        topo: Topology = json.load(open(topo, "r"))
        topo = {int(src): items for src, items in topo.items()} 
    else:
        error, device_count = cudart.cudaGetDeviceCount()
        assert error == cudart.cudaError_t.cudaSuccess
        topo: Topology = {src: [] for src in range(device_count)}
        for src in range(device_count):
            for dst in range(src + 1, device_count):
                error, can_p2p = cudart.cudaDeviceCanAccessPeer(src, dst)
                if can_p2p: # only testing p2p devices
                    # Further do the bandwidth measurement
                    bandwidth = get_p2p_speed(src, dst)
                    # NOTE Following treatment is based on full bidirectional link
                    # like NvLink, PCIe or CXL might have differnet property
                    topo[src].append((dst, bandwidth))
                    topo[dst].append((src, bandwidth))
        topo = {src: sorted(items, reverse=True) for src, items in topo.items()}
        json.dump(topo, open("topo.json", "w")) # persist to disk
    return topo