"""Primitives for Device Built-in and 128Bit Op, also some reduction op"""

from enum import Enum
import triton
import triton.language as tl

"""
Basic CUDA Primitives like __syncthreads() which are lacked in Triton because
Triton mostly automatically handle these. However, as Communciation have many
different design need, like more synchronization and protocol, we have to 
do more manual work. Following are equivalent of __device_builtin__ in cuda.h

NOTE There might be some weird `mov.u64 $0, 0;` in following asm, and this is
     due to a Triton "bug" that inline_asm_elementwise doesn't return anything 
     will be auto optimized aka removed :( and `mov.u64` is to prevent this...
     But `mov.u64` will be optimizead by PTXAS so no effect to real program!
NOTE Similarly `is_pure` is set to False to prevent Triton optimization...
"""

__AXIS_XYZ__ = {0: "x", 1: "y", 2: "z"}

@triton.jit
def thread_id(axis: tl.constexpr = 0):
    # Simplified threadIdx.x as we organize threads 1D only
    # Noted that tl.program_id(0) is blockIdx not threadIdx...
    if axis == 0:
        return tl.inline_asm_elementwise("mov.u32 $0, %tid.x;", "=r", args=[], 
                dtype=(tl.int32), is_pure=False, pack=1)
    elif axis == 1:
        return tl.inline_asm_elementwise("mov.u32 $0, %tid.y;", "=r", args=[], 
                dtype=(tl.int32), is_pure=False, pack=1)
    elif axis == 2:
        return tl.inline_asm_elementwise("mov.u32 $0, %tid.z;", "=r", args=[], 
                dtype=(tl.int32), is_pure=False, pack=1)

@triton.jit
def barrier_sync(name, nthreads):
    tl.inline_asm_elementwise(
        "barrier.sync.aligned 6, $1;\nmov.u64 $0, 0;",
        "=l,l,~{memory}", # TODO fix memory clobber 
        args=[name, nthreads],
        dtype=(),
        is_pure=True,
        pack=1,
    )

@triton.jit
def barrier_sync(name: tl.constexpr):
    tl.inline_asm_elementwise(
        "barrier.sync.aligned $1;\nmov.u64 $0, 0;",
        "=l,r,~{memory}", # TODO fix memory clobber 
        args=[name],
        dtype=(tl.uint64),
        is_pure=False,
        pack=1,
    )

@triton.jit
def syncwarp(mask: tl.constexpr = 0xFFFFFFFF):
    tl.inline_asm_elementwise(
        """bar.warp.sync  0xffffffff;
        mov.u64 $0, 0;""", "=l,~{memory}", 
        args=[], dtype=(tl.uint64), is_pure=False, pack=1
    )

@triton.jit
def syncthreads(name: tl.constexpr = 0): # remove __ to make Python import automatically
    tl.inline_asm_elementwise(
        "bar.sync $1;\nmov.u64 $0, 0;", "=l,r,~{memory}", 
        args=[name], dtype=(tl.uint64), is_pure=False, pack=1
    )

@triton.jit
def threadfence():
    tl.inline_asm_elementwise(
        "membar.gl;\nmov.u64 $0, 0;", "=l", 
        args=[], dtype=(tl.uint64), is_pure=False, pack=1
    )

# TODO a better __threadfence_system()
@triton.jit
def threadfence_system():
    tl.inline_asm_elementwise(
        "membar.sys;\nmov.u64 $0, 0;", "=l",
        args=[], dtype=(tl.uint64), is_pure=False, pack=1
    )

@triton.jit
def any_sync(pred, mask: tl.constexpr = 0xFFFFFFFF):
    # NVPTX don't support pred in asm, see https://forums.developer.nvidia.com/t/ptx-asm-constraint-letter-for-predicate/283337
    return tl.inline_asm_elementwise(
        """{
            .reg .pred p, q;
            setp.ne.b32  q, $1, 0;
            vote.sync.any.pred p, q, 0xFFFFFFFF;
            selp.u32     $0, 1, 0, p;
        }""", 
        "=r,r,~{memory}", 
        args=[pred], # 
        dtype=(tl.int32),   # NOTE use int1, i.e., pred
        is_pure=False,     # up to others
        pack=1
    ).to(tl.int1)

@triton.jit
def nanosleep(ns):
    return tl.inline_asm_elementwise(
        "nanosleep.u32 $0;", 
        "r", 
        args=[ns], 
        dtype=(), 
        is_pure=True, # a pure function
        pack=1
    )

@triton.jit
def trap(): # raise bugs
    return tl.inline_asm_elementwise(
        "trap;\nmov.u64 $0, 0;", "=l", args=[], dtype=(tl.uint64), is_pure=True, pack=1
    )

@triton.jit
def cvta_generic_to_shared(generic):
    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cvta-generic-to-shared
    return tl.inline_asm_elementwise(
        "cvta.to.shared.u64 $0, $1;",
        "=l, l, ~{memory}", # TODO fix memory clobber 
        args=[generic],
        dtype=(tl.uint64),
        is_pure=True,
        pack=1,
    )

@triton.jit
def cvta_shared_to_generic(shared):
    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cvta-shared-to-generic
    return tl.inline_asm_elementwise(
        "cvta.shared.u64 $0, $1;",
        "=l, l, ~{memory}",
        args=[shared],
        dtype=(tl.uint64),
        is_pure=True,
        pack=1,
    )

@triton.jit
def funnelshift_r(lo, hi, shift):
    # extracted from compiled PTX...
    return tl.inline_asm_elementwise(
        "shf.r.wrap.b32 %0, %1, %2, %3;",
        "=r, r, r, r",
        args=[lo, hi, shift],
        dtype=(tl.uint32),
        is_pure=True,
        pack=1,
    )

"""Primitives for 128bit Conversion as Triton doesn't support it (no tl.int128
Refer to nccl/src/device/op128.h

TODO Support different qualifiers for better performanes:
* [current] ld/st.volatile, will enforce access and bypass L1, used by default
* ld/st.relaxed.sys, for not important or protected
* ld/st.relaxed.gpu + .L1::no_allocate, L1 bypass saving + local access
* ld.acquire/st.release, used to enfore memory barrier with ordering
* (DeepEP) ld.global.nc.L1::no_allocate.L2::256B + st.global.L1::no_allocate"""

# Following are primitive to load/save data between REGISTER and GLOBAL memory
# NOTE we don't use shared memory, the same as NCCL when buffer is aligned 
# Read this first: https://medium.com/gpgpu/multi-gpu-programming-6768eeb42e2c
# First we need to clarify hardware steps inside ld and st between gpu pairs:
# GPU(0)-ld->GPU(1): SM(0)->NvLink->L2(1)->NvLink->L1(0)->SM(0)
# GPU(0)-st->GPU(1): SM(0)->L2(0)->NvLink->L2(1)->NvLink->SM(0)
#   missed GPU(0) load data: SM(0)->L1(0)->L2(0)->GMEM(0)->SM(0) 
# Compared with reading local GMEM, we can find L2 on both GPU is used while
# only local L1 cache is used. Two caches are of granularity 128Byte
# Also noted that most of our access is STREAMING, meaning data used only ONCE
# so adding sufficient cache bypass could be beneficial to performance

@triton.jit
def load128(addr):
    return tl.inline_asm_elementwise(
        "ld.volatile.global.v2.u64 {$0,$1}, [$2];",
        "=l,=l,l,~{memory}", 
        args=[addr],
        dtype=(tl.uint64, tl.uint64),
        is_pure=False, # prevent Triton optimization
        pack=1,
    )

@triton.jit
def store128(addr, v0, v1):
    tl.inline_asm_elementwise(
        """
        st.volatile.global.v2.u64 [$3], {$1,$2};
        mov.u64 $0, 0;
        """, # mov.u64 will be optimized when compiling into SASS
        "=l,l,l,l,~{memory}", 
        args=[v0, v1, addr],
        dtype=(tl.uint64),
        is_pure=False, # I think it's true but don't know
        pack=1,
    )

@triton.jit
def load64(addr):
    return tl.inline_asm_elementwise(
        "ld.volatile.global.u64 $0, [$1];",
        "=l,l,~{memory}", 
        args=[addr],
        dtype=(tl.uint64),
        is_pure=False, 
        pack=1,
    )

@triton.jit
def store64(addr, v):
    # asm volatile("barrier.sync.aligned %0, %1;" :: "r"(name), "r"(nThreads) : "memory");
    return tl.inline_asm_elementwise(
        """st.volatile.global.u64 [$2], $1;
        mov.u64 $0, 0;""",
        "=l,l,l,~{memory}",
        args=[v, addr],
        dtype=(tl.uint64),
        is_pure=False,
        pack=1,
    )


"""Reduce Primitives using of width u64/s64

For Communication we don't care about types so we formulates a byte-oriented
system for the need of protocol like LL128. But for necessary computation like
Sum in Collectives, we shall care about type, so we build following primitive 
whose input and output is of u64 and dtype supplied as argument.

TODO Support FP4 and FP8, Problem is that FP4/FP8 is Tensor Core ONLY, not even
     add for these. We have to do cvt to cast them to and from fp16x2"""

__DTYPE_MAP__: dict[tl.constexpr, str] = {
    tl.float32:  "f32",
    tl.float16:  "f16x2",
    tl.bfloat16: "bf16x2",
    tl.int32:    "s32",
    tl.int16:    "s16x2",
    tl.uint32:   "u32",
    tl.uint16:   "u16x2",
} # Map triton dtype to ptx dtype

__REDOP_TEMPLATE__ = """{
    .reg .v2 .b32 %a, %b;  
    mov.b64 {%a.x, %a.y}, $1;
    mov.b64 {%b.x, %b.y}, $2;
    redop.dtype %a.x, %a.x, %b.x;
    redop.dtype %a.y, %a.y, %b.y;
    mov.b64 $0, %a;
}""" # A template, use replace instead of format as `{}` is also ptx syntax :(

__FP8_REDOP_TEMPLATE__ = ""
__FP4_REDOP_TEMPLATE__ = ""

__SUM_FP16_TEMPLATE__ = """{
    .reg .v2 .b32 %a, %b;
    mov.b64 {%a.x, %a.y}, $1;
    mov.b64 {%b.x, %b.y}, $2;
    add.f16x2 %a.x, %a.x, %b.x;
    add.f16x2 %a.y, %a.y, %b.y;
    mov.b64 $0, %a;
}"""

@triton.jit
def SumFP16(a: tl.tensor, b: tl.tensor):
    return tl.inline_asm_elementwise(
        """{
    .reg .v2 .b32 %a, %b;
    mov.b64 {%a.x, %a.y}, $1;
    mov.b64 {%b.x, %b.y}, $2;
    add.f16x2 %a.x, %a.x, %b.x;
    add.f16x2 %a.y, %a.y, %b.y;
    mov.b64 $0, %a;
}""",
        constraints="=l, l, l", 
        args=(a, b),       # inputs 2 u64
        dtype=(tl.uint64), # return 1 u64
        is_pure=False, 
        pack=1)

@triton.jit
def _Sum(a: tl.tensor, b: tl.tensor, dtype: tl.constexpr):
    return tl.inline_asm_elementwise(
        __REDOP_TEMPLATE__.replace("dtype", __DTYPE_MAP__[dtype]).replace("redop", "add"),
        constraints="=l, l, l", 
        args=(a, b),       # inputs as u64
        dtype=(tl.uint64), # return as u64
        is_pure=True, 
        pack=1)

@triton.jit
def _Prod(a: tl.tensor, b: tl.tensor, dtype: tl.constexpr):
    return tl.inline_asm_elementwise(
        __REDOP_TEMPLATE__.replace("dtype", __DTYPE_MAP__[dtype]).replace("redop", "mul"),
        constraints="=l, l, l", 
        args=(a, b), 
        dtype=(tl.uint64),
        is_pure=True, 
        pack=1)

@triton.jit
def _Min(a: tl.tensor, b: tl.tensor, dtype: tl.constexpr):
    return tl.inline_asm_elementwise(
        __REDOP_TEMPLATE__.replace("dtype", __DTYPE_MAP__[dtype]).replace("redop", "min"),
        constraints="=l, l, l", 
        args=(a, b), 
        dtype=(tl.uint64),
        is_pure=True, 
        pack=1)

@triton.jit
def _Max(a: tl.tensor, b: tl.tensor, dtype: tl.constexpr):
    return tl.inline_asm_elementwise(
        __REDOP_TEMPLATE__.replace("dtype", __DTYPE_MAP__[dtype]).replace("redop", "max"),
        constraints="=l, l, l", 
        args=(a, b), 
        dtype=(tl.uint64),
        is_pure=True, 
        pack=1)

class RedOp(Enum): # Export as an enum
    Sum  = _Sum
    Prod = _Prod
    Min  = _Min
    Max  = _Max