import triton
import triton.language as tl
import torch
import torch_npu
import triton.testing
import time
import pandas as pd
from typing import Dict, List, Callable

DEV = "npu"
activation = "leaky_relu_custom"

AUTOTUNE_CONFIGS_FULL = [
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}),  
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64}), 
    triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128}),
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64}),
    triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128}), 
    triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64}),
]
AUTOTUNE_CONFIGS_BASE = [
    triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64}),
]

MATRIX_SHAPES = [
    (512, 512, 512),       # Small Square
    (2048, 2048, 2048),    # Medium Square
    (4096, 4096, 4096),    # Large Square
    (2048, 1024, 4096),    # Rectangular (K-dominant)
    (4096, 4096, 1024),    # Rectangular (M, N dominant)
]
NUM_TEST_RUNS = 100
NUM_WARMUP_RUNS = 20

@triton.jit
def leaky_relu_custom(x):
    return tl.where(x >= 0, x, 0.01 * x) + 1.0

@triton.jit
def relu_custom(x):
    pos = tl.maximum(x, 0.0)
    neg = tl.minimum(x, 0.0)
    return pos + 0.01 * neg + 1.0

def torch_matmul(a, b, activation=""):
    c = torch.matmul(a, b)
    if activation == "leaky_relu_custom":
        c = torch.where(c >= 0, c, 0.01 * c) + 1.0
    return c

@triton.jit
def _matmat_core_logic(
    a_ptr, b_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    K_SPLITS: tl.constexpr
):
    """Calculates the dot product for one block and returns the accumulator and PIDs.
    Supports 2D program_id mapping (m, n) and optional K_SPLITS which splits the K-loop
    across `K_SPLITS` program instances along axis=2 when grid is 3D.
    """
    # program_id mapping: axis 0 -> m, axis 1 -> n, optional axis 2 -> k split
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs_base = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs_base = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    msk_m = offs_am < M
    msk_n = offs_bn < N
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)
    num_k_iters = tl.cdiv(K, BLOCK_SIZE_K)
    # Determine K-loop stepping: if K_SPLITS > 1, distribute k-iterations across axis=2
    pid_k = tl.program_id(2)
    k_start = pid_k
    k_step = K_SPLITS
    S_UNROLL: tl.constexpr = 8
    # iterate k indices assigned to this program, strided by k_step
    step = k_step * S_UNROLL
    for k_idx in range(k_start, num_k_iters, step):
        # inner unroll over S_UNROLL iterations (if available)
        for s in range(S_UNROLL):
            k = k_idx + s * k_step
            if k < num_k_iters:
                a_ptrs = a_ptrs_base + k * BLOCK_SIZE_K * stride_ak
                b_ptrs = b_ptrs_base + k * BLOCK_SIZE_K * stride_bk

                rem_k = K - k * BLOCK_SIZE_K
                mask_a_k = offs_k[None, :] < rem_k
                mask_b_k = offs_k[:, None] < rem_k

                a = tl.load(a_ptrs, mask=msk_m[:, None] & mask_a_k, other=0.0)
                b = tl.load(b_ptrs, mask=msk_n[None, :] & mask_b_k, other=0.0)
                accumulator = tl.dot(a, b, accumulator)
    return accumulator, pid_m, pid_n, msk_m, msk_n

@triton.jit
def _matmat_single_writeback(
    c_ptr, accumulator, pid_m, pid_n, M, N, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    accumulator=relu_custom(accumulator)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

@triton.jit
def _matmat_parallel_writeback(
    c_ptr, accumulator, pid_m, pid_n, M, N, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    """Writes back C with NPU vector core parallelism (tl.parallel)."""
    SUB_BLK_M: tl.constexpr = BLOCK_SIZE_M // 4
    accumulator=relu_custom(accumulator)
    for s in tl.parallel(0, 4, bind_sub_block=True):
        vec_sub_blk = tl.extract_slice(
            accumulator, (s * SUB_BLK_M, 0), (SUB_BLK_M, BLOCK_SIZE_N), (1, 1)
        )
        offs_cm = pid_m * BLOCK_SIZE_M + s * SUB_BLK_M + tl.arange(0, SUB_BLK_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, vec_sub_blk, mask=c_mask)

def create_kernel(name: str, autotune_configs: List[triton.Config], parallel: bool, k_splits: int = 1) -> Callable:
    """Dynamically creates a JIT kernel with specified autotune configs and parallelism."""
    @triton.autotune(
        configs=autotune_configs,
        key=["M", "N", "K"],
    )
    @triton.jit
    def kernel( 
        a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_ck, stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, ACTIVATION: tl.constexpr,
        PARALLEL: tl.constexpr, K_SPLITS: tl.constexpr
    ):
        accumulator, pid_m, pid_n, msk_m, msk_n = _matmat_core_logic(
            a_ptr, b_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, K_SPLITS
        )

        # If K_SPLITS > 1 we are doing K-axis parallelism; each pid_k writes its own slab
        if K_SPLITS > 1:
            pid_k = tl.program_id(2)
            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            # offset into 3D buffer: pid_k * stride_ck + usual m/n strides
            c_ptrs = c_ptr + pid_k * stride_ck + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            tl.store(c_ptrs, accumulator, mask=c_mask)
            return

        if PARALLEL:
            _matmat_parallel_writeback(
                c_ptr, accumulator, pid_m, pid_n, M, N, stride_cm, stride_cn,
                BLOCK_SIZE_M, BLOCK_SIZE_N
            )
        else:
            _matmat_single_writeback(
                c_ptr, accumulator, pid_m, pid_n, M, N, stride_cm, stride_cn,
                BLOCK_SIZE_M, BLOCK_SIZE_N
            )
    
    kernel.parallel_flag = parallel
    kernel.k_splits = k_splits

    return kernel


# K-axis parallel variant (splits K-loop across axis=2), uses atomic_add into float32 C buffer
KERNEL_V4_KPAR = create_kernel("V4_Opt_Full_KPAR", AUTOTUNE_CONFIGS_FULL, parallel=True, k_splits=2)

def matmul_wrapper(a, b, kernel_func):
    """Generic Python wrapper to launch a Matmul Triton kernel."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    
    M, K = a.shape 
    K, N = b.shape
    # If kernel requests K-axis parallelism, allocate float32 accumulation buffer for atomic adds
    k_splits = getattr(kernel_func, 'k_splits', 1)
    if k_splits > 1:
        # allocate per-split accumulation buffer (k_splits, M, N) in float32
        c = torch.zeros((k_splits, M, N), device=a.device, dtype=torch.float16)
        stride_ck, stride_cm, stride_cn = c.stride(0), c.stride(1), c.stride(2)
    else:
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)
        stride_ck = 0
        stride_cm, stride_cn = c.stride(0), c.stride(1)

    # Build grid: use 2D grid (num_pid_m, num_pid_n) and optionally a 3rd axis for K splits
    def grid(META):
        num_m = triton.cdiv(M, META["BLOCK_SIZE_M"])
        num_n = triton.cdiv(N, META["BLOCK_SIZE_N"])
        if k_splits > 1:
            return (num_m, num_n, k_splits)
        return (num_m, num_n)
    
    is_parallel = getattr(kernel_func, 'parallel_flag', False)
    act = getattr(kernel_func, 'activation', activation)
    kernel_func[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        stride_ck, stride_cm, stride_cn,
        ACTIVATION=act,
        PARALLEL=is_parallel,
        K_SPLITS=k_splits
    )
    # If we used a per-split accumulation buffer, reduce along k_splits and cast
    if k_splits > 1:
        c_final = c.sum(dim=0)
        return c_final.to(torch.float16)
    return c
