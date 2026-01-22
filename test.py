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
triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}),
triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}),
triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}),
triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}),
triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}),
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
    (4096,8192,4096),
    (2048,8192,2048),
 (4096,32,4096),
    (2048,32,2048),
    (4096,32,2048)
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

    #num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    #num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs_base = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs_base = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    msk_m = offs_am < M
    msk_n = offs_bn < N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

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
    #accumulator = tl.where(accumulator >= 0, accumulator, 0.01*accumulator) + 1.0
    return accumulator, pid_m, pid_n, msk_m, msk_n

@triton.jit
def matmul_splitk_stage1(
    a_ptr, b_ptr, c_partial_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_pk, stride_pm, stride_pn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_SPLITS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_tiles = tl.cdiv(K, BLOCK_K)

    # split-K 核心：stride = K_SPLITS
    for k in range(pid_k, num_k_tiles, K_SPLITS):
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k * BLOCK_K + offs_k)[None, :] * stride_ak
        b_ptrs = b_ptr + (k * BLOCK_K + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (k * BLOCK_K + offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_n[None, :] < N) & (k * BLOCK_K + offs_k[:, None] < K),
            other=0.0,
        )

        acc = tl.dot(a, b, acc)

    # 写 partial
    # compute base offset (in elements) for this (pid_k, pid_m, pid_n)
    base_off = pid_k * stride_pk + pid_m * stride_pm + pid_n * stride_pn
    c_ptrs = c_partial_ptr + base_off + offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn

    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )

@triton.jit
def matmul_splitk_stage2(
    c_partial_ptr, c_ptr,
    M, N,
    stride_pk, stride_pm, stride_pn,
    stride_cm, stride_cn,
    K_SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(K_SPLITS):
        # compute base offset for this k,pid_m,pid_n
        base_off = k * stride_pk + pid_m * stride_pm + pid_n * stride_pn
        ptrs = c_partial_ptr + base_off + offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn
        acc += tl.load(ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )



@triton.jit
def _matmat_single_writeback(
    c_ptr, accumulator, pid_m, pid_n, M, N, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, ACTIVATION: tl.constexpr
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
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, ACTIVATION: tl.constexpr
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
            # accumulator already float32
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
                BLOCK_SIZE_M, BLOCK_SIZE_N, ACTIVATION
            )
        else:
            _matmat_single_writeback(
                c_ptr, accumulator, pid_m, pid_n, M, N, stride_cm, stride_cn,
                BLOCK_SIZE_M, BLOCK_SIZE_N, ACTIVATION
            )

    kernel.parallel_flag = parallel
    kernel.k_splits = k_splits

    return kernel

def create_kernel_SplitK(
    name: str,
    autotune_configs: List[triton.Config],
    parallel: bool,
    k_splits: int = 1,
) -> Callable:
    """
    Returns a callable that performs matmul.
    If k_splits > 1, uses true two-stage split-K.
    """

    # =========================
    # Stage 1: split-K compute
    # =========================
    @triton.autotune(
        configs=autotune_configs,
        key=["M", "N", "K"],
    )
    @triton.jit
    def _stage1_kernel(
        a_ptr, b_ptr, c_partial_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_pk, stride_pm, stride_pn,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        K_SPLITS: tl.constexpr,
    ):
        matmul_splitk_stage1(
            a_ptr, b_ptr,
            c_partial_ptr,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_pk, stride_pm, stride_pn,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
            K_SPLITS,
        )

    # =========================
    # Stage 2: reduce kernel
    # =========================
    @triton.jit
    def _stage2_kernel(
        c_partial_ptr, c_ptr,
        M, N,
        stride_pk, stride_pm, stride_pn,
        stride_cm, stride_cn,
        K_SPLITS: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        matmul_splitk_stage2(
            c_partial_ptr,
            c_ptr,
            M, N,
            stride_pk, stride_pm, stride_pn,
            stride_cm, stride_cn,
            K_SPLITS,
            BLOCK_SIZE_M, BLOCK_SIZE_N,
        )

    # =========================
    # Python wrapper (关键)
    # =========================
    def kernel(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
        #assert a.is_cuda and b.is_cuda and c.is_cuda
        assert a.dtype == b.dtype

        M, K = a.shape
        K2, N = b.shape
        assert K == K2

        # strides (in elements)
        stride_am, stride_ak = a.stride()
        stride_bk, stride_bn = b.stride()
        stride_cm, stride_cn = c.stride()

        BLOCK_M = autotune_configs[0].kwargs["BLOCK_SIZE_M"]
        BLOCK_N = autotune_configs[0].kwargs["BLOCK_SIZE_N"]
        BLOCK_K = autotune_configs[0].kwargs["BLOCK_SIZE_K"]

        grid_m = triton.cdiv(M, BLOCK_M)
        grid_n = triton.cdiv(N, BLOCK_N)

        # ==============
        # No split-K
        # ==============
        if k_splits == 1:
            grid = (grid_m, grid_n)
            _stage1_kernel[grid](
                a, b, c,
                M=M, N=N, K=K,
                stride_am=stride_am, stride_ak=stride_ak,
                stride_bk=stride_bk, stride_bn=stride_bn,
                stride_pk=0, stride_pm=stride_cm, stride_pn=stride_cn,
                K_SPLITS=1,
            )
            return

        # ===================
        # True split-K path
        # ===================
        # allocate partial buffer: [K_SPLITS, M, N] in blocks
        c_partial = torch.empty(
            (k_splits, M, N),
            device=c.device,
            dtype=torch.float32,
        )

        stride_pk, stride_pm, stride_pn = c_partial.stride()

        # ---- Stage 1 ----
        grid = (grid_m, grid_n, k_splits)
        _stage1_kernel[grid](
            a, b, c_partial,
            M=M, N=N, K=K,
            stride_am=stride_am, stride_ak=stride_ak,
            stride_bk=stride_bk, stride_bn=stride_bn,
            stride_pk=stride_pk, stride_pm=stride_pm, stride_pn=stride_pn,
            K_SPLITS=k_splits,
        )

        # ---- Stage 2 ----
        grid = (grid_m, grid_n)
        _stage2_kernel[grid](
            c_partial, c,
            M, N,
            stride_pk, stride_pm, stride_pn,
            stride_cm, stride_cn,
            k_splits,
            BLOCK_M, BLOCK_N,
        )

    kernel.parallel_flag = parallel
    kernel.k_splits = k_splits
    kernel.__name__ = name

    return kernel

KERNEL_V1 = create_kernel("V1_Baseline", AUTOTUNE_CONFIGS_BASE, parallel=False)
KERNEL_V2 = create_kernel("V2_Opt_Autotune", AUTOTUNE_CONFIGS_FULL, parallel=False)
KERNEL_V3 = create_kernel("V3_Opt_NPU_Parallel", AUTOTUNE_CONFIGS_BASE, parallel=True)
KERNEL_V4 = create_kernel("V4_Opt_Full", AUTOTUNE_CONFIGS_FULL, parallel=True)
KERNEL_V4_KPAR = create_kernel("V4_Opt_Full_KPAR", AUTOTUNE_CONFIGS_FULL, parallel=True, k_splits=2)
KERNEL_V5_SplitK = create_kernel_SplitK("V5_Opt_Full_SplitK", AUTOTUNE_CONFIGS_FULL, parallel=True, k_splits=4)
# use relu_custom for this KPAR variant
#KERNEL_V4_KPAR.activation = "relu_custom"

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

def matmul_wrapper_splitK(a, b, kernel_func):
    """
    kernel_func is a Python callable returned by create_kernel_SplitK
    """
    assert a.shape[1] == b.shape[0]

    M, K = a.shape
    _, N = b.shape

    # 输出张量
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # 直接调用 Python wrapper
    kernel_func(a, b, c)

    return c


EXPERIMENT_VERSIONS = {
    "V1_Baseline (Base Config | No Parallel)": KERNEL_V1,
    "V4_Opt_Full_KPAR (Full Config | Parallel | K-axis Parallelism)": KERNEL_V4_KPAR,
    "V5_Opt_Full_SplitK (Full Config | Parallel | Split-K)": KERNEL_V5_SplitK,
}

def run_performance_test():
    torch.npu.set_device(0)
    torch.manual_seed(0)

    all_results = []

    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)

    print("--- Triton 算子优化多版本性能对比实验开始 ---")

    for M, K, N in MATRIX_SHAPES:
        print(f"\n--- 测试形状: M={M}, K={K}, N={N} ---")

        a = torch.randn((M, K), device=DEV, dtype=torch.float16)
        b = torch.randn((K, N), device=DEV, dtype=torch.float16)

        print(f"  > 预热 ({NUM_WARMUP_RUNS}次)...")
        for _ in range(NUM_WARMUP_RUNS):
            torch_matmul(a, b, activation)
            for kernel_func in EXPERIMENT_VERSIONS.values():
                if kernel_func == KERNEL_V4_KPAR:
                    a=a.contiguous()
                    b=b.contiguous()
                    matmul_wrapper(a, b, kernel_func)
                if kernel_func == KERNEL_V5_SplitK:
                    a=a.contiguous()
                    b=b.contiguous()
                    matmul_wrapper_splitK(a, b, kernel_func)
                else:
                    matmul_wrapper(a, b, kernel_func)
        torch.npu.synchronize()
        print("  > 预热完成。")

        shape_results = {"Shape": f"{M}x{K}x{N}"}
        reference_output = None

        for name, kernel_func in EXPERIMENT_VERSIONS.items():
            times_ms = []

            for _ in range(NUM_TEST_RUNS):
                start_event.record()
                if kernel_func == KERNEL_V5_SplitK:
                    output = matmul_wrapper_splitK(a, b, kernel_func)
                else:
                    output = matmul_wrapper(a, b, kernel_func)
                end_event.record()
                end_event.synchronize()
                times_ms.append(start_event.elapsed_time(end_event))

            avg_time_ms = sum(times_ms) / NUM_TEST_RUNS
            shape_results[name] = f"{avg_time_ms:.3f} ms"

            if "V1_Baseline" in name:
                reference_output = output

            print(f"  - {name}: {avg_time_ms:.3f} ms")


        torch_output_ref = torch_matmul(a, b, activation)

        TOLERANCE_ATOL = 1e-2
        TOLERANCE_RTOL = 1e-2
        is_close = torch.allclose(reference_output, torch_output_ref, atol=TOLERANCE_ATOL, rtol=TOLERANCE_RTOL, equal_nan=False)

        if not is_close:
            print(f"  警告: Triton 基线结果与 PyTorch 参照在容忍度 ({TOLERANCE_ATOL}) 外不一致。")
        else:
            print(f"  校验成功: Triton 基线结果与 PyTorch 参照一致。")

        all_results.append(shape_results)

    print("\n" + "=" * 100)
    print("Triton 算子优化性能量化结果")
    print("=" * 100)
    print("下表展示了四种优化策略在不同矩阵形状下的平均运行时间(ms)")

    df = pd.DataFrame(all_results)

    v1_times = df["V1_Baseline (Base Config | No Parallel)"].str.replace(' ms', '').astype(float)
    v4_times = df["V4_Opt_Full_KPAR (Full Config | Parallel | K-axis Parallelism)"].str.replace(' ms', '').astype(float)
    df['V4/V1 提速比'] = (v1_times / v4_times).apply(lambda x: f"{x:.2f}x")

    df = df[['Shape',
             'V1_Baseline (Base Config | No Parallel)',
             #'V2_Opt_Autotune (Full Config | No Parallel)',
             #'V3_Opt_NPU_Parallel (Base Config | Parallel)',
             #'V4_Opt_Full (Full Config | Parallel)',
             'V4_Opt_Full_KPAR (Full Config | Parallel | K-axis Parallelism)',
             'V5_Opt_Full_SplitK (Full Config | Parallel | Split-K)',
             #'V4/V1 提速比'
             ]]

    print(df.to_string(index=False))


if __name__ == '__main__':
    print("通过四个版本对比 Autotune 和 NPU 向量核并行优化的效果")
    print("V1: 基线 | V2: Autotune | V3: NPU Parallel | V4: Autotune + NPU Parallel| K-axis Parallelism")
    run_performance_test()
