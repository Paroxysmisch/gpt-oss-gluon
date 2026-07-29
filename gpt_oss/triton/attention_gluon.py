import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    tma, mbarrier, fence_async_shared,
    warpgroup_mma_init, warpgroup_mma, warpgroup_mma_wait,
)


def _pick_n(block_n):
    """Pick the largest n <= 256 that evenly divides block_n for WGMMA instr_shape."""
    n = 256
    while n > block_n or block_n % n != 0:
        n -= 8
    return n


@gluon.jit
def _attn_fwd_kernel(
    Q_desc, K_desc, V_desc, Sinks, sm_scale, M, Out_desc, Start_q,
    Z, H, N_Q_CTX, N_KV_CTX,
    HEAD_DIM: gl.constexpr, BANDWIDTH: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    N_QK: gl.constexpr, N_PV: gl.constexpr,
    Q_SMEM_LAYOUT: gl.constexpr, KV_SMEM_LAYOUT: gl.constexpr,
    P_SMEM_LAYOUT: gl.constexpr, num_warps: gl.constexpr,
):
    """
    Gluon attention-forward kernel using Hopper WGMMA + TMA.

    Translates the reference Triton kernel's online-softmax attention into
    explicit warpgroup MMA calls.  Each CTA processes one (BLOCK_M, HEAD_DIM)
    tile of Q and iterates over K/V tiles with causal masking and optional
    sliding-window (bandwidth) support.

    Key design decisions (correctness-first, single pipeline stage):
    - Q is loaded once via TMA into shared memory (NVMMASharedLayout).
    - K, V tiles are loaded via TMA in the inner loop.
    - Q@Kᵀ uses WGMMA with the transposed K-smem view.
    - Online softmax (m_i, l_i, alpha rescaling) is done on the WGMMA
      accumulator using NVMMADistributedLayout and SliceLayout for reductions.
    - The softmax result p is stored to fp16 shared memory, then consumed
      by a second WGMMA for P@V.
    - Each P@V matmul uses a fresh warpgroup_mma_init accumulator; the
      result is added to the running accumulator (which stays a plain tensor
      and thus supports elementwise scaling by alpha).
    - Attention sinks are handled identically to the reference.
    - Log-sum-exp M is stored per-row using a BlockedLayout.
    """
    dtype: gl.constexpr = Q_desc.dtype
    start_q = gl.load(Start_q).to(gl.int32)
    start_m = gl.program_id(0)
    off_hz = gl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    sink = gl.load(Sinks + off_h).to(gl.float32)

    m: gl.constexpr = 16
    k_instr: gl.constexpr = 256 // dtype.primitive_bitwidth
    warps_per_cta: gl.constexpr = [num_warps, 1]

    mma_qk_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=warps_per_cta, instr_shape=[m, N_QK, k_instr],
    )
    mma_pv_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=warps_per_cta, instr_shape=[m, N_PV, k_instr],
    )

    slice_m_qk: gl.constexpr = gl.SliceLayout(dim=1, parent=mma_qk_layout)
    slice_n_qk: gl.constexpr = gl.SliceLayout(dim=0, parent=mma_qk_layout)
    slice_m_pv: gl.constexpr = gl.SliceLayout(dim=1, parent=mma_pv_layout)

    q_smem = gl.allocate_shared_memory(dtype, [BLOCK_M, HEAD_DIM], Q_SMEM_LAYOUT)
    # Double-buffer K and V for pipelining
    k_smem = gl.allocate_shared_memory(dtype, [BLOCK_N, HEAD_DIM], KV_SMEM_LAYOUT)
    v_smem = gl.allocate_shared_memory(dtype, [2, BLOCK_N, HEAD_DIM], KV_SMEM_LAYOUT)
    p_smem = gl.allocate_shared_memory(dtype, [BLOCK_M, BLOCK_N], P_SMEM_LAYOUT)

    # Separate mbarriers for K and V loads
    bar_k = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    bar_v = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar_k, count=1)
    mbarrier.init(bar_v, count=1)
    phase_k = 0
    phase_v = 0

    q_row = off_hz * N_Q_CTX + start_m * BLOCK_M
    mbarrier.expect(bar_k, Q_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(Q_desc, [q_row, 0], bar_k, q_smem)
    mbarrier.wait(bar_k, phase=phase_k)
    phase_k ^= 1

    m_i = gl.full([BLOCK_M], sink, gl.float32, slice_m_pv)
    l_i = gl.zeros([BLOCK_M], gl.float32, slice_m_pv)
    acc = gl.zeros((BLOCK_M, HEAD_DIM), dtype=gl.float32, layout=mma_pv_layout)

    qk_scale = sm_scale

    if BANDWIDTH:
        lo = gl.maximum(start_q, start_q + start_m * BLOCK_M - BANDWIDTH)
    else:
        lo = start_q
    hi = start_q + (start_m + 1) * BLOCK_M

    # Prologue: issue K[0] and V[0] loads (don't wait yet)
    kv_row_0 = off_hz * N_KV_CTX + lo
    mbarrier.expect(bar_k, K_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(K_desc, [kv_row_0, 0], bar_k, k_smem)
    mbarrier.expect(bar_v, V_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(V_desc, [kv_row_0, 0], bar_v, v_smem.index(0))

    cur = 0
    start_n = lo
    while start_n < hi:
        next_buf = cur ^ 1

        # Wait for K (completes load from previous iteration / prologue)
        mbarrier.wait(bar_k, phase=phase_k)
        phase_k ^= 1

        k_trans = k_smem.permute((1, 0))
        qk = warpgroup_mma_init(gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_qk_layout))
        qk = warpgroup_mma(q_smem, k_trans, qk, is_async=True)
        qk = warpgroup_mma_wait(num_outstanding=0, deps=(qk,))

        # Issue K[next] load (if not last tile) — k_smem is now free, overlaps with softmax+P@V
        next_start_n = start_n + BLOCK_N
        if next_start_n < hi:
            mbarrier.expect(bar_k, K_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(K_desc, [off_hz * N_KV_CTX + next_start_n, 0], bar_k, k_smem)

        offs_m = gl.arange(0, BLOCK_M, layout=slice_m_qk)
        offs_n = gl.arange(0, BLOCK_N, layout=slice_n_qk)
        mask = (start_n + offs_n[None, :]) > (start_q + start_m * BLOCK_M + offs_m[:, None])
        if BANDWIDTH:
            too_old = (start_n + offs_n[None, :]) < (start_q + start_m * BLOCK_M + offs_m[:, None] - BANDWIDTH + 1)
            mask = mask | too_old

        qk = qk * qk_scale + gl.where(mask, -1.0e6, 0.0)

        m_ij_qk = gl.max(qk, 1)
        m_ij = gl.convert_layout(m_ij_qk, slice_m_pv)
        alpha = gl.exp(m_i - m_ij)
        qk = qk - gl.convert_layout(m_ij, slice_m_qk)[:, None]
        p = gl.exp(qk)
        l_ij_qk = gl.sum(p, 1)
        l_ij = gl.convert_layout(l_ij_qk, slice_m_pv)

        acc = acc * alpha[:, None]

        p_smem.store(p.to(dtype))
        fence_async_shared()

        # Wait for V[cur]
        mbarrier.wait(bar_v, phase=phase_v)
        phase_v ^= 1

        # Issue V[next] load (if not last tile) — overlaps with P@V on V[cur]
        if next_start_n < hi:
            mbarrier.expect(bar_v, V_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(V_desc, [off_hz * N_KV_CTX + next_start_n, 0], bar_v, v_smem.index(next_buf))

        v_cur = v_smem.index(cur)
        new_acc = warpgroup_mma_init(gl.zeros((BLOCK_M, HEAD_DIM), dtype=gl.float32, layout=mma_pv_layout))
        new_acc = warpgroup_mma(p_smem, v_cur, new_acc, is_async=True)
        new_acc = warpgroup_mma_wait(num_outstanding=0, deps=(new_acc,))
        acc = acc + new_acc

        l_i = l_i * alpha + l_ij
        m_i = m_ij

        cur = next_buf
        start_n = next_start_n

    mbarrier.invalidate(bar_k)
    mbarrier.invalidate(bar_v)

    sink = gl.exp(sink - m_i)
    z = l_i + sink
    acc = acc / z[:, None]

    m_i = m_i + gl.log(l_i)
    m_store_layout: gl.constexpr = gl.BlockedLayout([1], [32], [num_warps], [0])
    offs_m_store = gl.arange(0, BLOCK_M, layout=m_store_layout)
    m_ptrs = M + off_hz * N_Q_CTX + offs_m_store
    gl.store(m_ptrs, gl.convert_layout(m_i, m_store_layout))

    # Reuse q_smem for output staging instead of allocating a separate out_smem buffer.
    # q_smem is no longer needed after the inner loop, and both q_smem (Q_SMEM_LAYOUT)
    # and out_smem would use identical NVMMASharedLayout for [BLOCK_M, HEAD_DIM].
    q_smem.store(acc.to(Out_desc.dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(Out_desc, [q_row, 0], q_smem)
    tma.store_wait(pendings=0)


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sinks, sm_scale, bandwidth, start_q):
        assert len(start_q) == 1
        bs, n_ctx, n_kv_heads, repeat_kv, HEAD_DIM_Q = q.shape
        bs, n_kv_ctx, n_kv_heads, HEAD_DIM_K = k.shape
        bs, n_kv_ctx, n_kv_heads, HEAD_DIM_V = v.shape
        n_heads = n_kv_heads * repeat_kv
        q = q.view(bs, n_ctx, n_heads, HEAD_DIM_Q)
        k = k.view(bs, n_kv_ctx, n_kv_heads, HEAD_DIM_K)
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        q = q.transpose(1, 2).contiguous()
        k = k.repeat_interleave(repeat_kv, dim=2).transpose(1, 2).contiguous()
        v = v.repeat_interleave(repeat_kv, dim=2).transpose(1, 2).contiguous()

        BLOCK_M = 64
        BLOCK_N = 64
        m_pad_size = BLOCK_M - n_ctx % BLOCK_M if n_ctx % BLOCK_M != 0 else 0
        q = torch.nn.functional.pad(q, (0, 0, 0, m_pad_size))
        n_pad_size = BLOCK_N - n_kv_ctx % BLOCK_N if n_kv_ctx % BLOCK_N != 0 else 0
        k = torch.nn.functional.pad(k, (0, 0, 0, n_pad_size))
        v = torch.nn.functional.pad(v, (0, 0, 0, n_pad_size))

        dtype_gl = gl.float16

        q_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, HEAD_DIM_K], dtype_gl)
        k_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_N, HEAD_DIM_K], dtype_gl)
        v_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_N, HEAD_DIM_K], dtype_gl)
        o_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, HEAD_DIM_K], dtype_gl)
        p_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], dtype_gl)

        y_dim = q.shape[0] * q.shape[1] * q.shape[2]

        q_desc = TensorDescriptor(q, shape=[y_dim, HEAD_DIM_K],
                                  strides=[HEAD_DIM_K, 1],
                                  block_shape=[BLOCK_M, HEAD_DIM_K],
                                  layout=q_layout)
        k_desc = TensorDescriptor(k, shape=[y_dim, HEAD_DIM_K],
                                  strides=[HEAD_DIM_K, 1],
                                  block_shape=[BLOCK_N, HEAD_DIM_K],
                                  layout=k_layout)
        v_desc = TensorDescriptor(v, shape=[y_dim, HEAD_DIM_K],
                                  strides=[HEAD_DIM_K, 1],
                                  block_shape=[BLOCK_N, HEAD_DIM_K],
                                  layout=v_layout)

        o = torch.empty_like(q)
        M = torch.empty((bs, n_heads, n_ctx + m_pad_size), device=q.device, dtype=torch.float32)

        o_desc = TensorDescriptor(o, shape=[y_dim, HEAD_DIM_K],
                                  strides=[HEAD_DIM_K, 1],
                                  block_shape=[BLOCK_M, HEAD_DIM_K],
                                  layout=o_layout)

        n_qk = _pick_n(BLOCK_N)
        n_pv = _pick_n(HEAD_DIM_K)

        num_pid_m = triton.cdiv(n_ctx + m_pad_size, BLOCK_M)
        num_pid_n = bs * n_heads
        grid = (num_pid_m, num_pid_n)

        _attn_fwd_kernel[grid](
            q_desc, k_desc, v_desc, sinks, sm_scale, M, o_desc, start_q,
            q.shape[0], q.shape[1], n_ctx + m_pad_size, n_kv_ctx,
            HEAD_DIM=HEAD_DIM_K, BANDWIDTH=bandwidth or 0,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, N_QK=n_qk, N_PV=n_pv,
            Q_SMEM_LAYOUT=q_layout, KV_SMEM_LAYOUT=k_layout,
            P_SMEM_LAYOUT=p_layout, num_warps=4,
        )

        ctx.save_for_backward(q, k, v, sinks, o, M, start_q)
        ctx.sm_scale = sm_scale
        ctx.bandwidth = bandwidth

        o = o[:, :, :n_ctx, :].transpose(1, 2).contiguous()
        o = o.view(bs, n_ctx, n_heads * HEAD_DIM_V)
        return o


attention = _attention.apply
