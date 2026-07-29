"""Prefill attention test bench: standard Triton kernel vs. optimized Gluon kernel.

Compares the two forward attention implementations in this package on the
*prefill* path (the only path where these kernels run — decode uses the reference
path for both):

  - standard: ``gpt_oss.triton.attention.attention``      (reference FlashAttention, bf16)
  - gluon:    ``gpt_oss.triton.attention_gluon.attention`` (Hopper WGMMA + TMA, fp16 math)

Both expose an identical API and shape contract, so this script feeds each the same
synthetic inputs shaped like the real model's attention (head_dim=64, 64 query
heads, 8 KV heads => repeat_kv=8), sweeps sequence length and sliding-window mode,
verifies correctness against ``attention_ref``, then times each with CUDA events.

Reported per (backend, seqlen, window): wall-clock latency (mean/median ms),
**tokens/sec** (= batch * seqlen / mean_latency), achieved attention TFLOP/s, and
a gluon-vs-standard speedup ratio.

No model checkpoint is required. Run on a Hopper GPU (e.g. GH200):

    python -m gpt_oss.triton.bench_attention --seqlens 1024,2048,4096 --iters 50

The Gluon backend needs ``triton.experimental.gluon`` (recent Triton, possibly
built from source). If it is unavailable the bench still runs the standard kernel
and reports gluon as skipped.
"""

import argparse
import math
import statistics
import sys

import torch

from gpt_oss.torch.model import ModelConfig
from gpt_oss.triton.attention import attention as attention_standard, attention_ref

try:
    from gpt_oss.triton.attention_gluon import attention as attention_gluon
    _GLUON_IMPORT_ERROR = None
except Exception as e:  # older/non-gluon Triton, or import-time failure
    attention_gluon = None
    _GLUON_IMPORT_ERROR = e

# Correctness thresholds on max abs error of the bf16 output vs. the fp32 reference.
# Attention outputs are ~unit magnitude; bf16/fp16 accumulation lands well under
# these bounds when correct.
_PASS_TOL = 0.05
_WARN_TOL = 0.25

_BACKENDS = {
    "standard": attention_standard,
    "gluon": attention_gluon,
}


def _make_inputs(batch_size, seqlen, cfg, dtype, device, seed=0):
    """Build synthetic prefill inputs matching the model's attention shapes."""
    g = torch.Generator(device=device).manual_seed(seed)
    n_kv_heads = cfg.num_key_value_heads
    repeat_kv = cfg.num_attention_heads // cfg.num_key_value_heads
    head_dim = cfg.head_dim
    n_heads = cfg.num_attention_heads

    q = torch.randn(batch_size, seqlen, n_kv_heads, repeat_kv, head_dim,
                    generator=g, device=device, dtype=torch.float32).to(dtype)
    k = torch.randn(batch_size, seqlen, n_kv_heads, head_dim,
                    generator=g, device=device, dtype=torch.float32).to(dtype)
    v = torch.randn(batch_size, seqlen, n_kv_heads, head_dim,
                    generator=g, device=device, dtype=torch.float32).to(dtype)
    sinks = torch.randn(n_heads, generator=g, device=device, dtype=torch.float32).to(dtype)
    start_q = torch.zeros(1, dtype=torch.int32, device=device)
    sm_scale = 1.0 / math.sqrt(head_dim)
    return q, k, v, sinks, sm_scale, start_q


def _attended_pairs(seqlen, window):
    """Number of unmasked (query, key) pairs under causal (+ optional sliding window)."""
    S = seqlen
    if window is None:
        return S * (S + 1) // 2
    W = window
    if S <= W:
        return S * (S + 1) // 2
    # queries [0, W): attend q+1 keys; queries [W, S): attend W keys
    return W * (W + 1) // 2 + (S - W) * W


def _flops(batch_size, seqlen, cfg, window):
    """Attention FLOPs: QK^T + P@V = 4 * heads * head_dim * (attended pairs)."""
    pairs = _attended_pairs(seqlen, window)
    return 4.0 * batch_size * cfg.num_attention_heads * cfg.head_dim * pairs


def _verify(fn, q, k, v, sinks, sm_scale, window, start_q):
    out = fn(q, k, v, sinks, sm_scale, window, start_q)
    ref = attention_ref(q, k, v, sinks, sm_scale, window, start_q)
    err = (out.float() - ref.float()).abs()
    max_err = err.max().item()
    mean_err = err.mean().item()
    status = "PASS" if max_err < _PASS_TOL else ("WARN" if max_err < _WARN_TOL else "FAIL")
    return max_err, mean_err, status


def _time(fn, q, k, v, sinks, sm_scale, window, start_q, warmup, iters):
    for _ in range(warmup):
        fn(q, k, v, sinks, sm_scale, window, start_q)
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(iters):
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        fn(q, k, v, sinks, sm_scale, window, start_q)
        end_evt.record()
        torch.cuda.synchronize()
        times_ms.append(start_evt.elapsed_time(end_evt))
    return times_ms


def _parse_windows(spec):
    out = []
    for tok in spec.split(","):
        tok = tok.strip().lower()
        if tok in ("none", "full", "0", ""):
            out.append(None)
        else:
            out.append(int(tok))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seqlens", default="512,1024,2048,4096,8192",
                        help="Comma-separated prefill sequence lengths.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"],
                        help="Input dtype (the model runs bf16).")
    parser.add_argument("--windows", default="none,128",
                        help="Comma-separated sliding windows; 'none' = full causal. "
                             "Model uses full causal and 128 on alternating layers.")
    parser.add_argument("--backends", default="standard,gluon",
                        help="Comma-separated subset of {standard,gluon}.")
    parser.add_argument("--csv", default=None, help="Optional path to write results as CSV.")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Run this bench on a Hopper GPU (e.g. GH200).",
              file=sys.stderr)
        return 1

    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    cfg = ModelConfig()
    seqlens = [int(s) for s in args.seqlens.split(",") if s.strip()]
    windows = _parse_windows(args.windows)
    requested = [b.strip() for b in args.backends.split(",") if b.strip()]

    backends = []
    for name in requested:
        if name not in _BACKENDS:
            print(f"WARNING: unknown backend {name!r}, skipping.", file=sys.stderr)
            continue
        if _BACKENDS[name] is None:
            print(f"NOTE: backend {name!r} unavailable ({_GLUON_IMPORT_ERROR}); skipping.",
                  file=sys.stderr)
            continue
        backends.append(name)

    if not backends:
        print("ERROR: no runnable backends.", file=sys.stderr)
        return 1

    print(f"device={torch.cuda.get_device_name(0)}  dtype={args.dtype}  "
          f"batch={args.batch_size}  warmup={args.warmup}  iters={args.iters}")
    print(f"model shapes: heads={cfg.num_attention_heads} kv_heads={cfg.num_key_value_heads} "
          f"head_dim={cfg.head_dim}")
    print(f"backends: {', '.join(backends)}\n")

    header = (f"{'window':>7} {'seqlen':>7} {'backend':>9} {'verify':>6} "
              f"{'max_err':>9} {'lat_ms':>9} {'median_ms':>10} {'tok/s':>12} {'TFLOP/s':>9}")
    print(header)
    print("-" * len(header))

    rows = []  # for CSV and speedup computation
    for window in windows:
        for S in seqlens:
            q, k, v, sinks, sm_scale, start_q = _make_inputs(
                args.batch_size, S, cfg, dtype, device)
            flops = _flops(args.batch_size, S, cfg, window)
            per_window_mean = {}
            for name in backends:
                fn = _BACKENDS[name]
                try:
                    max_err, mean_err, status = _verify(
                        fn, q, k, v, sinks, sm_scale, window, start_q)
                    times = _time(fn, q, k, v, sinks, sm_scale, window, start_q,
                                  args.warmup, args.iters)
                    mean_ms = statistics.mean(times)
                    median_ms = statistics.median(times)
                    tok_s = args.batch_size * S / (mean_ms / 1e3)
                    tflops = flops / (mean_ms / 1e3) / 1e12
                    per_window_mean[name] = mean_ms
                    win_str = "full" if window is None else str(window)
                    print(f"{win_str:>7} {S:>7} {name:>9} {status:>6} "
                          f"{max_err:>9.4f} {mean_ms:>9.3f} {median_ms:>10.3f} "
                          f"{tok_s:>12,.0f} {tflops:>9.1f}")
                    rows.append({
                        "window": "full" if window is None else window,
                        "seqlen": S, "backend": name, "verify": status,
                        "max_err": max_err, "mean_err": mean_err,
                        "lat_ms": mean_ms, "median_ms": median_ms,
                        "tokens_per_s": tok_s, "tflops": tflops,
                    })
                except Exception as e:
                    win_str = "full" if window is None else str(window)
                    print(f"{win_str:>7} {S:>7} {name:>9} {'ERROR':>6}  {type(e).__name__}: {e}")
                    rows.append({
                        "window": "full" if window is None else window,
                        "seqlen": S, "backend": name, "verify": "ERROR",
                        "max_err": float("nan"), "mean_err": float("nan"),
                        "lat_ms": float("nan"), "median_ms": float("nan"),
                        "tokens_per_s": float("nan"), "tflops": float("nan"),
                    })

            # Speedup line when both backends produced a timing for this config.
            if "standard" in per_window_mean and "gluon" in per_window_mean:
                speedup = per_window_mean["standard"] / per_window_mean["gluon"]
                win_str = "full" if window is None else str(window)
                print(f"{win_str:>7} {S:>7} {'speedup':>9}  gluon is {speedup:.2f}x "
                      f"{'faster' if speedup >= 1 else 'slower'} than standard")
            print()

    if args.csv:
        import csv
        fields = ["window", "seqlen", "backend", "verify", "max_err", "mean_err",
                  "lat_ms", "median_ms", "tokens_per_s", "tflops"]
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {args.csv}")

    any_fail = any(r["verify"] in ("FAIL", "ERROR") for r in rows)
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
