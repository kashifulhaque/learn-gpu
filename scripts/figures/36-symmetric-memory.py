#!/usr/bin/env python3
"""Numbers for chapter 36's overlap model, chunk sweep, and tile-ready order.

The model is the alpha-beta cost of one all-gather plus one GEMM, per GPU:

  comm_total = (p - 1) * alpha + (p - 1) / p * n / B
  comp_total = flops / peak
  serial     = comm_total + comp_total

With c chunks, a two-stage pipeline of communication then compute is

  T = comm_chunk + (c - 1) * max(comm_chunk, comp_chunk) + comp_chunk

Separate collectives pay (p - 1) * alpha per chunk and one kernel launch per
chunk. A fused kernel pays one signal latency per chunk and no launch. Both
lose sm_steal of the compute rate while communication runs on the SMs.

Hardware: NVLink 3 on A100, 300 GB/s per direction (docs/FACTS.md section
7.1) and 312 TFLOP/s dense bf16. The latencies alpha, launch, and signal and
the SM share are assumed values, not measurements. Run this file to print the
table that frontend/src/components/viz/36/overlap-sweep.tsx embeds.
"""

from __future__ import annotations

LINK_GBS = 300.0
PEAK_TFLOPS = 312.0
ALPHA_S = 5e-6
LAUNCH_S = 5e-6
SIGNAL_S = 2e-6
SM_STEAL = 0.10


def overlap_model(n_bytes, flops, p, link_gbs, peak_tflops, alpha_s, launch_s, signal_s, sm_steal, chunks):
    bandwidth = link_gbs * 1e9
    comm_total = (p - 1) * alpha_s + (p - 1) / p * n_bytes / bandwidth
    comp_total = flops / (peak_tflops * 1e12)
    serial = comm_total + comp_total

    def pipeline(comm_chunk, comp_chunk):
        return comm_chunk + (chunks - 1) * max(comm_chunk, comp_chunk) + comp_chunk

    wire = (p - 1) / p * (n_bytes / chunks) / bandwidth
    comp_chunk = comp_total / chunks * (1 + sm_steal)
    overlapped = pipeline((p - 1) * alpha_s + wire, comp_chunk + launch_s)
    fused = pipeline(signal_s + wire, comp_chunk)
    times = {"serial": serial, "overlapped": overlapped, "fused": fused}
    winner = min(times, key=lambda k: (times[k], ("serial", "overlapped", "fused").index(k)))
    return {"serial_s": serial, "overlapped_s": overlapped, "fused_s": fused, "winner": winner,
            "comm_total_s": comm_total, "comp_total_s": comp_total}


def tile_order(pe, n_pes):
    return [(pe + k) % n_pes for k in range(n_pes)]


# Scenario A: tensor-parallel layer, 8 GPUs. A is 8192 x 8192 bf16 (128 MiB
# gathered), B is 8192 x 1024 per GPU. Scenario B: a small decode-sized layer.
SCENARIOS = {
    "A": dict(n_bytes=8192 * 8192 * 2, flops=2 * 8192 * 8192 * 1024, p=8),
    "B": dict(n_bytes=256 * 8192 * 2, flops=2 * 256 * 8192 * 1024, p=8),
}
CHUNKS = [1, 2, 4, 8, 16, 32, 64]


def run(name, chunks):
    s = SCENARIOS[name]
    return overlap_model(s["n_bytes"], s["flops"], s["p"], LINK_GBS, PEAK_TFLOPS,
                         ALPHA_S, LAUNCH_S, SIGNAL_S, SM_STEAL, chunks)


if __name__ == "__main__":
    for name in SCENARIOS:
        base = run(name, 1)
        print(f"scenario {name}: n={SCENARIOS[name]['n_bytes']} flops={SCENARIOS[name]['flops']}")
        print(f"  comm_total {base['comm_total_s']*1e6:.3f} us, comp_total {base['comp_total_s']*1e6:.3f} us, "
              f"serial {base['serial_s']*1e6:.3f} us")
        for c in CHUNKS:
            r = run(name, c)
            print(f"  c={c:>2}: overlapped {r['overlapped_s']*1e6:9.3f} us  fused {r['fused_s']*1e6:9.3f} us  winner {r['winner']}")
    print("tile order, 4 PEs:", {pe: tile_order(pe, 4) for pe in range(4)})
    # JSON for the component
    import json
    out = {n: {"serial": round(run(n, 1)["serial_s"] * 1e6, 3),
               "overlapped": [round(run(n, c)["overlapped_s"] * 1e6, 3) for c in CHUNKS],
               "fused": [round(run(n, c)["fused_s"] * 1e6, 3) for c in CHUNKS]} for n in SCENARIOS}
    print(json.dumps(out))
