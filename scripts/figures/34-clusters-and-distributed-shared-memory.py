#!/usr/bin/env python3
"""Numbers for chapter 34's cluster-fit and multicast-traffic figures.

Cluster fit. The H100 SXM5 has 8 GPCs and 132 SMs (NVIDIA Hopper in-depth
blog). NVIDIA does not publish the SM count of each GPC. This script assumes
six GPCs with 16 SMs and two with 18, which sums to 132. All blocks of a
cluster sit in one GPC, so a GPC with S SMs holds S * b // c clusters of size
c when each SM holds b blocks. The real count comes from
cudaOccupancyMaxActiveClusters.

Multicast traffic. A 4096 by 4096 by 4096 bf16 GEMM with 128 by 256 output
tiles has 32 by 16 = 512 blocks. Each block loads a 128 by 4096 slice of A
and a 4096 by 256 slice of B from L2. A cluster of cm by cn blocks shares
one A slice per cluster row and one B slice per cluster column. With TMA
multicast, each distinct slice is loaded once per cluster.

Run:  python3 scripts/figures/34-clusters-and-distributed-shared-memory.py
Paste the printed arrays into frontend/src/components/viz/34/.
"""

from __future__ import annotations

GPC_SMS = (16, 16, 16, 16, 16, 16, 18, 18)
MIB = 1024 * 1024


def clusters_per_gpc(cluster: int, per_sm: int) -> list[int]:
    return [sms * per_sm // cluster for sms in GPC_SMS]


def multicast(m, n, k, tile_m, tile_n, elem, cm, cn):
    tiles_m, tiles_n = -(-m // tile_m), -(-n // tile_n)
    a_slice, b_slice = tile_m * k * elem, tile_n * k * elem
    a_base, b_base = tiles_m * tiles_n * a_slice, tiles_m * tiles_n * b_slice
    a_mc = b_mc = 0
    for im in range(-(-tiles_m // cm)):
        dm = min(cm, tiles_m - im * cm)
        for jn in range(-(-tiles_n // cn)):
            dn = min(cn, tiles_n - jn * cn)
            a_mc += dm * a_slice
            b_mc += dn * b_slice
    return a_base, b_base, a_mc, b_mc


def main() -> None:
    print("// cluster fit: [cluster size, blocks per SM, clusters per GPC..., active clusters, idle slots]")
    for per_sm in (1, 2):
        for c in range(1, 17):
            per_gpc = clusters_per_gpc(c, per_sm)
            total = sum(per_gpc)
            idle = sum(GPC_SMS) * per_sm - total * c
            print(f"[{c}, {per_sm}, {per_gpc}, {total}, {idle}],")
    print()
    print("// multicast: [cm, cn, A baseline MiB, B baseline MiB, A multicast MiB, B multicast MiB]")
    for cm, cn in ((1, 1), (2, 1), (1, 2), (2, 2), (4, 1), (4, 2), (2, 4), (4, 4)):
        a0, b0, a1, b1 = multicast(4096, 4096, 4096, 128, 256, 2, cm, cn)
        print(f"[{cm}, {cn}, {a0 / MIB:g}, {b0 / MIB:g}, {a1 / MIB:g}, {b1 / MIB:g}],")


if __name__ == "__main__":
    main()
