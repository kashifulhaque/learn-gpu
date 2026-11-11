"""Lab 36: a symmetric-heap simulator, an all-gather from puts, and an overlap model.

Reference solution. Everything here runs on a CPU with plain torch.
"""

from __future__ import annotations

import torch

HEAP_BASE = 0x7F00_0000_0000  # virtual address of PE 0's heap
PE_STRIDE = 0x1_0000_0000  # virtual spacing between two PEs' heaps
ELEM = 4  # bytes per float32 element


class SignalNotReady(RuntimeError):
    """Raised when a wait can never finish: no put or signal is left in flight."""


class SymmetricHeap:
    """N simulated PEs, each with a float32 heap of the same size.

    A PE is a processing element, the NVSHMEM word for one rank. Offsets are in
    elements. The network is modeled by `pending`, a list of puts and signals
    that were issued and not yet delivered.
    """

    def __init__(self, n_pes: int, heap_elems: int, policy: str = "fifo") -> None:
        self.n_pes = n_pes
        self.heap_elems = heap_elems
        self.policy = policy  # "fifo", "reverse", or "signals_first"
        self.mem = torch.zeros(n_pes, heap_elems)
        self.top = 0
        self.pending: list[dict] = []
        self.epoch: dict[tuple[int, int], int] = {}
        self.seq = 0

    # Given: the network. You do not edit these.
    def _deliverable(self) -> list[dict]:
        out = []
        for op in self.pending:
            blocked = any(
                other["src"] == op["src"]
                and other["dst"] == op["dst"]
                and other["epoch"] < op["epoch"]
                for other in self.pending
            )
            if not blocked:
                out.append(op)
        return out

    def _deliver(self, op: dict) -> None:
        self.pending.remove(op)
        if op["kind"] == "data":
            values = op["values"]
            self.mem[op["dst"], op["offset"] : op["offset"] + values.numel()] = values
        else:
            self.mem[op["dst"], op["offset"]] = op["value"]

    def step(self) -> bool:
        """Deliver one in-flight operation, chosen by the policy. False if none."""
        ready = self._deliverable()
        if not ready:
            return False
        if self.policy == "reverse":
            op = max(ready, key=lambda o: o["seq"])
        elif self.policy == "signals_first":
            op = min(ready, key=lambda o: (o["kind"] != "signal", o["seq"]))
        else:
            op = min(ready, key=lambda o: o["seq"])
        self._deliver(op)
        return True

    # Yours.
    def address(self, pe: int, offset: int) -> int:
        return HEAP_BASE + pe * PE_STRIDE + offset * ELEM

    def translate(self, addr: int, from_pe: int, to_pe: int) -> int:
        local = addr - self.address(from_pe, 0)
        if not 0 <= local < self.heap_elems * ELEM:
            raise ValueError(f"address {addr:#x} is not in PE {from_pe}'s heap")
        return self.address(to_pe, 0) + local

    def malloc(self, n_elems: int) -> int:
        if self.top + n_elems > self.heap_elems:
            raise MemoryError(f"symmetric heap full: {self.top} + {n_elems} > {self.heap_elems}")
        offset = self.top
        self.top += n_elems
        return offset

    def _issue(self, kind: str, src: int, dst: int, offset: int, **payload) -> None:
        self.seq += 1
        self.pending.append(
            {
                "kind": kind,
                "src": src,
                "dst": dst,
                "offset": offset,
                "epoch": self.epoch.get((src, dst), 0),
                "seq": self.seq,
                **payload,
            }
        )

    def put(self, src: int, dst: int, offset: int, values: torch.Tensor) -> None:
        self._issue("data", src, dst, offset, values=values.detach().clone().reshape(-1))

    def signal(self, src: int, dst: int, offset: int, value: float) -> None:
        self._issue("signal", src, dst, offset, value=float(value))

    def fence(self, src: int) -> None:
        for dst in range(self.n_pes):
            self.epoch[(src, dst)] = self.epoch.get((src, dst), 0) + 1

    def quiet(self, src: int) -> None:
        mine = sorted((o for o in self.pending if o["src"] == src), key=lambda o: (o["epoch"], o["seq"]))
        for op in mine:
            self._deliver(op)

    def barrier_all(self) -> None:
        for src in range(self.n_pes):
            self.quiet(src)

    def get(self, caller: int, src: int, offset: int, n_elems: int) -> torch.Tensor:
        del caller
        return self.mem[src, offset : offset + n_elems].clone()

    def wait_until(self, pe: int, offset: int, value: float) -> None:
        while self.mem[pe, offset].item() != value:
            if not self.step():
                raise SignalNotReady(
                    f"PE {pe} waits for {value} at offset {offset}, which is {self.mem[pe, offset].item()}, "
                    "and nothing is in flight"
                )


def all_gather_with_puts(chunks: list[torch.Tensor], policy: str = "fifo") -> list[torch.Tensor]:
    """Every PE ends with the concatenation of all chunks, gathered by puts and signals."""
    n = len(chunks)
    c = chunks[0].numel()
    heap = SymmetricHeap(n, n * c + n, policy)
    buf = heap.malloc(n * c)
    sig = heap.malloc(n)
    for pe in range(n):
        heap.mem[pe, buf + pe * c : buf + (pe + 1) * c] = chunks[pe].reshape(-1)
    for src in range(n):
        for dst in range(n):
            if dst == src:
                continue
            heap.put(src, dst, buf + src * c, chunks[src])
            heap.fence(src)
            heap.signal(src, dst, sig + src, 1.0)
    outs = []
    for pe in range(n):
        for src in range(n):
            if src != pe:
                heap.wait_until(pe, sig + src, 1.0)
        outs.append(heap.get(pe, pe, buf, n * c))
    return outs


def tile_order(pe: int, n_pes: int) -> list[int]:
    """The order PE `pe` computes row blocks: its own chunk, then the next PEs' in ring order."""
    return [(pe + k) % n_pes for k in range(n_pes)]


def ag_gemm(
    shards: list[torch.Tensor], weights: list[torch.Tensor], policy: str = "fifo"
) -> tuple[list[torch.Tensor], list[tuple]]:
    """All-gather the row shards of A, and on each PE compute full-A @ weights[pe] tile by tile.

    Returns the per-PE outputs and an event log of ("wait", pe, chunk) and
    ("compute", pe, chunk) tuples, in the order they happened.
    """
    n = len(shards)
    rows, k = shards[0].shape
    heap = SymmetricHeap(n, n * rows * k + n, policy)
    buf = heap.malloc(n * rows * k)
    sig = heap.malloc(n)
    for pe in range(n):
        heap.mem[pe, buf + pe * rows * k : buf + (pe + 1) * rows * k] = shards[pe].reshape(-1)
    for src in range(n):
        for dst in range(n):
            if dst != src:
                heap.put(src, dst, buf + src * rows * k, shards[src])
                heap.fence(src)
                heap.signal(src, dst, sig + src, 1.0)
    log: list[tuple] = []
    outs = []
    for pe in range(n):
        out = torch.zeros(n * rows, weights[pe].shape[1])
        for chunk in tile_order(pe, n):
            if chunk != pe:
                heap.wait_until(pe, sig + chunk, 1.0)
                log.append(("wait", pe, chunk))
            tile = heap.get(pe, pe, buf + chunk * rows * k, rows * k).reshape(rows, k)
            out[chunk * rows : (chunk + 1) * rows] = tile @ weights[pe]
            log.append(("compute", pe, chunk))
        outs.append(out)
    return outs, log


def overlap_model(
    n_bytes: float,
    flops: float,
    p: int,
    link_gbs: float,
    peak_tflops: float,
    alpha_s: float,
    launch_s: float,
    signal_s: float,
    sm_steal: float,
    chunks: int,
) -> dict:
    """Time of serial, overlapped (separate collectives), and fused execution, in seconds."""
    bandwidth = link_gbs * 1e9
    comm_total = (p - 1) * alpha_s + (p - 1) / p * n_bytes / bandwidth
    comp_total = flops / (peak_tflops * 1e12)
    serial = comm_total + comp_total

    def pipeline(comm_chunk: float, comp_chunk: float) -> float:
        return comm_chunk + (chunks - 1) * max(comm_chunk, comp_chunk) + comp_chunk

    wire = (p - 1) / p * (n_bytes / chunks) / bandwidth
    comp_chunk = comp_total / chunks * (1 + sm_steal)
    overlapped = pipeline((p - 1) * alpha_s + wire, comp_chunk + launch_s)
    fused = pipeline(signal_s + wire, comp_chunk)
    times = {"serial": serial, "overlapped": overlapped, "fused": fused}
    winner = min(times, key=lambda name: (times[name], ("serial", "overlapped", "fused").index(name)))
    return {
        "serial_s": serial,
        "overlapped_s": overlapped,
        "fused_s": fused,
        "winner": winner,
    }
