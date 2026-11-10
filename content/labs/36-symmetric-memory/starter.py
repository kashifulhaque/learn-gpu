"""Lab 36: a symmetric-heap simulator, an all-gather from puts, and an overlap model.

You write the pieces of NVSHMEM-style one-sided communication on a CPU, then
use them. Everything here runs on plain torch with no GPU.

A *PE* is a processing element, the NVSHMEM word for one rank. Every PE owns a
heap of the same size, and an object allocated at offset `o` on one PE sits at
offset `o` on every PE. The network between PEs is simulated: a `put` or a
`signal` goes into `pending` and arrives later, in an order the policy picks.
The policy `"signals_first"` is adversarial: it delivers a signal before the
data it was meant to announce, whenever the rules let it. You write the
operations that make that safe.

The checks run in the order of the TODOs. The first needs only `address`.
A separate two-GPU check runs only when the machine has two GPUs.
"""

from __future__ import annotations

import torch

HEAP_BASE = 0x7F00_0000_0000  # virtual address of PE 0's heap
PE_STRIDE = 0x1_0000_0000  # virtual spacing between two PEs' heaps
ELEM = 4  # bytes per float32 element


class SignalNotReady(RuntimeError):
    """Raised when a wait can never finish: no put or signal is left in flight."""


class SymmetricHeap:
    """N simulated PEs, each with a float32 heap of the same size. Offsets are in elements."""

    def __init__(self, n_pes: int, heap_elems: int, policy: str = "fifo") -> None:
        self.n_pes = n_pes
        self.heap_elems = heap_elems
        self.policy = policy  # "fifo", "reverse", or "signals_first"
        self.mem = torch.zeros(n_pes, heap_elems)  # mem[pe] is PE pe's heap
        self.top = 0  # the bump allocator's next free offset
        self.pending: list[dict] = []  # puts and signals in flight
        self.epoch: dict[tuple[int, int], int] = {}  # (src, dst) -> fence count
        self.seq = 0  # issue order of operations

    # Given: the network. You do not edit these three methods.
    def _deliverable(self) -> list[dict]:
        """In-flight operations the network may deliver now.

        An operation is held back while an operation with a smaller `epoch`
        from the same source to the same destination is still in flight. That
        is the whole meaning of a fence.
        """
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
        """Deliver one in-flight operation, chosen by the policy. False if none is in flight."""
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
        """The virtual address of element `offset` in PE `pe`'s heap.

        PE p's heap starts at HEAP_BASE + p * PE_STRIDE. An element is ELEM bytes.
        Offset 3 on PE 1 is HEAP_BASE + PE_STRIDE + 12.
        """
        # TODO
        raise NotImplementedError

    def translate(self, addr: int, from_pe: int, to_pe: int) -> int:
        """Turn an address in `from_pe`'s heap into the same object's address in `to_pe`'s heap.

        Keep the byte offset into the heap, and swap the base. Raise ValueError
        when `addr` is not inside `from_pe`'s heap.
        """
        # TODO
        raise NotImplementedError

    def malloc(self, n_elems: int) -> int:
        """Allocate `n_elems` on every PE at once and return the shared offset.

        A bump allocator: the first call returns 0, the next returns the
        previous size, and so on. Raise MemoryError when the heap would overflow.
        """
        # TODO
        raise NotImplementedError

    def put(self, src: int, dst: int, offset: int, values: torch.Tensor) -> None:
        """Issue a put from PE `src` into PE `dst`'s heap at `offset`. Do not deliver it.

        Append {"kind": "data", "src", "dst", "offset", "epoch", "seq", "values"}
        to `pending`. `epoch` is `self.epoch.get((src, dst), 0)`. Increase
        `self.seq` first and store it as `seq`. Copy `values`: the caller may
        reuse its buffer once a put returns.
        """
        # TODO
        raise NotImplementedError

    def signal(self, src: int, dst: int, offset: int, value: float) -> None:
        """Issue a signal write of `value` to `offset` on PE `dst`. Do not deliver it.

        Same as `put`, with {"kind": "signal", ..., "value": value}.
        """
        # TODO
        raise NotImplementedError

    def fence(self, src: int) -> None:
        """Order PE `src`'s earlier operations before its later ones, per destination.

        Do not deliver anything. Raise the epoch of every (src, dst) pair by one.
        """
        # TODO
        raise NotImplementedError

    def quiet(self, src: int) -> None:
        """Complete every operation PE `src` issued. Leave other PEs' operations in flight.

        Deliver them in (epoch, seq) order with `self._deliver`.
        """
        # TODO
        raise NotImplementedError

    def barrier_all(self) -> None:
        """Every PE quiets. Given the shape of `quiet`, this is one loop."""
        for src in range(self.n_pes):
            self.quiet(src)

    def get(self, caller: int, src: int, offset: int, n_elems: int) -> torch.Tensor:
        """Read `n_elems` from PE `src`'s heap as it is now. Return a copy."""
        del caller
        return self.mem[src, offset : offset + n_elems].clone()

    def wait_until(self, pe: int, offset: int, value: float) -> None:
        """Spin until PE `pe`'s word at `offset` equals `value`.

        One spin is one `self.step()`. When `step` returns False and the word
        still differs, nothing can ever change it: raise SignalNotReady.
        """
        # TODO
        raise NotImplementedError


def all_gather_with_puts(chunks: list[torch.Tensor], policy: str = "fifo") -> list[torch.Tensor]:
    """Every PE ends with `torch.cat(chunks)`, gathered with puts and signals.

    Make a SymmetricHeap with room for the gathered buffer (n * c elements) and
    n signal words. Each PE writes its own chunk into its heap at slot
    `pe * c`. Each PE then puts its chunk into every other PE's slot, calls
    `fence`, and signals that PE's word `sig + src`. Each PE waits for the n - 1
    signals it expects and then reads its whole buffer. Return one tensor per PE.

    The policy "signals_first" delivers a signal before its data unless you fence.
    """
    # TODO
    raise NotImplementedError


def tile_order(pe: int, n_pes: int) -> list[int]:
    """The order PE `pe` computes row blocks: its own chunk first, then the next PEs' in ring order.

    For PE 1 of 4: [1, 2, 3, 0].
    """
    # TODO
    raise NotImplementedError


def ag_gemm(
    shards: list[torch.Tensor], weights: list[torch.Tensor], policy: str = "fifo"
) -> tuple[list[torch.Tensor], list[tuple]]:
    """All-gather the row shards of A, and on each PE compute full-A @ weights[pe] tile by tile.

    PE `pe` computes row block `chunk` of its output as `A_chunk @ weights[pe]`
    in `tile_order(pe, n)`. Before it reads a remote chunk it must `wait_until`
    that chunk's signal is set. Append ("wait", pe, chunk) after each wait and
    ("compute", pe, chunk) after each tile. A PE's own chunk needs no wait.

    Return the per-PE outputs (each `n * rows` by `weights[pe].shape[1]`) and the log.
    """
    # TODO
    raise NotImplementedError


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
    """Time, in seconds, of serial, overlapped, and fused execution of one all-gather plus one GEMM.

    comm_total = (p - 1) * alpha_s + (p - 1) / p * n_bytes / (link_gbs * 1e9)
    comp_total = flops / (peak_tflops * 1e12)
    serial     = comm_total + comp_total

    A pipeline of `chunks` chunks, with one communication stage then one compute stage:
    T = comm_chunk + (chunks - 1) * max(comm_chunk, comp_chunk) + comp_chunk

    wire       = (p - 1) / p * (n_bytes / chunks) / (link_gbs * 1e9)
    comp_chunk = comp_total / chunks * (1 + sm_steal)
    overlapped: comm_chunk = (p - 1) * alpha_s + wire,  compute stage = comp_chunk + launch_s
    fused:      comm_chunk = signal_s + wire,           compute stage = comp_chunk

    Return {"serial_s", "overlapped_s", "fused_s", "winner"}. `winner` is the
    smallest of the three; a tie goes to "serial", then "overlapped", then "fused".
    """
    # TODO
    raise NotImplementedError
