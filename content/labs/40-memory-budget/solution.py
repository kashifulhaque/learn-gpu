"""Lab 40: a training-memory budget, a caching-allocator simulator, a snapshot
parser, and a checkpoint chooser. Worked solution."""

from __future__ import annotations

import itertools  # noqa: F401  (kept for readers who brute-force)

MiB = 1 << 20

# PyTorch's native caching allocator, c10/cuda/CUDACachingAllocator.cpp.
MIN_BLOCK = 512  # every size rounds up to a multiple of this
SMALL_SIZE = 1 * MiB  # a request up to this size uses the small pool
SMALL_BUFFER = 2 * MiB  # the segment for a small request
LARGE_BUFFER = 20 * MiB  # the segment for a request of 1 MiB to 10 MiB
MIN_LARGE_ALLOC = 10 * MiB  # from here up, the segment is the request rounded up
ROUND_LARGE = 2 * MiB  # ... to a multiple of this


# ---------------------------------------------------------------------------
# Part 1: the byte budget
# ---------------------------------------------------------------------------

_BYTES = {
    # mode: (weights, grads, master copy, Adam m, Adam v), bytes per parameter
    "fp32": (4, 4, 0, 4, 4),
    "mixed": (2, 2, 4, 4, 4),
}


def state_bytes(n_params: int, mode: str) -> dict:
    w, g, master, m, v = _BYTES[mode]
    out = {
        "weights": w * n_params,
        "grads": g * n_params,
        "master": master * n_params,
        "adam_m": m * n_params,
        "adam_v": v * n_params,
    }
    out["total"] = sum(out.values())
    return out


def activation_layer_bytes(s: int, b: int, h: int, a: int, flash: bool = False) -> int:
    linear = 34 * s * b * h
    scores = 0 if flash else 5 * a * s * s * b
    return linear + scores


def peak_activation_bytes(full, boundary, ckpt) -> int:
    ckpt = set(ckpt)
    stored = [boundary[i] if i in ckpt else full[i] for i in range(len(full))]
    peak = sum(stored)
    prefix = 0
    for i in range(len(full)):
        prefix += stored[i]
        if i in ckpt:
            peak = max(peak, prefix + full[i])
    return peak


def mlp_stack_saved_bytes(layers: int, tokens: int, hidden: int, elem_bytes: int) -> int:
    # Each Linear(h, h) saves its input and each GELU saves its input.
    return layers * 2 * tokens * hidden * elem_bytes


# ---------------------------------------------------------------------------
# Part 2: the caching allocator
# ---------------------------------------------------------------------------


class OutOfMemory(Exception):
    pass


def round_size(nbytes: int) -> int:
    if nbytes < MIN_BLOCK:
        return MIN_BLOCK
    return MIN_BLOCK * ((nbytes + MIN_BLOCK - 1) // MIN_BLOCK)


def segment_size(rounded: int) -> int:
    if rounded <= SMALL_SIZE:
        return SMALL_BUFFER
    if rounded < MIN_LARGE_ALLOC:
        return LARGE_BUFFER
    return ROUND_LARGE * ((rounded + ROUND_LARGE - 1) // ROUND_LARGE)


class CachingAllocator:
    def __init__(self, capacity=None):
        self.capacity = capacity
        self.segments = []  # each: {"address", "size", "small", "blocks": [block, ...]}
        self.handles = {}  # handle -> block
        self.next_handle = 1
        self.next_address = 0
        self.allocated = 0
        self.reserved = 0
        self.peak_allocated = 0
        self.peak_reserved = 0

    # A block is {"offset", "size", "free", "requested", "handle"}.

    def _best_fit(self, small, rounded):
        best = None
        for seg in self.segments:
            if seg["small"] != small:
                continue
            for blk in seg["blocks"]:
                if blk["free"] and blk["size"] >= rounded:
                    key = (blk["size"], seg["address"] + blk["offset"])
                    if best is None or key < best[0]:
                        best = (key, seg, blk)
        return None if best is None else (best[1], best[2])

    def _release_free_segments(self):
        keep = []
        for seg in self.segments:
            if len(seg["blocks"]) == 1 and seg["blocks"][0]["free"]:
                self.reserved -= seg["size"]
            else:
                keep.append(seg)
        self.segments = keep

    def _new_segment(self, small, rounded):
        size = segment_size(rounded)
        if self.capacity is not None and self.reserved + size > self.capacity:
            self._release_free_segments()
            if self.reserved + size > self.capacity:
                raise OutOfMemory(f"cannot reserve {size} bytes: reserved {self.reserved} of {self.capacity}")
        seg = {
            "address": self.next_address,
            "size": size,
            "small": small,
            "blocks": [{"offset": 0, "size": size, "free": True, "requested": 0, "handle": None}],
        }
        self.next_address += size
        self.reserved += size
        self.segments.append(seg)
        return seg, seg["blocks"][0]

    def malloc(self, nbytes: int) -> int:
        rounded = round_size(nbytes)
        small = rounded <= SMALL_SIZE
        found = self._best_fit(small, rounded)
        if found is None:
            found = self._new_segment(small, rounded)
        seg, blk = found
        remaining = blk["size"] - rounded
        if remaining >= MIN_BLOCK if small else remaining > SMALL_SIZE:
            rest = {"offset": blk["offset"] + rounded, "size": remaining, "free": True, "requested": 0, "handle": None}
            blk["size"] = rounded
            seg["blocks"].insert(seg["blocks"].index(blk) + 1, rest)
        handle = self.next_handle
        self.next_handle += 1
        blk["free"] = False
        blk["requested"] = nbytes
        blk["handle"] = handle
        self.handles[handle] = (seg, blk)
        self.allocated += blk["size"]
        self.peak_allocated = max(self.peak_allocated, self.allocated)
        self.peak_reserved = max(self.peak_reserved, self.reserved)
        return handle

    def free(self, handle: int) -> None:
        seg, blk = self.handles.pop(handle)
        self.allocated -= blk["size"]
        blk["free"] = True
        blk["requested"] = 0
        blk["handle"] = None
        blocks = seg["blocks"]
        i = blocks.index(blk)
        if i + 1 < len(blocks) and blocks[i + 1]["free"]:
            blk["size"] += blocks[i + 1]["size"]
            del blocks[i + 1]
        if i > 0 and blocks[i - 1]["free"]:
            blocks[i - 1]["size"] += blk["size"]
            del blocks[i]

    def snapshot(self) -> dict:
        segs = []
        for seg in self.segments:
            blocks = []
            for blk in seg["blocks"]:
                blocks.append({
                    "address": seg["address"] + blk["offset"],
                    "size": blk["size"],
                    "requested_size": blk["requested"],
                    "state": "inactive" if blk["free"] else "active_allocated",
                    "frames": [],
                })
            active = sum(b["size"] for b in blocks if b["state"] != "inactive")
            segs.append({
                "address": seg["address"],
                "total_size": seg["size"],
                "segment_type": "small" if seg["small"] else "large",
                "stream": 0,
                "allocated_size": active,
                "active_size": active,
                "blocks": blocks,
            })
        return {"segments": segs, "device_traces": [[]]}


def run_trace(events, capacity=None) -> dict:
    alloc = CachingAllocator(capacity)
    handles = {}
    allocated, reserved = [], []
    oom_at = None
    for i, ev in enumerate(events):
        try:
            if ev[0] == "alloc":
                handles[ev[1]] = alloc.malloc(ev[2])
            else:
                alloc.free(handles.pop(ev[1]))
        except OutOfMemory:
            oom_at = i
            break
        allocated.append(alloc.allocated)
        reserved.append(alloc.reserved)
    return {
        "allocated": allocated,
        "reserved": reserved,
        "peak_allocated": alloc.peak_allocated,
        "peak_reserved": alloc.peak_reserved,
        "oom_at": oom_at,
        "allocator": alloc,
    }


# ---------------------------------------------------------------------------
# Part 3: read a snapshot
# ---------------------------------------------------------------------------


def _blocks(snapshot):
    for seg in snapshot["segments"]:
        for blk in seg["blocks"]:
            yield blk


def reserved_and_allocated(snapshot) -> tuple:
    reserved = sum(seg["total_size"] for seg in snapshot["segments"])
    active = sum(b["size"] for b in _blocks(snapshot) if b["state"] != "inactive")
    return reserved, active


def largest_free_block(snapshot) -> int:
    sizes = [b["size"] for b in _blocks(snapshot) if b["state"] == "inactive"]
    return max(sizes, default=0)


def fragmentation_ratio(snapshot) -> float:
    free = sum(b["size"] for b in _blocks(snapshot) if b["state"] == "inactive")
    if free == 0:
        return 0.0
    return 1.0 - largest_free_block(snapshot) / free


def peak_allocation(snapshot) -> dict:
    live = 0
    best = {"peak_bytes": 0, "event": None, "addr": None, "size": 0, "frame": None}
    for i, e in enumerate(snapshot["device_traces"][0]):
        if e["action"] == "alloc":
            live += e["size"]
            if live > best["peak_bytes"]:
                f = e["frames"][0] if e["frames"] else None
                best = {
                    "peak_bytes": live,
                    "event": i,
                    "addr": e["addr"],
                    "size": e["size"],
                    "frame": None if f is None else f"{f['filename']}:{f['line']} {f['name']}",
                }
        elif e["action"] == "free_requested":
            live -= e["size"]
    return best


def classify_oom(snapshot):
    ooms = [e for e in snapshot["device_traces"][0] if e["action"] == "oom"]
    if not ooms:
        return None
    want = ooms[-1]["size"]
    free = sum(b["size"] for b in _blocks(snapshot) if b["state"] == "inactive")
    if free < want:
        return "capacity"
    if largest_free_block(snapshot) < want:
        return "fragmentation"
    return "other"


# ---------------------------------------------------------------------------
# Part 4: choose which layers to checkpoint
# ---------------------------------------------------------------------------


def choose_checkpoints(full, boundary, cost, budget):
    # State: the bytes stored by the layers so far. Value: the cheapest set.
    states = {0: (0, ())}
    for i in range(len(full)):
        nxt = {}
        for stored, (spent, chosen) in states.items():
            # Keep the layer whole: its saved tensors stay.
            total = stored + full[i]
            if total <= budget and (total not in nxt or spent < nxt[total][0]):
                nxt[total] = (spent, chosen)
            # Checkpoint it: keep the boundary, and rebuild the layer later.
            total = stored + boundary[i]
            if total + full[i] <= budget:
                c = spent + cost[i]
                if total not in nxt or c < nxt[total][0]:
                    nxt[total] = (c, chosen + (i,))
        states = nxt
        if not states:
            return None
    spent, chosen = min(states.values(), key=lambda v: v[0])
    return chosen, spent


# ---------------------------------------------------------------------------
# Part 5: measure on the GPU
# ---------------------------------------------------------------------------


def measure_peaks(layers: int, tokens: int, hidden: int) -> tuple:
    import torch
    from torch.utils.checkpoint import checkpoint

    dev = torch.device("cuda")
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        *[torch.nn.Sequential(torch.nn.Linear(hidden, hidden, bias=False), torch.nn.GELU()) for _ in range(layers)]
    ).to(dev)
    x = torch.randn(tokens, hidden, device=dev)

    def step(use_ckpt):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        y = x
        for layer in model:
            y = checkpoint(layer, y, use_reentrant=False) if use_ckpt else layer(y)
        y.pow(2).mean().backward()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - base

    step(False)
    step(True)  # warm up cuBLAS and the allocator
    return step(False), step(True)
