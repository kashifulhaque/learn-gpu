"""Tests for lab 30.

Nine checks run on the CPU with plain Python and torch-CPU references. The
tenth, a CUDA radix-digit histogram, needs a GPU and is skipped without one.

Plausible wrong implementations that these checks catch:

- no_sign_flip (check 1): int32_to_key masks to 32 bits but keeps the sign
  bit, so every negative key sorts after every positive one.
- flip_all_for_positive (check 2): float32_to_key inverts every bit of a
  positive float and only the sign bit of a negative one. That is the
  transform reversed. 1.0 maps to 0x407FFFFF instead of 0xBF800000, and -1.0
  sorts above 1.0.
- raw_float_key (check 3): sorting with float32_to_key alone. A negative NaN
  gets a key below -inf, and -0.0 sorts before +0.0, so an equal pair loses
  its stable order.
- unstable_scatter (check 4): a pass that walks the input forward but fills
  each bin from its end. Equal digits come out in reverse input order.
- ties_to_b (check 6): the merge-path search takes the element of b first on
  a tie, which breaks stability.
- keep_all_ties (check 7): radix select that returns every element equal to
  the threshold, and so more than k elements when the k-th value is tied.
- inclusive_scan_mask (check 8): top-p that keeps a token only when the mass
  through it is at most p. It drops the token that crosses the threshold.
- reused_counter (check 9): a sampler that ignores the row id and the step, so
  every row and every step share one uniform draw.
- signed_shift (check 10): the histogram kernel shifts a signed int, which
  sign-extends, so a negative key has the wrong top digit.
"""

from __future__ import annotations

import random

from lab_common import A100, Checks

NAMES = (
    "int32_to_key",
    "float32_to_key",
    "key_to_float32",
    "float32_sort_key",
    "radix_pass",
    "radix_sort",
    "merge_path_partition",
    "merge_path_merge",
    "radix_select_topk",
    "top_p_mask",
    "philox4x32_10",
    "sample_categorical",
)

MASK32 = 0xFFFFFFFF


def _bits(t):
    """The int32 bit pattern of a float32 tensor, as a list of unsigned ints."""
    import torch

    return [v & MASK32 for v in t.contiguous().view(torch.int32).tolist()]


def _floats_from_bits(words):
    import torch

    signed = [w - (1 << 32) if w >= (1 << 31) else w for w in words]
    return torch.tensor(signed, dtype=torch.int32).view(torch.float32)


def _fail(want, got):
    return False, f"expected {want}, actual {got}"


def _hex(values):
    return "[" + ", ".join(f"{int(v):#010x}" for v in values) + "]"


def _special_floats(gen):
    """Random floats plus the values that break a sloppy sort: both zeros, infinities, and NaNs."""
    import torch

    base = torch.randn(60, generator=gen) * 4
    base = torch.round(base * 2) / 2  # many exact ties
    extra = _floats_from_bits(
        [
            0x80000000,  # -0.0
            0x00000000,  # +0.0
            0xFF800000,  # -inf
            0x7F800000,  # +inf
            0x7FC00000,  # NaN
            0xFFC00000,  # NaN with the sign bit set
            0x7FC00001,  # NaN with a payload
            0x80000000,
            0x00000000,
        ]
    )
    return torch.cat([base, extra])


def run(submission):
    c = Checks()
    if not c.require(submission, *NAMES):
        return c.finish()
    try:
        import torch
    except ImportError:
        c.add("torch is installed", False, "expected torch, actual ImportError")
        return c.finish()
    reader = submission
    gen = torch.Generator().manual_seed(30)

    # 1. The smallest piece: one function.
    def int_keys():
        x = torch.tensor([-2147483648, -1, 0, 1, 2147483647], dtype=torch.int32)
        want = [0, 2147483647, 2147483648, 2147483649, 4294967295]
        try:
            got = reader.int32_to_key(x).tolist()
        except Exception as exc:  # noqa: BLE001
            return _fail(want, f"{type(exc).__name__}: {exc}")
        if got != want:
            return _fail(want, got)
        r = torch.randint(-1000, 1000, (200,), generator=gen, dtype=torch.int32)
        order = torch.sort(reader.int32_to_key(r), stable=True).indices
        want_order = torch.sort(r, stable=True).indices
        if not torch.equal(order, want_order):
            return _fail("key order equal to value order", "key order differs on 200 random int32")
        return True, f"expected {want}, actual {got}"

    c.check("int32 keys flip the sign bit and keep the order", int_keys)

    # 2. Float keys.
    def float_keys():
        # flip_all_for_positive fails here: 1.0 must be 0xBF800000.
        probe = torch.tensor([float("-inf"), -1.0, -0.0, 0.0, 1.0, float("inf")], dtype=torch.float32)
        want = [0x007FFFFF, 0x407FFFFF, 0x7FFFFFFF, 0x80000000, 0xBF800000, 0xFF800000]
        try:
            got = reader.float32_to_key(probe).tolist()
        except Exception as exc:  # noqa: BLE001
            return _fail(_hex(want), f"{type(exc).__name__}: {exc}")
        if got != want:
            return _fail(_hex(want), _hex(got))
        strict = [got[0] < got[1] < got[2], got[2] <= got[3], got[3] < got[4] < got[5]]
        if not all(strict):
            return _fail("-inf < -1 < -0.0 <= 0.0 < 1 < inf in key order", _hex(got))
        words = torch.randint(-(2**31), 2**31 - 1, (500,), generator=gen, dtype=torch.int32).view(torch.float32)
        keys = reader.float32_to_key(words)
        back = reader.key_to_float32(keys)
        if _bits(back) != _bits(words):
            return _fail("key_to_float32(float32_to_key(x)) with the same bits as x", "bits differ on random patterns")
        finite = torch.randn(300, generator=gen) * 10
        order = torch.sort(reader.float32_to_key(finite), stable=True).indices
        if not torch.equal(order, torch.sort(finite, stable=True).indices):
            return _fail("key order equal to value order on 300 random floats", "orders differ")
        return True, f"expected {_hex(want)}, actual {_hex(got)}"

    c.check("float keys are bit-exact, invertible, and ordered", float_keys)

    # 3. Sort keys: -0.0 equals +0.0, and every NaN is the largest key.
    def sort_keys():
        # raw_float_key fails here: the negative NaN would be below -inf.
        probe = _floats_from_bits([0x80000000, 0x00000000, 0x7F800000, 0x7FC00000, 0xFFC00000, 0x7FC00001])
        want = [0x80000000, 0x80000000, 0xFF800000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF]
        try:
            got = reader.float32_sort_key(probe).tolist()
        except Exception as exc:  # noqa: BLE001
            return _fail(_hex(want), f"{type(exc).__name__}: {exc}")
        if got != want:
            return _fail(_hex(want), _hex(got))
        x = _special_floats(gen)
        order = torch.sort(reader.float32_sort_key(x), stable=True).indices
        want_order = torch.sort(x, stable=True).indices
        if not torch.equal(order, want_order):
            return _fail(want_order[:12].tolist(), order[:12].tolist())
        return True, f"expected {_hex(want)}, actual {_hex(got)}"

    c.check("sort keys merge the zeros and put every NaN last", sort_keys)

    # 4. One stable pass.
    def one_pass():
        # unstable_scatter fails here: the three 1 digits must keep indices 1, 3, 7.
        keys = torch.tensor([3, 1, 2, 1, 0, 3, 2, 1], dtype=torch.int64)
        idx = torch.arange(8, dtype=torch.int64)
        want_keys, want_idx = [0, 1, 1, 1, 2, 2, 3, 3], [4, 1, 3, 7, 2, 6, 0, 5]
        try:
            ok, oi = reader.radix_pass(keys, idx, 0, 2)
        except Exception as exc:  # noqa: BLE001
            return _fail(want_idx, f"{type(exc).__name__}: {exc}")
        if ok.tolist() != want_keys or oi.tolist() != want_idx:
            return _fail(f"keys {want_keys}, indices {want_idx}", f"keys {ok.tolist()}, indices {oi.tolist()}")
        big = torch.randint(0, 2**16, (400,), generator=gen, dtype=torch.int64)
        big_idx = torch.arange(400, dtype=torch.int64)
        ok, oi = reader.radix_pass(big, big_idx, 8, 8)
        digit = (big >> 8) & 255
        ref = torch.sort(digit, stable=True).indices
        if not torch.equal(oi, ref) or not torch.equal(ok, big[ref]):
            return _fail(ref[:10].tolist(), oi[:10].tolist())
        return True, f"expected indices {want_idx}, actual {want_idx}"

    c.check("one radix pass is a stable counting sort of one digit", one_pass)

    # 5. The full sort.
    def full_sort():
        cases = []
        for n in (1, 7, 300, 1500):
            cases.append(("int32", torch.randint(-50, 50, (n,), generator=gen, dtype=torch.int32), 8))
        cases.append(("int32", torch.randint(-(2**31), 2**31 - 1, (500,), generator=gen, dtype=torch.int32), 4))
        cases.append(("float32", _special_floats(gen), 8))
        cases.append(("float32", torch.round(torch.randn(800, generator=gen) * 3), 16))
        for name, x, bits in cases:
            try:
                values, indices = reader.radix_sort(x, bits)
            except Exception as exc:  # noqa: BLE001
                return _fail("a sorted copy of x", f"{type(exc).__name__}: {exc}")
            ref_v, ref_i = torch.sort(x, stable=True)
            if not torch.equal(indices, ref_i):
                first = int((indices != ref_i).nonzero()[0])
                return _fail(
                    f"indices[{first}] = {int(ref_i[first])} for {name} n={x.numel()} bits={bits}",
                    f"indices[{first}] = {int(indices[first])}",
                )
            same = _bits(values) == _bits(ref_v) if name == "float32" else torch.equal(values, ref_v)
            if not same:
                return _fail(f"values equal to torch.sort for {name} n={x.numel()}", "values differ")
        return True, f"expected torch.sort(stable=True) on {len(cases)} cases, actual equal"

    c.check("the LSD radix sort equals torch.sort(stable=True), values and indices", full_sort)

    # 6. Merge path.
    def merge_path():
        rng = random.Random(30)
        for _ in range(40):
            a = sorted(rng.randint(0, 5) for _ in range(rng.randint(0, 9)))
            b = sorted(rng.randint(0, 5) for _ in range(rng.randint(0, 9)))
            tagged = sorted([(v, 0) for v in a] + [(v, 1) for v in b])  # ties: a before b
            for diag in range(len(a) + len(b) + 1):
                want = sum(1 for _, src in tagged[:diag] if src == 0)
                try:
                    got = reader.merge_path_partition(a, b, diag)
                except Exception as exc:  # noqa: BLE001
                    return _fail(want, f"{type(exc).__name__}: {exc}")
                if got != want:
                    # ties_to_b fails here.
                    return _fail(f"{want} from a at diagonal {diag} of a={a}, b={b}", got)
            for parts in (1, 2, 3, 5):
                merged, source = reader.merge_path_merge(a, b, parts)
                if merged != [v for v, _ in tagged] or source != [s for _, s in tagged]:
                    return _fail(
                        f"merged {[v for v, _ in tagged]} from {[s for _, s in tagged]} with {parts} parts",
                        f"merged {merged} from {source}",
                    )
        return True, "expected the stable merge on 40 random pairs, actual equal"

    c.check("merge-path diagonals and slices match a brute-force stable merge", merge_path)

    # 7. Top-k by radix select.
    def top_k():
        cases = [
            torch.randn(500, generator=gen),
            torch.round(torch.randn(300, generator=gen) * 2),  # heavy ties at the boundary
            torch.zeros(40),
            _floats_from_bits([0x80000000, 0x00000000] * 8 + [0x3F800000] * 3),
        ]
        for x in cases:
            for k in (1, 5, min(17, x.numel()), x.numel()):
                try:
                    values, indices = reader.radix_select_topk(x, k)
                except Exception as exc:  # noqa: BLE001
                    return _fail(f"{k} indices", f"{type(exc).__name__}: {exc}")
                ref_values = torch.topk(x, k).values
                ref_idx = torch.sort(x, descending=True, stable=True).indices[:k]
                if indices.numel() != k:
                    # keep_all_ties fails here.
                    return _fail(f"{k} elements for n={x.numel()}", f"{indices.numel()} elements")
                if not torch.equal(values, ref_values):
                    return _fail(f"values {ref_values[:6].tolist()}", f"values {values[:6].tolist()}")
                if not torch.equal(indices, ref_idx):
                    return _fail(f"indices {ref_idx[:6].tolist()} (lower index wins a tie)", f"indices {indices[:6].tolist()}")
        return True, "expected torch.topk values and the lowest-index tie rule, actual equal"

    c.check("radix-select top-k equals torch.topk and keeps exactly k on ties", top_k)

    # 8. Top-p without a full sort.
    def top_p():
        # The chapter 7 example: p = 0.85 keeps four of seven tokens.
        sorted_probs = [0.50, 0.20, 0.12, 0.08, 0.05, 0.03, 0.02]
        shuffle = [3, 0, 5, 1, 6, 2, 4]
        probs = torch.tensor([sorted_probs[j] for j in shuffle], dtype=torch.float32)
        try:
            got = reader.top_p_mask(probs, 0.85)
        except Exception as exc:  # noqa: BLE001
            return _fail("4 tokens kept", f"{type(exc).__name__}: {exc}")
        if int(got.sum()) != 4:
            # inclusive_scan_mask fails here: it keeps three.
            return _fail("4 tokens kept for p = 0.85", f"{int(got.sum())} kept")
        # Dyadic probabilities make every sum exact, so the boundary is not a rounding question.
        for trial in range(12):
            raw = torch.randint(0, 6, (64,), generator=gen).double() ** 2
            raw[torch.randint(0, 64, (20,), generator=gen)] = 0
            counts = raw.long()
            if int(counts.sum()) == 0:
                counts[0] = 1
            units = torch.round(counts.double() / counts.sum() * 1024).long()
            units[int(units.argmax())] += 1024 - int(units.sum())
            probs = (units.double() / 1024).float()
            for p in (0.1, 0.5, 0.9, 1.0):
                want = _reference_top_p(probs, p)
                got = reader.top_p_mask(probs, p)
                if not torch.equal(got.bool(), want):
                    bad = int((got.bool() != want).nonzero()[0])
                    return _fail(
                        f"token {bad} {'kept' if want[bad] else 'dropped'} for p={p}, trial {trial}",
                        f"token {bad} {'kept' if got[bad] else 'dropped'}",
                    )
        return True, "expected the sort-based mask on 48 dyadic cases, actual equal"

    c.check("the top-p mask equals the sort-based reference, with no full sort", top_p)

    # 9. Philox and seeded sampling.
    def philox_sampling():
        kat = (
            ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
            (
                (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
                (0xA4093822, 0x299F31D0),
                (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
            ),
        )
        for counter, key, want in kat:
            try:
                got = tuple(reader.philox4x32_10(counter, key))
            except Exception as exc:  # noqa: BLE001
                return _fail(_hex(want), f"{type(exc).__name__}: {exc}")
            if got != want:
                return _fail(f"{_hex(want)} for counter {_hex(counter)}", _hex(got))
        flat = torch.full((6, 1000), 1 / 1000)
        first = reader.sample_categorical(flat, 1234, 0)
        again = reader.sample_categorical(flat, 1234, 0)
        if not torch.equal(first, again):
            return _fail("the same tokens for the same seed", f"{first.tolist()} then {again.tolist()}")
        order = [4, 2, 5, 0, 3, 1]
        shuffled = reader.sample_categorical(flat[order], 1234, 0, row_ids=order)
        if shuffled.tolist() != [first[r].item() for r in order]:
            return _fail(
                f"{[first[r].item() for r in order]} for the permuted rows",
                f"{shuffled.tolist()}",
            )
        many = reader.sample_categorical(torch.full((64, 1000), 1 / 1000), 99, 0)
        if len(set(many.tolist())) < 32:
            # reused_counter fails here: all 64 rows share one draw.
            return _fail("at least 32 distinct tokens over 64 rows of a 1000-token vocabulary", f"{len(set(many.tolist()))} distinct")
        step0 = reader.sample_categorical(torch.full((64, 1000), 1 / 1000), 99, 0)
        step1 = reader.sample_categorical(torch.full((64, 1000), 1 / 1000), 99, 1)
        if int((step0 == step1).sum()) > 8:
            return _fail("step 0 and step 1 draws that mostly differ", f"{int((step0 == step1).sum())} of 64 rows match")
        target = torch.tensor([0.5, 0.25, 0.125, 0.125])
        draws = reader.sample_categorical(target.repeat(4000, 1), 7, 0)
        counts = torch.bincount(draws, minlength=4).double()
        expected = target.double() * 4000
        chi2 = float(((counts - expected) ** 2 / expected).sum())
        if chi2 > 16.27:  # chi-square, 3 degrees of freedom, p = 0.001
            return _fail("chi-square at most 16.27 over 4000 draws", f"chi-square {chi2:.2f}, counts {counts.tolist()}")
        return True, f"expected chi-square at most 16.27, actual {chi2:.2f}"

    c.check("Philox matches its known answers and the sampler is reproducible and unbiased", philox_sampling)

    _gpu_section(c, reader)
    return c.finish()


def _reference_top_p(probs, p):
    """Sort-based top-p: stable descending sort, keep where the mass before the token is below p."""
    import torch

    values, order = torch.sort(probs.double(), descending=True, stable=True)
    before = torch.cumsum(values, dim=0) - values
    keep_sorted = before < p
    mask = torch.zeros(probs.numel(), dtype=torch.bool)
    mask[order] = keep_sorted
    return mask


def _skip_reason():
    try:
        import torch
    except ImportError:
        return "SKIP: the CUDA radix-digit histogram was not run. torch is not installed."
    if not torch.cuda.is_available():
        return "SKIP: the CUDA radix-digit histogram was not run. No CUDA GPU is visible. It needs an A100 (sm_80)."
    return None


def _gpu_section(c: Checks, reader) -> None:
    """Kernel check. Skipped, not failed, when this machine has no CUDA GPU."""
    reason = _skip_reason()
    if reason is not None:
        print(reason, flush=True)
        c.metric("hist_gbs", "skipped (no CUDA)")
        return
    import torch

    from lab_common import bandwidth_gbs, cuda_time, require_cuda

    if not c.require(reader, "radix_histogram"):
        return
    device = require_cuda()
    gen = torch.Generator().manual_seed(30)

    def reference(keys, shift, bits):
        digit = ((keys.long() & MASK32) >> shift) & ((1 << bits) - 1)
        return torch.bincount(digit, minlength=1 << bits)

    def hist_cases():
        cases = ((1, 0, 8), (255, 24, 8), (1_000_003, 8, 8), (100_000, 0, 11), (100_000, 28, 4), (4096, 31, 1))
        for n, shift, bits in cases:
            keys = torch.randint(-(2**31), 2**31 - 1, (n,), generator=gen, dtype=torch.int32)
            try:
                got = reader.radix_histogram(keys.to(device), shift, bits).cpu().long()
            except Exception as exc:  # noqa: BLE001
                return False, f"expected {1 << bits} bins, actual {type(exc).__name__}: {exc}"
            want = reference(keys, shift, bits)
            if got.shape != want.shape or not torch.equal(got, want):
                # signed_shift fails here, on a negative key's top digit.
                bad = int((got != want).nonzero()[0]) if got.shape == want.shape else -1
                return False, f"expected bin {bad} = {int(want[bad]) if bad >= 0 else '?'} for n={n} shift={shift} bits={bits}, actual {int(got[bad]) if bad >= 0 else got.shape}"
        return True, f"expected torch.bincount on {len(cases)} cases, actual equal"

    def correct_and_fast():
        ok, detail = hist_cases()
        if not ok:
            return ok, detail
        n = 2**24
        keys = torch.randint(-(2**31), 2**31 - 1, (n,), generator=gen, dtype=torch.int32).to(device)
        reader.radix_histogram(keys, 8, 8)
        ms = cuda_time(lambda: reader.radix_histogram(keys, 8, 8), warmup=5, iters=20, flush_l2=True)
        gbs = bandwidth_gbs(4 * n, ms)
        c.metric("hist_gbs", round(gbs, 1))
        c.metric("hist_gkeys_per_s", round(n / ms / 1e6, 2))
        floor = 0.20 * A100.COPY_CEILING_GBS
        return gbs >= floor, (
            f"{detail}; expected at least {floor:.0f} GB/s counting 4n bytes, actual {gbs:.0f} GB/s"
        )

    c.check("the radix-digit histogram matches torch.bincount and reads at 20% of the copy ceiling", correct_and_fast)
