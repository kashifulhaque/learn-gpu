"""Tests for lab 13: CuTe layout algebra.

Each documentation case below is a worked example from the CuTe documentation
(cute/02_layout_algebra.md), reproduced by hand. The property tests then check
the defining identities on many layouts, and the last test drives an index
gather with a layout. The core is pure Python and runs on any machine; the
gather runs on the GPU when one is present.
"""

import random

from lab_common import Checks, close, pick_device

NAMES = (
    "Layout",
    "coalesce",
    "composition",
    "complement",
    "logical_divide",
    "logical_product",
    "zipped_divide",
    "swizzle",
    "bank_conflict_degree",
    "gather_with_layout",
)


def run(submission):
    c = Checks()
    if not c.require(submission, *NAMES):
        return c.finish()
    S = submission
    L = S.Layout.from_str

    # ---- the layout as a function -------------------------------------------------
    c.check("size and cosize of (4,3):(1,4)", lambda: (L("(4,3):(1,4)").size(), L("(4,3):(1,4)").cosize()) == (12, 12))
    c.check("size and cosize of (4,3):(2,16)", lambda: (L("(4,3):(2,16)").size(), L("(4,3):(2,16)").cosize()) == (12, 39))
    c.check("size 1 layout 1:0 has cosize 1", lambda: (L("1:0").size(), L("1:0").cosize()) == (1, 1))

    def call_cases():
        a = L("(4,3):(1,4)")  # column-major 4x3
        b = L("(4,3):(3,1)")  # row-major 4x3
        ok = [a((2, 1)) == 6, b((2, 1)) == 7, a(9) == 9]
        # The integer coordinate 9 in shape (4,3) is (1,2): offset 1*3 + 2*1 = 5.
        ok.append(b(9) == 5)
        h = L("(2,(3,2)):(1,(2,6))")  # a hierarchical shape
        ok.append(h((1, (2, 1))) == 1 + 4 + 6)
        ok.append(h((1, 4)) == h((1, (1, 1))) == 1 + 2 + 6)  # int into a nested mode is colexicographic
        ok.append([h(i) for i in range(12)] == list(range(12)))  # compact column-major
        return all(ok), f"checks {ok}"

    c.check("coordinate to offset: tuple, int, and nested coordinates", call_cases)

    # ---- hand-verified documentation examples ------------------------------------
    cases = []

    def case(name, got, want):
        cases.append((name, lambda g=got, w=want: (str(g()) == w, f"got {g()}, expected {w}")))

    case("coalesce (2,(1,6)):(1,(6,2)) = 12:1", lambda: S.coalesce(L("(2,(1,6)):(1,(6,2))")), "12:1")
    case("by-mode coalesce gives (2,6):(1,2)", lambda: S.coalesce(L("(2,(1,6)):(1,(6,2))"), (1, 1)), "(2,6):(1,2)")
    case("coalesce keeps a gap: (2,3):(1,3)", lambda: S.coalesce(L("(2,3):(1,3)")), "(2,3):(1,3)")
    case("composition (6,2):(8,2) o (4,3):(3,1)", lambda: S.composition(L("(6,2):(8,2)"), L("(4,3):(3,1)")), "((2,2),3):((24,2),8)")
    case("composition 20:2 o (5,4):(4,1)", lambda: S.composition(L("20:2"), L("(5,4):(4,1)")), "(5,4):(8,2)")
    case(
        "composition (10,2):(16,4) o (5,4):(1,5)",
        lambda: S.composition(L("(10,2):(16,4)"), L("(5,4):(1,5)")),
        "(5,(2,2)):(16,(80,4))",
    )
    a12 = "(12,(4,8)):(59,(13,1))"
    case("by-mode composition with tiler <3:4, 8:2>", lambda: S.composition(L(a12), (L("3:4"), L("8:2"))), "(3,(2,4)):(236,(26,1))")
    case("by-mode composition with shape tiler (3,8)", lambda: S.composition(L(a12), (3, 8)), "(3,(4,2)):(59,(13,1))")
    for text, m, want in [
        ("4:1", 24, "6:4"),
        ("6:4", 24, "4:1"),
        ("(4,6):(1,4)", 24, "1:0"),
        ("4:2", 24, "(2,3):(1,8)"),
        ("(2,4):(1,6)", 24, "3:2"),
        ("(2,2):(1,6)", 24, "(3,2):(2,12)"),
    ]:
        case(f"complement({text}, {m}) = {want}", lambda t=text, m=m: S.complement(L(t), m), want)
    case(
        "logical_divide (4,2,3):(2,1,8) by 4:2",
        lambda: S.logical_divide(L("(4,2,3):(2,1,8)"), L("4:2")),
        "((2,2),(2,3)):((4,1),(2,8))",
    )
    case(
        "logical_product (2,2):(4,1) by 6:1",
        lambda: S.logical_product(L("(2,2):(4,1)"), L("6:1")),
        "((2,2),(2,3)):((4,1),(2,8))",
    )
    case(
        "zipped_divide tile of (9,(4,8)):(59,(13,1)) by <3:3, (2,4):(1,8)>",
        lambda: S.zipped_divide(L("(9,(4,8)):(59,(13,1))"), (L("3:3"), L("(2,4):(1,8)")))[0],
        "(3,(2,4)):(177,(13,2))",
    )
    case("coalesce (4,(1,2,3)):(1,(0,4,8)) = 24:1", lambda: S.coalesce(L("(4,(1,2,3)):(1,(0,4,8))")), "24:1")
    case("blocked product (2,2):(1,2) by (3,2):(1,3)", lambda: S.logical_product(L("(2,2):(1,2)"), L("(3,2):(1,3)")), "((2,2),(3,2)):((1,2),(4,12))")
    case("zipped_divide (8,6):(1,8) by (4,3)", lambda: S.zipped_divide(L("(8,6):(1,8)"), (L("4:1"), L("3:1"))), "((4,3),(2,2)):((1,8),(4,24))")
    case("zipped_divide (128,64):(64,1) by (8,16)", lambda: S.zipped_divide(L("(128,64):(64,1)"), (L("8:1"), L("16:1"))), "((8,16),(16,4)):((64,1),(512,16))")
    case("logical_divide 16:1 by 4:1", lambda: S.logical_divide(L("16:1"), L("4:1")), "(4,4):(1,4)")
    case("logical_divide 16:1 by 4:4", lambda: S.logical_divide(L("16:1"), L("4:4")), "(4,4):(4,1)")
    case("swizzle(65) with Swizzle<5,0,6> = 64", lambda: S.swizzle(65, 5, 0, 6), "64")
    case("swizzle(0b1010_0000) with Swizzle<3,0,3>", lambda: S.swizzle(0b10100000, 3, 0, 3), str(0b10100000 ^ 0b100))
    passed = 0
    for name, fn in cases:
        passed += c.check(name, fn)
    c.metric("doc_examples_passed", f"{passed}/{len(cases)}")

    # ---- identities on many layouts ------------------------------------------------
    rng = random.Random(13)

    def random_layout(max_modes=4):
        """A random layout built by shuffling the modes of a compact one, with gaps."""
        n = rng.randint(1, max_modes)
        shape = tuple(rng.choice([1, 2, 3, 4, 6, 8]) for _ in range(n))
        stride, run_ = [], 1
        for s in shape:
            stride.append(run_ * rng.choice([1, 1, 2]))
            run_ = run_ * s * 2
        return S.Layout(shape, tuple(stride)) if n > 1 else S.Layout(shape[0], stride[0])

    def coalesce_props():
        for _ in range(200):
            lay = random_layout()
            co = S.coalesce(lay)
            if co.size() != lay.size() or [co(i) for i in range(co.size())] != [lay(i) for i in range(lay.size())]:
                return False, f"coalesce changed the function of {lay}: got {co}"
            if isinstance(co.shape, tuple) and any(s == 1 for s in co.shape):
                return False, f"coalesce left a size-1 mode in {co}"
            if S.coalesce(co) != co:
                return False, f"coalesce is not idempotent on {lay}"
        return True, "200 random layouts"

    c.check("coalesce preserves the function, has no size-1 modes, and is idempotent", coalesce_props)

    def composition_props():
        for _ in range(300):
            n = rng.choice([12, 24, 36, 48])
            lhs = S.Layout(n, rng.choice([1, 2, 3]))  # a strided line
            if rng.random() < 0.5:
                lhs = S.Layout((2, n // 2), (rng.choice([1, 3]), rng.choice([8, 16])))
            rhs = S.Layout((rng.choice([1, 2, 3]), rng.choice([1, 2, 4])), (rng.choice([1, 2, 6]), rng.choice([6, 12, 24])))
            try:
                out = S.composition(lhs, rhs)
            except (ValueError, AssertionError):
                continue  # not divisible: the operation is undefined for this pair
            for i in range(rhs.size()):
                if out(i) != lhs(rhs(i)):
                    return False, f"({lhs}) o ({rhs}) = {out} disagrees at {i}"
        return True, "300 random pairs"

    c.check("composition satisfies R(i) = A(B(i)) wherever it is defined", composition_props)

    def composition_compact():
        # A compact A composed with 1D B is A's own function at B's offsets.
        a = S.Layout((4, 6), (1, 4))
        tested = 0
        for stride in (1, 2, 3, 4, 6, 12):
            for shape in (1, 2, 3, 4):
                if shape * stride > 24:
                    continue
                b = S.Layout(shape, stride)
                try:
                    r = S.composition(a, b)
                except (ValueError, AssertionError):
                    continue  # shape and stride don't divide a's modes: undefined
                tested += 1
                if [r(i) for i in range(shape)] != [i * stride for i in range(shape)]:
                    return False, f"compact o {b} = {r}"
        return tested >= 12, f"{tested} defined pairs checked"

    c.check("composing a compact layout with n:d yields the strided line", composition_compact)

    def complement_props():
        for text, m in [("4:2", 24), ("(2,2):(1,6)", 24), ("(3,2):(2,12)", 48), ("(2,3):(1,4)", 24), ("2:5", 40), ("(2,2):(8,2)", 64)]:
            lay = L(text)
            try:
                comp = S.complement(lay, m)
            except (ValueError, AssertionError):
                return False, f"complement({text}, {m}) raised"
            a_set = {lay(i) for i in range(lay.size())}
            r_set = {comp(i) for i in range(comp.size())}
            if comp.size() * lay.size() < m or comp.size() > m:
                return False, f"size of {comp} is out of range for {text}, {m}"
            offsets = [lay(i) + comp(j) for i in range(lay.size()) for j in range(comp.size())]
            if len(set(offsets)) != len(offsets):
                return False, f"complement({text}, {m}) = {comp} overlaps the input"
            if a_set & (r_set - {0}):
                return False, f"codomains of {text} and {comp} share an offset"
            strides = [d for d in (comp.stride if isinstance(comp.stride, tuple) else (comp.stride,))]
            if strides != sorted(strides):
                return False, f"strides of {comp} are not increasing"
        return True, ""

    c.check("complement is disjoint from its input, ordered, and covers the cotarget", complement_props)

    def divide_props():
        a = S.Layout((4, 2, 3), (2, 1, 8))
        d = S.logical_divide(a, L("4:2"))
        # The divide only reorders: it visits exactly the offsets of A.
        same = sorted(d(i) for i in range(d.size())) == sorted(a(i) for i in range(a.size()))
        tile = [d((i, 0)) for i in range(4)]
        return same and d.size() == a.size(), f"tile 0 reads offsets {tile}"

    c.check("logical_divide is a rearrangement of the same offsets", divide_props)

    def tile_of_matrix():
        # An 8x8 row-major matrix divided into 4x4 tiles: tile (1,0) starts at row 4.
        m = S.Layout((8, 8), (8, 1))
        z = S.zipped_divide(m, (L("4:1"), L("4:1")))
        starts = {(i, j): z(((0, 0), (i, j))) for i in range(2) for j in range(2)}
        want = {(0, 0): 0, (1, 0): 32, (0, 1): 4, (1, 1): 36}
        return starts == want, f"tile origins {starts}"

    c.check("zipped_divide of an 8x8 row-major matrix by (4,4) finds the tile origins", tile_of_matrix)

    # ---- swizzle and bank conflicts ----------------------------------------------
    def swizzle_bijective():
        for b, m, s in [(3, 0, 3), (2, 3, 3), (5, 0, 5), (3, 4, -3), (1, 2, 2)]:
            period = 1 << (m + b + abs(s))
            seen = {S.swizzle(x, b, m, s) for x in range(period)}
            if len(seen) != period:
                return False, f"Swizzle<{b},{m},{s}> is not a bijection on 0..{period - 1}"
            if any(S.swizzle(S.swizzle(x, b, m, s), b, m, s) != x for x in range(period)):
                return False, f"Swizzle<{b},{m},{s}> is not an involution"
        return True, ""

    c.check("swizzle is a bijection and its own inverse on each aligned block", swizzle_bijective)
    c.check(
        "Swizzle<3,4,3> on byte offsets equals Swizzle<3,3,3> on 2-byte elements",
        lambda: all(S.swizzle(2 * o, 3, 4, 3) == 2 * S.swizzle(o, 3, 3, 3) for o in range(8192)),
    )

    def conflict_checks():
        tile = L("(32,32):(32,1)")  # a 32x32 float tile, row-major, one word per element
        col = [tile((r, 5)) for r in range(32)]  # a warp reads column 5
        row = [tile((5, k)) for k in range(32)]  # a warp reads row 5
        unsw = S.bank_conflict_degree(col)
        c.add("a column read of a 32x32 row-major tile is a 32-way conflict", unsw == 32, f"got {unsw}")
        c.add("a row read is conflict free", S.bank_conflict_degree(row) == 1, f"got {S.bank_conflict_degree(row)}")
        padded = [r * 33 + 5 for r in range(32)]
        c.add("padding the row to 33 words removes the column conflict", S.bank_conflict_degree(padded) == 1)
        c.add("a broadcast read of one word is conflict free", S.bank_conflict_degree([7] * 32) == 1)
        c.add("stride-16 words give a 16-way conflict", S.bank_conflict_degree([16 * i for i in range(32)]) == 16)
        c.add("stride-2 words give a 2-way conflict", S.bank_conflict_degree([2 * i for i in range(32)]) == 2)
        sw_col = [S.swizzle(x, 5, 0, 5) for x in col]
        sw_row = [S.swizzle(tile((5, k)), 5, 0, 5) for k in range(32)]
        swz = S.bank_conflict_degree(sw_col)
        c.add("Swizzle<5,0,5> makes the column read conflict free", swz == 1, f"got {swz}")
        c.add("Swizzle<5,0,5> keeps the row read conflict free", S.bank_conflict_degree(sw_row) == 1)
        c.metric("conflicts_unswizzled", unsw)
        c.metric("conflicts_swizzled", swz)

    try:
        conflict_checks()
    except Exception as exc:  # noqa: BLE001 - a half-written starter must fail, not crash
        c.add("bank-conflict checks", False, f"{type(exc).__name__}: {exc}")
        c.metric("conflicts_unswizzled", "n/a")
        c.metric("conflicts_swizzled", "n/a")

    # ---- a layout drives a gather ------------------------------------------------------
    try:
        import torch
    except ImportError:
        print("      torch is not installed here, so the gather tests are skipped", flush=True)
        c.metric("gather_gbs", "skipped (no torch)")
        return c.finish()
    device = pick_device()
    print(f"      gather device = {device}", flush=True)

    def gather_transpose():
        for m_, n_ in [(1, 1), (1, 7), (5, 1), (13, 29), (64, 100), (257, 33)]:
            x = torch.randn(m_, n_, device=device)
            # Row-major M x N, read in column-major order: the colexicographic
            # walk of the layout visits the matrix in transposed order.
            lay = S.Layout((m_, n_), (n_, 1))
            got = S.gather_with_layout(x, lay).reshape(n_, m_)
            ok, detail = close(got, x.t().contiguous(), 0)
            if not ok:
                return False, f"{m_}x{n_}: {detail}"
        return True, ""

    c.check("a layout gathers the transpose of a matrix (awkward sizes, size 1)", gather_transpose)

    def gather_tiles():
        # Divide a 12x20 matrix into 4x5 tiles and gather them in tile-major order.
        x = torch.arange(12 * 20, device=device, dtype=torch.float32).reshape(12, 20)
        z = S.zipped_divide(S.Layout((12, 20), (20, 1)), (S.Layout(4, 1), S.Layout(5, 1)))
        got = S.gather_with_layout(x, z).reshape(3 * 4, 4 * 5)  # (tile id, element in tile)
        want = torch.stack(
            [x[4 * i : 4 * i + 4, 5 * j : 5 * j + 5].t().reshape(-1) for j in range(4) for i in range(3)]
        )
        # Colexicographic order: the element index runs fastest, so each row of
        # `got` is one tile with its 4 rows varying fastest. Tiles run i, then j.
        return close(got, want, 0)

    c.check("a zipped_divide layout gathers the 4x5 tiles of a 12x20 matrix", gather_tiles)

    def gather_swizzled():
        x = torch.arange(1024, device=device, dtype=torch.float32)
        lay = S.Layout((32, 32), (32, 1))
        idx = torch.tensor([S.swizzle(o, 5, 0, 5) for o in lay.offsets()], device=device)
        got = x[idx]
        perm_ok = torch.equal(torch.sort(idx).values, torch.arange(1024, device=device))
        return perm_ok and torch.equal(got.sort().values, x), "the swizzled offsets are a permutation of 0..1023"

    c.check("swizzled offsets of a 32x32 tile are a permutation", gather_swizzled)

    if device.type == "cuda":
        from lab_common import bandwidth_gbs, cuda_time

        n = 4096
        x = torch.randn(n, n, device=device)
        lay = S.Layout((n, n), (n, 1))
        idx = torch.arange(n * n, device=device).reshape(n, n).t().reshape(-1).contiguous()
        flat = x.reshape(-1)
        ms = cuda_time(lambda: flat[idx])
        # Reads 8 bytes of index and 4 bytes of data, writes 4 bytes: 16 bytes per element.
        c.metric("gather_gbs", round(bandwidth_gbs(16 * n * n, ms), 1))
        assert lay.size() == n * n
    else:
        c.metric("gather_gbs", "skipped (no GPU)")

    return c.finish()
