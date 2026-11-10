---
title: Blackwell tcgen05 and tensor memory
slug: 35-blackwell-tcgen05
part: "Part 3 — Matrix multiplication"
summary: Allocate and address tensor memory, pack the two tcgen05 descriptors, and count the operand traffic that a CTA pair saves. The kernel stub runs only on a B200.
minutes: 100
gpu: true
objectives:
  - "Explain why a 128 by 256 fp32 accumulator stopped fitting in registers, and where tensor memory (TMEM) puts it instead."
  - "Round an allocation to the power-of-two column count that tcgen05.alloc reserves, pack a TMEM address with the lane in bits 31:16, and say which warp reaches which lanes."
  - "Compute the TMEM footprint of an M by N accumulator for cta_group 1 and 2, and the number of accumulator stages that fit."
  - "Pack a tcgen05 instruction descriptor and a shared-memory descriptor, and name what differs from Hopper's wgmma descriptor."
  - "Count the operand bytes each CTA stages for a 1-CTA and a 2-CTA MMA, and trace the epilogue path from TMEM to a TMA store."
lab: 35-blackwell-tmem
---

# Blackwell tcgen05 and tensor memory

> [!TLDR]
> - Hopper keeps the MMA accumulator in registers. A 128 by 256 fp32 tile is 128 KiB, half of the 256 KiB register file. Blackwell keeps it in tensor memory (TMEM), a separate 256 KiB array of 128 lanes by 512 columns.
> - `tcgen05.alloc` reserves a power-of-two number of columns, at least 32 and at most 512. A 160-column request takes 256. A TMEM address has the lane in bits 31:16 and the column in bits 15:0.
> - One thread issues `tcgen05.mma`. It reads operands through a 64-bit shared-memory descriptor and a 32-bit instruction descriptor, and it signals completion with `tcgen05.commit` on an mbarrier. Warp w of a warpgroup reaches TMEM lanes 32*(w % 4) to 32*(w % 4) + 31.
> - A CTA pair (`cta_group::2`) computes a 256 by 256 tile. Each CTA stages its own 128 rows of A and half of B, so B traffic per CTA halves and total operand bytes fall to two thirds.
> - The lab models the allocator, the address, both descriptors, and the pair traffic in Python, and it runs a `tcgen05.st` and `tcgen05.ld` round trip when a B200 is configured.

[Chapter 12](/c/12-pipelining-and-async-copies) told you in about 570 words that Blackwell has tensor memory and a single-thread MMA. This chapter is the hands-on version. You allocate columns, build an address, pack both descriptors, and count bytes, so the instruction names stop being vocabulary and become functions you have written.

The reason to care is the same as for every chapter in this part: a model spends its arithmetic in matrix multiplication, and the B200 is where the 2.25 PFLOP/s of dense bf16 per GPU lives. A kernel that can't keep the tensor cores fed with operands, or can't drain finished tiles fast enough, leaves most of that on the floor. Tensor memory and CTA pairs are NVIDIA's answer to both problems.

No B200 is attached to the lab pool. The CPU checks run anywhere. The one kernel compiles for `sm_100a`, runs only when a B200 is configured, and is never sent to the A100.

## Before you start

You need three tools by name:

- **Warp specialization.** One warp loads, one warp issues the MMA, and other warps run the epilogue. [Chapter 12](/c/12-pipelining-and-async-copies) builds the ring that connects them, and [chapter 26](/c/26-hopper-gemm) builds the Hopper TMA box, the mbarrier byte count, and the `wgmma` descriptor. This chapter reuses all three ideas.
- **A descriptor is a packed word.** A shared-memory descriptor stores a start address and two byte offsets, each shifted right by 4, plus a swizzle field. You packed Hopper's in chapter 26. Blackwell's differs in two fields, and the lab tests both.
- **A layout is an address.** [Chapter 24](/c/24-cute-dsl-gemm) uses CuTe's layouts and `TiledMMA` in Python. The CuTe DSL section below shows the Blackwell versions of the same objects.

A chapter on thread block clusters and distributed shared memory is a companion to this one (chapter 34). This chapter uses clusters only to the extent a CTA pair needs them, and it links to that chapter for the rest.

## Why the accumulator left the registers

This section answers one question: what problem did a new memory solve?

Picture the Hopper consumer warpgroup in the middle of a mainloop. It holds the tile it is accumulating, a few addresses, and loop counters. On every iteration the `wgmma` instructions add to those accumulator registers, and the warpgroup can't touch them until `wgmma.wait_group` returns. When the K loop ends, the same registers feed the epilogue: convert, scale, add a bias, and store.

The accumulator is large. A 128 by 256 fp32 tile holds $128 \times 256 \times 4 = 131{,}072$ bytes, which is 128 KiB. The Hopper register file is 64K 32-bit registers per SM, which is 256 KiB (`docs/FACTS.md`, section 2), so one such tile takes half of it. A thread can hold at most 255 registers, so the tile is split across warpgroups: one `m64n256` fp32 accumulator is $64 \times 256 / 128 = 128$ registers per thread, and two consumer warpgroups hold the 128 KiB together.

> [!EXAMPLE] The register budget of a Hopper tile
> One `m64n256k16` accumulator takes 128 of a thread's 255 registers. Two consumer warpgroups of 128 threads hold $2 \times 128 \times 128 \times 4 = 131{,}072$ bytes. Doubling the tile to overlap the epilogue of one tile with the mainloop of the next would need another 128 KiB, which is the rest of the register file, and the epilogue's own temporaries would have nowhere to live.

Three costs follow from keeping the accumulator in registers:

1. **The tile has a ceiling.** The register file bounds how big a tile can be before spilling.
2. **The epilogue competes with the mainloop.** Both run on the same warps and the same registers, so a persistent kernel can't start the next tile's math until the previous tile's registers are free.
3. **The math warps hold state.** Every warp that issues `wgmma` owns accumulator registers for the life of the tile.

Blackwell removes all three by moving the accumulator out of the register file. The MMA writes its result into a dedicated memory, and the instruction that reads it back is explicit. The register file goes back to doing what registers do: hold the epilogue's math.

## Tensor memory is a 128-lane array of 32-bit columns

This section answers one question: what is tensor memory, and how do you name a cell in it?

Picture a grid. It has 128 rows, called *lanes*, and 512 columns. Each cell is 32 bits wide. Per SM that is $128 \times 512 \times 4 = 262{,}144$ bytes, which is 256 KiB, in a separate array from the register file and from shared memory ([PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html), per `docs/FACTS.md`, section 5.6).

A TMEM address is a 32-bit value. The lane is in bits 31:16 and the column is in bits 15:0. Lane 32, column 5 is therefore $32 \cdot 2^{16} + 5 = \texttt{0x200005}$, and lane 127, column 511 is `0x7F01FF`. The lab asks you to write the encoder and the decoder, and the wrong version it catches puts the two fields the other way round.

$$
\boxed{\mathrm{address} = (\hla{\mathrm{lane}} \ll 16) \;|\; \hlb{\mathrm{column}}}
$$

The $\hla{\mathrm{lane}}$ occupies the high half and the $\hlb{\mathrm{column}}$ the low half. The address is a pointer into a 2D array, not a byte offset: adding 1 moves one column along a row of 128 lanes.

```python
def tmem_address(lane, column):
    return (lane << 16) | column

def decode_tmem_address(address):
    return address >> 16, address & 0xFFFF
```

An address, once built, does arithmetic like any integer. To move down 32 lanes you add `32 << 16`. To move 8 columns right you add 8. A kernel that wants lane group $g$, column $c$ computes `base + (g * 32 << 16) + c`, which is the line you will see in the lab's kernel.

### Allocation is in whole columns, in powers of two

Shared memory and registers are sized at launch. TMEM is different: a kernel allocates it at run time with `tcgen05.alloc`. The instruction reserves a number of *columns*, so each allocation covers all 128 lanes. The rules:

- The count is a power of two.
- The minimum is 32 columns, which is 16 KiB ($128 \times 32 \times 4$ bytes).
- The maximum is 512, all of TMEM.
- One warp issues the instruction. It is `.sync.aligned`, so all 32 threads of that warp execute it.
- The instruction writes the base address into shared memory, at an address you pass. Every thread that needs the address then reads it from there.

The PTX takes the CTA group as a qualifier. These are the exact strings in CUTLASS's allocator for `cta_group::1`:

```ptx
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [smem_slot], num_columns;
tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
tcgen05.dealloc.cta_group::1.sync.aligned.b32 taddr, num_columns;
```

`relinquish_alloc_permit` says this CTA will allocate no more. `dealloc` returns the columns, and you give it the same count you allocated. CUTLASS's allocator documents that a single, fully active warp issues all allocator calls, that no two warps issue them at once, and that repeated allocations use the same warp.

> [!KEY] A request becomes the next power of two, with a floor of 32
> Request 33 columns and the hardware reserves 64. Request 129 and it reserves 256. A kernel that sizes an accumulator at 160 columns can't have two of them in separate allocations plus anything else, because each takes 256 of the 512.

Work through the numbers for an accumulator that is $N = 160$ columns wide. One `tcgen05.alloc` for it reserves 256. A second reserves another 256, and that is all 512. A third allocation fails. The lab's `TmemAllocator` models exactly this: it rounds with `tmem_alloc_columns`, places the run at the lowest free column, and raises `MemoryError` when no run is long enough.

There is a second way to use TMEM, and CUTLASS takes it. A kernel allocates all 512 columns once and places each accumulator at a column offset it chooses. No rounding applies to the pieces, so three 160-column accumulators fit, using 480 columns.

```viz
35-tmem-map
```

The figure draws both strategies. For $N = 160$ the separate allocations hold two stages and idle 192 of the 512 columns. The carved version holds three and idles 32. For $N = 128$ and $N = 256$ the two strategies agree, which is one reason kernels choose power-of-two tile widths.

| Accumulator width N | Allocated columns | Stages, one alloc each | Stages, carved from 512 |
|---|---|---|---|
| 64 | 64 | 8 | 8 |
| 96 | 128 | 4 | 5 |
| 128 | 128 | 4 | 4 |
| 160 | 256 | 2 | 3 |
| 192 | 256 | 2 | 2 |
| 256 | 256 | 2 | 2 |

The table comes from `scripts/figures/35-blackwell-tcgen05.py`, which calls the lab's functions. It is arithmetic on the rule above, not a measurement.

> [!NOTE] Placement is a model
> The lab's allocator places a run at the lowest free column. On a B200 the hardware chooses the base and writes it to shared memory, and your kernel reads it from there. Don't assume a base of 0. CUTLASS's allocator also documents an `exclusive` variant, and an allocation mode on one newer part that allows any multiple of 32 up to 576 columns. The lab doesn't model those.

### A warp reaches 32 lanes, and the choice is fixed

TMEM is not a flat array any thread can read. `tcgen05.ld` and `tcgen05.st` are warp-wide, and each warp can reach only 32 of the 128 lanes. Warp 0 of a warpgroup reaches lanes 0 to 31, warp 1 reaches 32 to 63, warp 2 reaches 64 to 95, and warp 3 reaches 96 to 127. The rule is by the warp's index within its warpgroup, so the general form is lanes $32 \cdot (w \bmod 4)$ to $32 \cdot (w \bmod 4) + 31$ ([Colfax](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/), which gives the rule; the PTX section on access restrictions is where to check it).

Warps 0 and 4 of an eight-warp CTA therefore share a lane group, and so do warps 1 and 5. The consequence is a design constraint: to read a whole 128-lane accumulator back, you need four warps, one per lane group. That is why Blackwell epilogues use a full warpgroup.

> [!WARNING] Warp 4 is not lanes 128 to 159
> A formula that multiplies the warp index by 32 without the `% 4` sends warp 4 off the end of the array. The lab calls the mistake `warp_id_without_mod_4`. The check prints that warp 4 should reach lanes 0 to 31.

## One thread issues the MMA

This section answers one question: what does a `tcgen05.mma` read, and who executes it?

Hopper's `wgmma` is a warpgroup instruction: 128 threads execute it together. `tcgen05.mma` is a single-thread instruction. One elected thread in the MMA warp issues it, and the tensor cores run it asynchronously. The other threads of the warp, and every other warp, do something else: load, or drain the previous tile.

The operands come from three places:

- The **accumulator** D is in TMEM. It always is.
- **B** is in shared memory, described by a 64-bit shared-memory descriptor.
- **A** is in shared memory, through a second descriptor, or in TMEM. When A is in TMEM it must be K-major.

Compare that with Hopper, where D lives in registers and A can be in registers or shared memory. The instruction takes the TMEM address of D (and of A when A is in TMEM), the descriptor of the shared-memory operands, the 32-bit instruction descriptor, and a predicate that says whether to accumulate into the old D or overwrite it.

```ptx
// One elected thread. d_tmem is a TMEM address, a_desc and b_desc are 64-bit.
tcgen05.mma.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, idesc, enable_input_d;
tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [mma_done_bar];
```

The second line is the completion signal. `tcgen05.commit` makes an mbarrier track all earlier `tcgen05` operations the thread issued. When they finish, the barrier gets an arrival. The warps that load operands wait on it before overwriting shared memory, and the epilogue warps wait on it before reading TMEM. CUTLASS wraps this as `umma_arrive`, and the Colfax tutorial describes the same helper ([Colfax](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/)). The exact qualifier spelling in PTX is the part to check against the ISA before you write it in a kernel.

### The shapes

For a dense kind::f16 instruction on one CTA, M is 64 or 128 and N goes up to 256, in steps of 8. K is 16 for bf16 and fp16. For a CTA pair, M is 128 or 256 ([PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html), via `docs/FACTS.md`, section 5.6). The step of N differs by source and by M, so the lab accepts any multiple of 8 from 8 to 256 and doesn't grade the finer rule.

The largest bf16 single-CTA instruction is $128 \times 256 \times 16$, which is $2 \times 128 \times 256 \times 16 = 1{,}048{,}576$ FLOP. NVIDIA's CuTe DSL guide says the `tcgen05` MMA runs at 2x to 4x the throughput of Hopper's `wgmma`, depending on the data type. FlashAttention-4's authors state 8,192 tensor-core operations per SM per clock for bf16 on B200, which gives that instruction $1{,}048{,}576 / 8{,}192 = 128$ clocks on one SM. The B200's 2.25 PFLOP/s dense bf16 figure is the HGX B200 per-GPU rating in `docs/FACTS.md`.

### The accumulator is a tile of lanes and columns

An M by N fp32 accumulator maps the row index M to the lane and the column index N to the TMEM column. A 128 by 256 tile occupies all 128 lanes and 256 columns, which is half of TMEM ([Colfax](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/)). That makes the footprint arithmetic simple:

$$
\mathrm{lanes} = M / \texttt{cta\_group}, \qquad \mathrm{columns} = N
$$

with $\mathrm{columns}$ then rounded up to a power of two if you allocate the accumulator on its own. A $64 \times 256$ tile in one CTA uses 64 lanes and 256 columns. A $256 \times 256$ tile across a pair uses 128 lanes and 256 columns in each CTA. A $128 \times 256$ tile across a pair uses 64 lanes in each.

> [!EXAMPLE] Two stages of 128 by 256, and then none left
> A $128 \times 256$ fp32 accumulator is 256 columns, so two fit in 512. That is exactly the double buffering that overlaps the epilogue of tile $i$ with the mainloop of tile $i + 1$. If operand A also sits in TMEM, it needs columns after the accumulator's, and two $128 \times 256$ stages no longer leave room. A kernel that wants two such stages therefore reads A from shared memory.

The lab's `accumulator_footprint(m, n, cta_group)` returns `(lanes, columns, allocated_columns)` and raises on a shape the instruction doesn't support. How lanes are laid out physically for $M = 64$ is a data-path layout detail in the PTX ISA that the lab doesn't grade. The count of lanes in use is arithmetic: $M / \texttt{cta\_group}$.

### The instruction descriptor

The 32-bit instruction descriptor says what kind of math this is. For dense `kind::f16` the bit fields, from CUTLASS's `mma_sm100_desc.hpp`, are:

| Field | Bits | Values |
|---|---|---|
| Saturate | 3 | 0 or 1 |
| C (accumulator) format | 4-5 | 0 is f16, 1 is f32, 2 is s32 |
| A format | 7-9 | 0 is f16, 1 is bf16, 2 is tf32 |
| B format | 10-12 | same as A |
| A negate, B negate | 13, 14 | 0 or 1 |
| A major, B major | 15, 16 | 0 is K-major, 1 is MN-major |
| N >> 3 | 17-22 | the width, in units of 8 |
| M >> 4 | 24-28 | the height, in units of 16 |

For M = 128, N = 256, bf16 inputs and an fp32 accumulator, both operands K-major, the word is built field by field. The C format of 1 at bit 4 gives `0x10`. The A format of 1 at bit 7 gives `0x80`, and the B format of 1 at bit 10 gives `0x400`. $N \gg 3 = 32$ at bit 17 gives `0x400000`, and $M \gg 4 = 8$ at bit 24 gives `0x8000000`. The sum is `0x8400490`.

```python
def instr_descriptor(m, n, a_format="bf16", b_format="bf16", c_format="f32",
                     a_major="K", b_major="K", a_negate=0, b_negate=0, saturate=0):
    word = saturate << 3
    word |= C_FORMAT[c_format] << 4
    word |= AB_FORMAT[a_format] << 7
    word |= AB_FORMAT[b_format] << 10
    word |= a_negate << 13 | b_negate << 14
    word |= MAJOR[a_major] << 15 | MAJOR[b_major] << 16
    word |= (n >> 3) << 17
    word |= (m >> 4) << 24
    return word
```

The plausible mistake is swapping the two shifted fields: `(m >> 3) << 17 | (n >> 4) << 24`. It produces a different word for every non-square shape. The lab calls it `swapped_m_n_fields`.

### The shared-memory descriptor, and what changed from Hopper

Chapter 26's Hopper descriptor has a 2-bit swizzle field at bits 62 and 63. Blackwell's descriptor is 64 bits wide too, but it differs in three places:

| Field | Hopper (chapter 26) | Blackwell tcgen05 |
|---|---|---|
| Start address, leading offset, stride offset | bits 0-13, 16-29, 32-45, each `(value & 0x3FFFF) >> 4` | the same |
| Version | none | bits 46-47, value 1 |
| Base offset | bits 49-51 | bits 49-51 |
| Layout type | 2 bits at 62-63 | 3 bits at 61-63 |
| 128-byte swizzle value | 1 | 2 |
| 64-byte swizzle value | 2 | 4 |
| 32-byte swizzle value | 3 | 6 |

The layout types on Blackwell are 0 for none, 1 for the 128-byte swizzle with a 32-byte base (`SWIZZLE_128B_BASE32B`), 2 for 128-byte, 4 for 64-byte, and 6 for 32-byte ([CUTLASS](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm100_desc.hpp)). CUTLASS also defines a bit 52, the leading-byte-offset mode, which the lab leaves at 0.

Notice something. Hopper writes the 128-byte mode as 1 shifted by 62, and Blackwell writes it as 2 shifted by 61. Both set bit 62. The 64-byte and 32-byte modes also land on the same bits in both formats. So a Hopper packer looks right in a quick comparison of the swizzle bits, and it still produces a wrong Blackwell word: it leaves the version at 0, and it has no way to write the new 128-byte mode with a 32-byte base. If you keep the Blackwell values but shift them by 62, the 128-byte value 2 lands on bit 63, which is the 64-byte mode. Start 0, leading 16, stride 1024, and the 128-byte layout pack to `0x4000404000010000`. The lab's two named mistakes are `forgot_version_bit` and `layout_type_at_bit_62`.

```python
def smem_descriptor(start, leading, stride, layout, base_offset=0):
    def enc(value):
        return (value & 0x3FFFF) >> 4
    return (enc(start) | (enc(leading) << 16) | (enc(stride) << 32)
            | (1 << 46) | (base_offset << 49) | (LAYOUT_TYPE[layout] << 61))
```

> [!WARNING] A Hopper descriptor is a silent wrong answer on Blackwell
> The packed word is still a valid 64-bit number. The instruction reads a layout that you didn't intend, and you get wrong math rather than an error. The lab's check compares the whole word.

## The epilogue reads TMEM, and registers come back

This section answers one question: how does a finished tile get from TMEM to global memory?

The path is four hops. `tcgen05.ld` moves values from TMEM into registers. The epilogue warps then do their math in registers: convert fp32 to bf16, add a bias, apply an activation. They store the result to shared memory, and a TMA store writes the tile to global memory. The MMA warp is not involved.

`tcgen05.ld` is warp-wide and comes in shapes. The shapes in the CUTLASS and Colfax material are `16x64b`, `16x128b`, `16x256b`, and `32x32b`, with a repeat count `.num` from x1 up to x128 for the narrow shapes. The simplest, in CUTLASS's string, is

```ptx
tcgen05.ld.sync.aligned.32x32b.x1.b32 {r}, [taddr];
tcgen05.wait::ld.sync.aligned;
```

In the `32x32b` shape, each of the 32 threads of the warp reads one 32-bit value from its own lane. The instruction is asynchronous with respect to the registers: you must execute `tcgen05.wait::ld` before you read `r`. A larger repeat count loads more columns per instruction and needs more registers per thread, which is the trade the epilogue tunes. The unit and limit for the largest count, as the Colfax article states them, look inconsistent, so check the PTX ISA before you rely on a number.

The reverse, `tcgen05.st`, writes registers to TMEM, with `tcgen05.wait::st`. FlashAttention-4 uses it in the other direction from a GEMM: softmax warps write the probabilities P back into TMEM as the next MMA's A operand.

Because each warp reaches only its 32 lanes, a 128-row accumulator needs four warps in the epilogue, with warp $w$ reading lanes $32 \cdot (w \bmod 4)$ onward. Those warps are a warpgroup.

```mermaid
flowchart LR
  MMA["MMA warp: tcgen05.mma"] -->|"writes D"| T["TMEM tile"]
  T -->|"tcgen05.ld, 4 warps"| R["Registers: convert, bias"]
  R --> S["Shared memory"]
  S -->|"TMA store"| G["Global memory"]
```

A persistent kernel uses two TMEM stages. While the epilogue warps drain stage 0, the MMA warp fills stage 1. The Colfax article calls this the ping-pong that `make_fragment_C` creates when you ask for two stages. When the epilogue finishes, it signals an mbarrier that tells the MMA warp the stage is free. The roles are now four: a TMA loader, the MMA issuer, the epilogue warpgroup, and sometimes a scheduler.

> [!KEY] The accumulator buffer is TMEM, so the pipeline's two sides stop sharing registers
> The mainloop owns no accumulator registers. The epilogue owns registers it can fill with its own math. The two sides meet at TMEM and an mbarrier.

## A CTA pair halves the B each CTA stages

This section answers one question: what does `cta_group::2` change, and by how much?

Picture two SMs working on one MMA. They are two CTAs in the same cluster, adjacent in the cluster, and together they issue a single instruction with `cta_group::2`. The instruction has M of 128 or 256. The accumulator is split along M: each CTA holds half of the rows in its own TMEM. A $256 \times 256$ tile gives each CTA a $128 \times 256$ accumulator, and the accumulator's footprint per CTA is the same as a 1-CTA $128 \times 256$ tile.

Only one thread in one CTA issues the pair's MMA. The Colfax tutorial states the leader is the even CTA of the pair, and its mainloop gates the issue on that CTA. The commit is a multicast, so both CTAs' mbarriers get the arrival. The TMA loads still run in each CTA, but the odd CTA's loads signal the even CTA's barrier, and the leader waits for the bytes of both ([Colfax, clusters tutorial](https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/)).

For the shared-memory fill, each CTA stages its own half of A, which is its 128 rows, and half of B. The MMA itself reads the whole B tile, with each CTA's half supplied from its own shared memory.

Count the bytes for one K = 64 bf16 slice, where each CTA owns 128 accumulator rows and N = 256 columns:

| Quantity | One CTA, 128 by 256 | Each CTA of a pair, 256 by 256 |
|---|---|---|
| A bytes | $128 \times 64 \times 2 = 16{,}384$ | $128 \times 64 \times 2 = 16{,}384$ |
| B bytes | $256 \times 64 \times 2 = 32{,}768$ | $128 \times 64 \times 2 = 16{,}384$ |
| Total | 49,152 (48 KiB) | 32,768 (32 KiB) |
| FLOP per staged byte | 85.3 | 128.0 |

B halves, and the total falls to $32{,}768 / 49{,}152 = 2/3$. The pair does the same FLOP as two 1-CTA tiles, so across the pair the operand bytes drop from 98,304 to 65,536.

$$
\boxed{\mathrm{B\ bytes\ per\ CTA} = \frac{\hlb{N}}{\hla{g}} \cdot K \cdot b}
$$

Here $\hla{g}$ is the CTA group size, 1 or 2, $\hlb{N}$ is the tile width, $K$ is the slice depth, and $b$ is the element size in bytes. A's per-CTA bytes don't depend on $g$, because each CTA still owns its own rows.

```viz
35-pair-traffic
```

The Colfax tutorial says the 256 by 256 pair "transfers half the operand data" of two 128 by 256 MMAs. Counting the tile shapes gives two thirds in total, and one half for B alone, so this chapter states B's half and the total's two thirds, which are the numbers the lab computes. FlashAttention-4's authors describe the same mechanism for B: each CTA stages half of operand B in its own shared memory.

> [!EXAMPLE] What the saved bytes buy
> Take the 227 KiB of shared memory a block can use on a B200 ([Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)). A 48 KiB stage fits four times, which is 192 KiB. A 32 KiB stage fits seven times, which is 224 KiB. Both ignore the epilogue's buffers and the barriers, so a real kernel has fewer stages. The pair gets 75% more stages from the same shared memory, which is more bytes in flight to cover memory latency.

There is a second view, on the shared-memory read side. Suppose each operand byte is read from shared memory once per MMA, and take the 128 bytes per clock per SM that FlashAttention-4 states for B200. A $128 \times 256 \times 16$ MMA reads $128 \times 16 \times 2 + 256 \times 16 \times 2 = 12{,}288$ bytes, which is 96 clocks of shared-memory reads against 128 clocks of math. In a pair, each SM supplies $4{,}096 + 4{,}096 = 8{,}192$ bytes, which is 64 clocks against the same 128. Both are a model under the read-once assumption, not a measurement. The point is the ratio: the 1-CTA kernel uses 75% of its shared-memory read bandwidth just to feed the tensor cores, and the pair uses 50%.

A pair needs the two CTAs in one cluster, adjacent by CTA id, and the first cluster dimension a multiple of 2 for CUTLASS's 2SM kernels. This chapter doesn't build clusters. Chapter 34 covers clusters and distributed shared memory, and [chapter 12](/c/12-pipelining-and-async-copies) shows the multicast that spreads one TMA load to several CTAs.

> [!NOTE] The time of the pair instruction
> A $256 \times 256 \times 16$ bf16 pair instruction is $2 \times 256 \times 256 \times 16 = 2{,}097{,}152$ FLOP. Two SMs at 8,192 FLOP per clock each do 16,384 per clock, so it takes 128 clocks, the same as one $128 \times 256 \times 16$ on one SM. The tile doubles, and the time per instruction doesn't. Both numbers are arithmetic from the 8,192 figure.

## Block-scaled kinds scale in hardware

This section answers one question: what are the `mxf8f6f4`, `mxf4`, and `nvf4` kinds, in one paragraph each?

The low-precision formats carry a *scale factor* for each small block of elements. Without hardware support, a kernel dequantizes each block with extra instructions before the MMA. Blackwell's block-scaled kinds apply the scale inside the tensor core.

The kinds differ in element type and in how many elements share a scale. `mxf8f6f4` takes the OCP microscaling types, fp8, fp6, or fp4, with 32 elements per scale factor and an 8-bit unsigned-exponent scale (`ue8m0`). `mxf4` takes fp4 with 32 elements per scale. `nvf4` is NVIDIA's fp4 format with 16 elements per scale and an `ue4m3` scale ([CUTLASS Blackwell functionality](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md)). Instruction K is 32 for `mxf8f6f4` and 64 for `mxf4` and `nvf4`. The accumulator is fp32.

Scale factors for A and B live in tensor memory too. A kernel stages them from shared memory with `tcgen05.cp`, which copies shared memory into TMEM, before it calls the MMA. The instruction descriptor changes shape for these kinds: the scale format and scale-factor ids occupy bits that the plain descriptor leaves unused. The lab packs only the dense `kind::f16` descriptor.

For why block scaling exists, what MX and NV formats do to accuracy, and how weights are quantized, see [chapter 18](/c/18-quantized-kernels). This chapter's only claim is about where the scales sit: in TMEM.

## CUTLASS 4 and the CuTe DSL expose all of it

This section answers one question: in which objects do you meet these instructions without writing PTX?

[Chapter 24](/c/24-cute-dsl-gemm) built an Ampere GEMM from `TiledMMA` and `TiledCopy`. The Blackwell kernels in the CuTe DSL use the same vocabulary with new operations ([tcgen05 programming guide](https://docs.nvidia.com/cutlass/4.6.0/media/docs/pythonDSL/mma_docs/tcgen05_programming.html)):

- **A TMEM allocator.** `tmem.allocate(num_cols=512)`, then `wait_for_alloc()` and `retrieve_ptr(...)`, which mirror `alloc`, the shared-memory slot, and the read of the base address. In C++ the same object is `Allocator1Sm` or `Allocator2Sm` with `allocate`, `free`, and `release_allocation_lock`.
- **An MMA operation with a CTA group.** `CtaGroup.ONE` or `CtaGroup.TWO`, and an operand source of `SMEM` or `TMEM` for A. The guide's examples are bf16 and fp16 at $128 \times 256 \times 16$, and block-scaled ones at $128 \times 256 \times 32$ for `mxf8f6f4` and $128 \times 256 \times 64$ for `nvf4`.
- **An accumulator fragment.** `make_fragment_C` with a stage count of 2 gives the ping-pong accumulator.
- **A TMEM-to-register copy.** `tcgen05.Ld32x32bOp(tcgen05.Repetition.x64)` is the epilogue load, and `tcgen05.Cp4x32x128bOp` is the copy of scale factors from shared memory into TMEM.
- **A pipeline.** `PipelineTmaUmma` ties the TMA loads to the MMA issue.

*UMMA* is CUTLASS's name for the instruction family, and you will see it in function names such as `umma_arrive`. The mapping to the PTX is one to one.

The worker image's CuTe DSL package and CUTLASS headers target the A100, and the lab doesn't use them. To build a CuTe DSL Blackwell GEMM you need a B200 and the package version on that image, and the guide's examples are the place to start.

## FlashAttention-4 is a user of all of it

This section answers one question: what does a production kernel do with tensor memory beyond a GEMM?

FlashAttention-4 targets the B200 ([FlashAttention-4 paper](https://arxiv.org/abs/2603.05451) and the [author's blog](https://tridao.me/blog/2026/flash4/)). The paper reports up to 1,613 TFLOP/s in bf16, 71% of peak, up to 1.3x faster than cuDNN 9.13 and up to 2.7x faster than Triton. Its appendix names the benchmark part differently from the text, so read those numbers as B200-class results and check the source before you compare.

Three of its design choices are the chapter's ideas at work:

1. **S and P share TMEM.** The MMA writes the score tile S into TMEM. The softmax warpgroups read it, and they write the probabilities P back into TMEM, where the next MMA consumes them. Rescale statistics also pass through TMEM. No accumulator registers sit in the MMA warp.
2. **Ping-pong between two tiles.** Two output tiles alternate, with two softmax warpgroups, one correction warpgroup, and one warpgroup that drives the tensor cores and TMA. This is the two-stage TMEM accumulator, used for attention.
3. **2-CTA in the backward pass.** The pair forms an $M = 256$ tile, and each CTA stages half of operand B in its own shared memory. The CTAs exchange half of $dS$ over distributed shared memory, so each one owns half the rows of $dQ$ and performs half as many global atomic reductions. The paper reports shared-memory cycles for a $128^3$ configuration falling from 3,328 to 2,688.

Its stated B200 resource ratio explains why the work is arranged this way: 8,192 tensor operations per SM per clock against 16 exponentials per clock and 128 bytes per clock of shared-memory read bandwidth. The tensor cores are so fast that the exponential unit and shared memory become the bottleneck, and the kernel is arranged around them.

## A kernel stub that allocates, writes, reads, and frees

This section answers one question: what is the smallest kernel that touches TMEM?

The lab's kernel allocates 32 columns, writes a $128 \times 32$ tensor into them with `tcgen05.st`, reads it back with `tcgen05.ld`, and frees the columns. One CTA of four warps does it. It never calls `tcgen05.mma`. The point is to exercise the allocator, the address, and the lane rule on real hardware.

```cuda
__shared__ uint32_t tmem_slot;
const int warp = threadIdx.x / 32;
const int lane = threadIdx.x % 32;

if (warp == 0) {
  uint32_t slot = static_cast<uint32_t>(__cvta_generic_to_shared(&tmem_slot));
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
               :: "r"(slot), "r"(32) : "memory");
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
__syncthreads();
asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

const uint32_t base = tmem_slot;
const uint32_t lane_base = static_cast<uint32_t>((warp % 4) * 32) << 16;
for (int c = 0; c < 32; ++c) {
  const uint32_t taddr = base + lane_base + c;
  // tcgen05.st, wait::st, tcgen05.ld, wait::ld; thread `lane` owns lane
  // `lane_base >> 16` + lane of column c.
}
```

Warp 0 allocates. The fences and `__syncthreads()` make the allocation visible to the other three warps. Each warp then adds its own lane group to the base and walks the 32 columns. In the `32x32b` shape, thread $i$ of the warp reaches lane $i$ of the group, so a $128 \times 32$ tensor maps one row to one thread.

The `tcgen05.fence` pair and the exact order of `relinquish_alloc_permit` relative to the first use are details to verify against the PTX ISA. The author of this chapter couldn't run the kernel on a B200, so the stub is untested on hardware, and the lab reports a skip rather than a failure without one. The kernel body in the lab's `solution.py` has the loop in full.

> [!TIP] Guard the kernel for the architecture-specific target
> The lab compiles with `arch="sm_100a"`, and the kernel body sits under `#if defined(__CUDA_ARCH_FEAT_SM100_ALL)`, with a `__trap()` otherwise. The `a` suffix selects architecture-specific features. The lab gates on compute capability 10.0 exactly. Whether an `sm_100a` binary runs on other compute capability 10.x parts is unconfirmed, so the lab doesn't assume it.

## What goes wrong

- **A non-power-of-two allocation.** You ask for 160 columns, plan on three accumulators, and the third `tcgen05.alloc` has no room. Each request rounds to 256. `multiple_of_32_not_power_of_two` rounds 129 to 160. The hardware rounds it to 256.
- **The lane and the column swapped.** `(column << 16) | lane` is a valid 32-bit number and points at the wrong cell. Lane 32, column 5 is `0x200005`, not `0x50020`. The lab calls the mistake `swapped_lane_column`.
- **A warp outside its lane group.** A warp that touches lanes it doesn't own has undefined behavior, and the usual cause is the missing `% 4`. A four-warp epilogue works by accident, and an eight-warp one doesn't.
- **A Hopper descriptor on Blackwell.** The version field is 1 in bits 46-47, and the layout type is 3 bits at bit 61. A packer that copies chapter 26 leaves the version at 0 and writes the layout one bit too high.
- **The M and N descriptor fields swapped.** N is stored as $N \gg 3$ at bit 17. M is stored as $M \gg 4$ at bit 24.
- **Reading TMEM before the wait.** `tcgen05.ld` is asynchronous. Use the result registers before `tcgen05.wait::ld` and you read stale values. The same holds for `tcgen05.st`.
- **Counting the pair's bytes wrong.** Each CTA still stages its own 128 rows of A. Only B halves. A model that halves everything claims a saving of one half and promises more stages than fit.
- **Building for sm_100 and running on a consumer part.** `tcgen05` and tensor memory are on data-center Blackwell, compute capability 10.0. Consumer Blackwell parts (compute capability 12.x) have 99 KB of shared memory per block and no `tcgen05` path ([Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)). A kernel tuned for the B200 won't run on them.

> [!RECAP]
> - The accumulator left the register file because a 128 by 256 fp32 tile is 128 KiB of a 256 KiB file, and the epilogue shared those registers with the mainloop. TMEM is 128 lanes by 512 columns of 32-bit cells, 256 KiB per SM.
> - `tcgen05.alloc` reserves a power-of-two column count from 32 to 512, so 129 becomes 256. A TMEM address is `(lane << 16) | column`. Warp $w$ reaches lanes $32 \cdot (w \bmod 4)$ onward.
> - One thread issues `tcgen05.mma`, reading B and usually A through a 64-bit shared-memory descriptor with version 1 and a 3-bit layout type at bit 61, and an instruction descriptor with $N \gg 3$ at bit 17 and $M \gg 4$ at bit 24. `tcgen05.commit` signals an mbarrier.
> - The epilogue is `tcgen05.ld` from TMEM into registers, then shared memory, then a TMA store. Four warps cover 128 lanes. Two TMEM stages let the epilogue of one tile overlap the mainloop of the next.
> - In a CTA pair, each CTA stages its own 128 rows of A and half of B. For a 256 by 256 bf16 tile at K = 64, a CTA stages 32 KiB instead of 48 KiB, B halves, and a 227 KiB block fits seven stages instead of four.

## Check your understanding

> [!QUESTION] A kernel asks `tcgen05.alloc` for 100 columns. How many does it reserve, and how many 100-column accumulators fit in one TMEM if each is a separate allocation?
> 128 columns, because 100 rounds up to the next power of two. $512 / 128 = 4$ accumulators fit. If the kernel allocated all 512 columns once and carved them, $\lfloor 512 / 100 \rfloor = 5$ would fit.

> [!QUESTION] What is the address of lane 96, column 256, and which warp of a warpgroup can load it?
> $96 \cdot 2^{16} + 256 = 6{,}291{,}456 + 256 = 6{,}291{,}712$, which is `0x600100`. Lane 96 is the first lane of the fourth group, so warp 3 of the warpgroup, or any warp whose index mod 4 is 3, reaches it.

> [!QUESTION] How much TMEM does a 256 by 256 fp32 accumulator take in each CTA of a pair, and in lanes and columns?
> The pair splits M, so each CTA holds $256 / 2 = 128$ lanes and 256 columns. That is $128 \times 256 \times 4 = 131{,}072$ bytes, 128 KiB, half of each SM's TMEM.

> [!QUESTION] Why can Blackwell double-buffer the accumulator without register pressure, when Hopper can't?
> A $128 \times 256$ fp32 accumulator is 128 KiB, half of Hopper's 256 KiB register file, so two would take all of it. On Blackwell it is 256 of TMEM's 512 columns, so two fit in memory the registers don't share. The epilogue and the mainloop each use a different resource.

> [!QUESTION] The instruction descriptor for M = 64, N = 128, f16 inputs, f16 accumulator, K-major, is what?
> $N \gg 3 = 16$, at bit 17, is `0x200000`. $M \gg 4 = 4$, at bit 24, is `0x4000000`. The formats are all 0, and the majors are 0. The word is `0x4200000`.

> [!QUESTION] Why does a Hopper-style descriptor packer fail for the 128-byte swizzle on Blackwell, even though the swizzle bit lands in the same place?
> It leaves the version field at 0, where Blackwell needs 1 at bits 46-47. Also, if it writes the Blackwell layout value 2 shifted by 62, the bit lands at 63 and describes the 64-byte mode. The 128-byte mode is 2 shifted by 61, which is bit 62.

> [!QUESTION] A 256 by 256 pair MMA does the FLOP of two 128 by 256 MMAs. What fraction of the operand bytes per CTA does it stage, and what fraction of B?
> Two thirds of the total, $32{,}768 / 49{,}152$, and one half of B, $16{,}384 / 32{,}768$. A is unchanged at 16,384 bytes.

> [!QUESTION] The shared-memory model says a 1-CTA 128 by 256 by 16 MMA needs 96 clocks of reads against 128 of math. Is 96 a measured number?
> No. It is arithmetic from 12,288 operand bytes and the 128 bytes per clock per SM that FlashAttention-4 states, under the assumption that each operand byte is read once. The lab measures nothing about shared-memory bandwidth.

## Lab

> [!TRY] Model tensor memory and the descriptors
> Open `35-blackwell-tmem`. `tmem_alloc_columns` is the one function the first check needs. Then implement `TmemAllocator`, `tmem_address` and `decode_tmem_address`, `warp_lane_range` and `can_access`, `accumulator_footprint` and `accumulator_stages`, `instr_descriptor`, `smem_descriptor`, and `per_cta_operand_bytes` and `operand_intensity`, all in plain Python. The checks catch the plausible mistakes named in this chapter: a non-power-of-two allocation, a swapped lane and column, a missing `% 4`, swapped M and N descriptor fields, and a Hopper-style shared-memory descriptor. The last function, `tmem_roundtrip`, is a tcgen05 kernel stub. On a machine without a B200 it prints a skip message and returns `None`. It is not compiled for `sm_80`, and the job is not sent to the A100 endpoint.

The lab reports one metric:

- `tmem_values_roundtripped`: the number of fp32 values the kernel wrote into TMEM and read back unchanged, 4,096 for a $128 \times 32$ tensor. A machine without a B200 records `skipped (no B200)`.

The descriptor bit fields in the checks come from CUTLASS's `mma_sm100_desc.hpp`. The PTX ISA tcgen05 section is the normative source, and the section on the instruction descriptor and the shared-memory descriptor is where to confirm a field before you rely on it.

## Further reading

- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html), NVIDIA. The `tcgen05` section lists tensor memory and its addressing, the allocation instructions, the access restrictions per warp, the shared-memory and instruction descriptors, the MMA shapes, and `tcgen05.commit`. It is the normative source for everything in this chapter.
- [tcgen05 MMA programming guide](https://docs.nvidia.com/cutlass/4.6.0/media/docs/pythonDSL/mma_docs/tcgen05_programming.html), CUTLASS 4.6.0 CuTe DSL. The allocator calls, `CtaGroup`, the operand source, the two-stage accumulator, the epilogue copy atoms, and block-scaled operations.
- [CUTLASS `mma_sm100_desc.hpp`](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm100_desc.hpp) and [`tmem_allocator_sm100.hpp`](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/tmem_allocator_sm100.hpp), NVIDIA. The descriptor bit layouts and the allocator strings in this chapter.
- [CUTLASS Blackwell functionality](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md), NVIDIA. Tile shapes per data type, block-scaled kinds, scale-factor types, and the cluster-shape rule for 2SM kernels.
- [CUTLASS tutorial: writing GEMM kernels using tensor memory](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/) and [GEMM with thread block clusters on Blackwell](https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/), Colfax Research. TMEM addressing, the lane rule, the epilogue load shapes, and the 2-SM pair UMMA.
- [FlashAttention-4: algorithm and kernel pipelining co-design](https://arxiv.org/abs/2603.05451), Zadouri and others, and [Tri Dao's blog post](https://tridao.me/blog/2026/flash4/). TMEM-resident S and P, the ping-pong schedule, and the 2-CTA backward pass.
- [CUDA Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html), NVIDIA. Shared memory per block, cluster limits, and the register file for compute capability 10.0.
- [Chapter 12](/c/12-pipelining-and-async-copies), for the ring and the short Blackwell section this chapter extends. [Chapter 26](/c/26-hopper-gemm), for the Hopper descriptor this chapter compares against. [Chapter 24](/c/24-cute-dsl-gemm), for the CuTe DSL objects. [Chapter 18](/c/18-quantized-kernels), for the formats behind the block-scaled kinds.
