from pathlib import Path

N = 21504
WORDS = N // 4

NUM_WARPS = 4
WARP_WIDTH = 16
NUM_THREADS = NUM_WARPS * WARP_WIDTH

assert N % 4 == 0
assert WORDS % NUM_THREADS == 0

A = [(i % 17) - 8 for i in range(N)]
B = [((i * 3) % 13) - 6 for i in range(N)]
E = [a + b for a, b in zip(A, B)]

assert sum(E) == -17
assert min(E) == -14
assert max(E) == 14


def pack4(vals):
    out = []

    for i in range(0, len(vals), 4):
        w = 0

        for j in range(4):
            w |= (vals[i + j] & 0xff) << (8 * j)

        out.append(w)

    return out


def signed_byte(x):
    x &= 0xff
    return x - 256 if x & 0x80 else x


def emit_u32(name, vals):
    lines = [
        f'alignas(64) __global uint32_t {name}[words] = {{'
    ]

    for i in range(0, len(vals), 8):
        chunk = vals[i:i + 8]

        lines.append(
            "  " +
            ", ".join(f"0x{x:08x}u" for x in chunk) +
            ","
        )

    lines.append("};")

    return "\n".join(lines)


Ap = pack4(A)
Bp = pack4(B)
Ep = pack4(E)

partials = [0 for _ in range(NUM_THREADS)]

for tid in range(NUM_THREADS):

    for idx in range(tid, WORDS, NUM_THREADS):
        w = Ep[idx]

        partials[tid] += signed_byte(w)
        partials[tid] += signed_byte(w >> 8)
        partials[tid] += signed_byte(w >> 16)
        partials[tid] += signed_byte(w >> 24)

assert sum(partials) == -17

partial_text = ", ".join(str(x) for x in partials)

text = f"""\
static constexpr uint32_t n = {N};
static constexpr uint32_t words = {WORDS};
static constexpr uint32_t expected_num_threads = {NUM_THREADS};

{emit_u32("A_raw", Ap)}

{emit_u32("B_raw", Bp)}

{emit_u32("E_raw", Ep)}

alignas(64) __global uint32_t C_raw[words] = {{0}};

alignas(64) __global int32_t expected_partial_checksum[
    expected_num_threads
] = {{
  {partial_text}
}};
"""

Path("data").write_text(text)

Path("expected").write_text(
    f"N={N}\n"
    f"words={WORDS}\n"
    f"threads={NUM_THREADS}\n"
    f"checksum={sum(E)}\n"
    f"mismatches=0\n"
)

print(f"N={N}")
print(f"words={WORDS}")
print(f"threads={NUM_THREADS}")
print(f"expected checksum={sum(E)}")
print("expected mismatches=0")
print(f"sum(partial checksums)={sum(partials)}")
