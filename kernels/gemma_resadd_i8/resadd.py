from pathlib import Path

N = 21504

A = [(i % 17) - 8 for i in range(N)]
B = [((i * 3) % 13) - 6 for i in range(N)]
EXPECTED = [a + b for a, b in zip(A, B)]

assert sum(EXPECTED) == -17
assert min(EXPECTED) == -14
assert max(EXPECTED) == 14

def emit_i8_array(name, values):
    lines = [f'alignas(64) __global int8_t {name}[n] = {{']
    width = 32

    for i in range(0, len(values), width):
        chunk = values[i:i + width]
        lines.append("  " + ", ".join(str(x) for x in chunk) + ",")

    lines.append("};")
    return "\n".join(lines)

text = f"""\
static constexpr uint32_t n = {N};

{emit_i8_array("A_raw", A)}

{emit_i8_array("B_raw", B)}

alignas(64) __global int8_t C_raw[n] = {{0}};

alignas(64) __global volatile int32_t verify_checksum[1] = {{0}};
alignas(64) __global volatile int32_t verify_mismatches[1] = {{-1}};
"""

Path("data").write_text(text)

Path("expected").write_text(
    f"N={N}\n"
    f"checksum={sum(EXPECTED)}\n"
    f"mismatches=0\n"
)

print(f"N={N}")
print(f"expected checksum={sum(EXPECTED)}")
print("expected mismatches=0")
print(f"range=[{min(EXPECTED)}, {max(EXPECTED)}]")
