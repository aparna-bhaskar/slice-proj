from pathlib import Path
import math
import torch
import torch.nn.functional as F

ROWS = 14
DIM = 6144
TOTAL = ROWS * DIM

# Positive-half tanh LUT:
# 0 <= z <= 8
LUT_SIZE = 1025
LUT_MAX = 8.0
LUT_SCALE = (LUT_SIZE - 1) / LUT_MAX

torch.manual_seed(0)

idx = torch.arange(TOTAL, dtype=torch.int64)

# Deterministic gate-projection-like values,
# approximately spanning [-6, 6].
X = (
    (((idx * 17) % 1021) - 510)
    .to(torch.float32)
    / 85.0
).reshape(ROWS, DIM)

# Actual Gemma activation reference.
Y = F.gelu(
    X,
    approximate="tanh",
)

# LUT used by the Muon implementation.
TANH_LUT = torch.tanh(
    torch.linspace(
        0.0,
        LUT_MAX,
        LUT_SIZE,
        dtype=torch.float32,
    )
)


def lut_tanh(z):
    az = torch.abs(z)

    clipped = torch.clamp(
        az,
        max=LUT_MAX,
    )

    scaled = clipped * LUT_SCALE

    lut_idx = torch.floor(
        scaled
    ).to(torch.int64)

    lut_idx = torch.clamp(
        lut_idx,
        max=LUT_SIZE - 2,
    )

    frac = (
        scaled -
        lut_idx.to(torch.float32)
    )

    t0 = TANH_LUT[lut_idx]
    t1 = TANH_LUT[lut_idx + 1]

    t = t0 + frac * (t1 - t0)

    # tanh saturates outside the LUT domain.
    t = torch.where(
        az >= LUT_MAX,
        torch.ones_like(t),
        t,
    )

    return torch.where(
        z < 0.0,
        -t,
        t,
    )


SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)

z = (
    SQRT_2_OVER_PI
    *
    (
        X
        +
        0.044715
        *
        X
        *
        X
        *
        X
    )
)

Y_LUT = (
    0.5
    *
    X
    *
    (
        1.0
        +
        lut_tanh(z)
    )
)

max_lut_error = float(
    torch.max(
        torch.abs(
            Y_LUT - Y
        )
    )
)

# Our LUT itself should already be very close
# to PyTorch tanh-GELU.
assert max_lut_error < 1.0e-5, max_lut_error


def fmt(v):
    s = f"{float(v):.9g}"

    if "e" not in s and "." not in s:
        s += ".0"

    return s + "f"


def emit(name, vals, size_expr):
    vals = vals.reshape(-1).tolist()

    lines = [
        f"alignas(64) __global float {name}[{size_expr}] = {{"
    ]

    for i in range(0, len(vals), 8):
        chunk = vals[i:i + 8]

        lines.append(
            "  "
            +
            ", ".join(
                fmt(v)
                for v in chunk
            )
            +
            ","
        )

    lines.append("};")

    return "\n".join(lines)


text = f"""\
static constexpr uint32_t rows = {ROWS};
static constexpr uint32_t dim = {DIM};
static constexpr uint32_t total = {TOTAL};

static constexpr uint32_t tanh_lut_size = {LUT_SIZE};

static constexpr float tanh_lut_max =
    {fmt(LUT_MAX)};

static constexpr float tanh_lut_scale =
    {fmt(LUT_SCALE)};

{emit("X_raw", X, "total")}

{emit("Y_expected", Y, "total")}

{emit("TANH_LUT", TANH_LUT, "tanh_lut_size")}

alignas(64)
__global float Y_raw[total] = {{0}};
"""

Path("data").write_text(text)

Path("expected").write_text(
    f"rows={ROWS}\n"
    f"dim={DIM}\n"
    f"elements={TOTAL}\n"
    f"activation=gelu_pytorch_tanh\n"
    f"lut_size={LUT_SIZE}\n"
    f"lut_max={LUT_MAX}\n"
    f"lut_vs_pytorch_max_error={max_lut_error:.9g}\n"
    f"checksum={float(Y.sum()):.9f}\n"
    f"min={float(Y.min()):.9f}\n"
    f"max={float(Y.max()):.9f}\n"
)

print("===== PYTORCH GELU REFERENCE =====")
print("shape:", tuple(Y.shape))
print("activation: gelu_pytorch_tanh")
print("elements:", TOTAL)
print("checksum:", float(Y.sum()))
print("min:", float(Y.min()))
print("max:", float(Y.max()))
print(
    "LUT vs PyTorch max error:",
    max_lut_error,
)
