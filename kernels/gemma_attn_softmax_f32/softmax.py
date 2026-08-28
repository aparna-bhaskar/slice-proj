from pathlib import Path
import torch

# Gemma-4 E2B layer-0 sliding attention.
HEADS = 1
SEQ = 2
SLIDING_WINDOW = 512

# Gemma-4 E2B TEXT attention:
# scaling = query_pre_attn_scalar ** -0.5
# query_pre_attn_scalar = 256
# scale = 1 / sqrt(256) = 0.0625
QUERY_PRE_ATTN_SCALAR = 256.0
SCALE = QUERY_PRE_ATTN_SCALAR ** -0.5

TOTAL = HEADS * SEQ * SEQ

# We only need exp(z) where z <= 0 after subtracting row max.
#
# Store exp(-u), 0 <= u <= 16.
LUT_SIZE = 4097
LUT_MAX = 16.0
LUT_SCALE = (LUT_SIZE - 1) / LUT_MAX

torch.manual_seed(0)

idx = torch.arange(
    TOTAL,
    dtype=torch.int64,
)

# Deterministic attention-score-like values approximately [-8, 8].
SCORES = (
    (((idx * 37) % 509) - 254)
    .to(torch.float32)
    / 32.0
).reshape(
    HEADS,
    SEQ,
    SEQ,
)

# pytorch 
masked_scores = torch.full_like(
    SCORES,
    float("-inf"),
)

valid_elements = 0

for row in range(SEQ):

    # General sliding causal mask:
    #
    # valid keys:
    # max(0, row-window+1) ... row
    #
    start = max(
        0,
        row - SLIDING_WINDOW + 1,
    )

    masked_scores[
        :,
        row,
        start : row + 1,
    ] = (
        SCORES[
            :,
            row,
            start : row + 1,
        ]
        * SCALE
    )

    valid_elements += (
        HEADS
        *
        (row - start + 1)
    )


Y = torch.softmax(
    masked_scores,
    dim=-1,
    dtype=torch.float32,
)

# LUT approximation used by Muon

EXP_LUT = torch.exp(
    -torch.linspace(
        0.0,
        LUT_MAX,
        LUT_SIZE,
        dtype=torch.float32,
    )
)


def exp_neg_lut(z):

    # z is <= 0 because max was subtracted.
    u = -z

    clipped = torch.clamp(
        u,
        max=LUT_MAX,
    )

    scaled = (
        clipped
        *
        LUT_SCALE
    )

    lut_idx = torch.floor(
        scaled
    ).to(torch.int64)

    lut_idx = torch.clamp(
        lut_idx,
        max=LUT_SIZE - 2,
    )

    frac = (
        scaled
        -
        lut_idx.to(torch.float32)
    )

    e0 = EXP_LUT[lut_idx]
    e1 = EXP_LUT[lut_idx + 1]

    e = (
        e0
        +
        frac
        *
        (e1 - e0)
    )

    # exp(-16) ~= 1.1e-7.
    # Treat values at/beyond this range as zero.
    e = torch.where(
        u >= LUT_MAX,
        torch.zeros_like(e),
        e,
    )

    return e


Y_LUT = torch.zeros_like(
    SCORES
)

for h in range(HEADS):

    for row in range(SEQ):

        start = max(
            0,
            row - SLIDING_WINDOW + 1,
        )

        vals = (
            SCORES[
                h,
                row,
                start : row + 1,
            ]
            *
            SCALE
        )

        row_max = vals.max()

        z = (
            vals
            -
            row_max
        )

        exp_vals = exp_neg_lut(
            z
        )

        probs = (
            exp_vals
            /
            exp_vals.sum()
        )

        Y_LUT[
            h,
            row,
            start : row + 1,
        ] = probs


max_lut_error = float(
    torch.max(
        torch.abs(
            Y_LUT - Y
        )
    )
)

row_sum_error = float(
    torch.max(
        torch.abs(
            Y.sum(dim=-1)
            -
            1.0
        )
    )
)

# Our LUT implementation should already be extremely close
# to PyTorch before RTL is involved.
assert max_lut_error < 2.0e-6, max_lut_error


def fmt(v):

    s = f"{float(v):.9g}"

    if "e" not in s and "." not in s:
        s += ".0"

    return s + "f"


def emit(
    name,
    vals,
    size_expr,
):

    vals = (
        vals
        .reshape(-1)
        .tolist()
    )

    lines = [
        f"alignas(64) __global float {name}[{size_expr}] = {{"
    ]

    for i in range(
        0,
        len(vals),
        8,
    ):

        chunk = vals[
            i : i + 8
        ]

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
static constexpr uint32_t heads = {HEADS};
static constexpr uint32_t seq = {SEQ};

static constexpr uint32_t sliding_window =
    {SLIDING_WINDOW};

static constexpr uint32_t total =
    {TOTAL};

static constexpr float attention_scale =
    {fmt(SCALE)};

static constexpr uint32_t exp_lut_size =
    {LUT_SIZE};

static constexpr float exp_lut_max =
    {fmt(LUT_MAX)};

static constexpr float exp_lut_scale =
    {fmt(LUT_SCALE)};

{emit("SCORES_raw", SCORES, "total")}

{emit("Y_expected", Y_LUT, "total")}

{emit("EXP_LUT", EXP_LUT, "exp_lut_size")}

alignas(64)
__global float Y_raw[total] = {{0}};
"""

Path("data").write_text(
    text
)

Path("expected").write_text(
    f"heads={HEADS}\n"
    f"seq={SEQ}\n"
    f"sliding_window={SLIDING_WINDOW}\n"
    f"scale={SCALE}\n"
    f"elements={TOTAL}\n"
    f"valid_elements={valid_elements}\n"
    f"masked_elements={TOTAL - valid_elements}\n"
    f"exp_lut_size={LUT_SIZE}\n"
    f"exp_lut_max={LUT_MAX}\n"
    f"lut_vs_pytorch_max_error={max_lut_error:.9g}\n"
    f"row_sum_max_error={row_sum_error:.9g}\n"
    f"output_checksum={float(Y.sum()):.9f}\n"
    f"output_min={float(Y.min()):.9f}\n"
    f"output_max={float(Y.max()):.9f}\n"
)

print(
    "===== PYTORCH ATTENTION SOFTMAX ====="
)

print(
    "shape:",
    tuple(Y.shape),
)

print(
    "scale:",
    SCALE,
)

print(
    "sliding window:",
    SLIDING_WINDOW,
)

print(
    "valid elements:",
    valid_elements,
)

print(
    "masked elements:",
    TOTAL - valid_elements,
)

print(
    "LUT vs PyTorch max error:",
    max_lut_error,
)

print(
    "row-sum max error:",
    row_sum_error,
)

print(
    "checksum:",
    float(Y.sum()),
)
