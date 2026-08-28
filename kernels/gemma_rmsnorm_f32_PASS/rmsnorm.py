from pathlib import Path
import torch

ROWS = 14
DIM = 1536
EPS = 1.0e-6

torch.manual_seed(0)

# Deterministic, exactly reproducible FP32 input.
idx = torch.arange(ROWS * DIM, dtype=torch.int64)
x = (((idx * 17) % 257) - 128).to(torch.float32) / 64.0
x = x.reshape(ROWS, DIM)

# Deterministic RMSNorm scale around 1.0.
j = torch.arange(DIM, dtype=torch.int64)
w = 1.0 + ((((j * 13) % 31) - 15).to(torch.float32) / 256.0)

# Match Gemma4RMSNorm:
#
# mean_squared = x.pow(2).mean(-1, keepdim=True) + eps
# y = x * mean_squared^(-0.5)
# y = y * weight
mean_squared = x.float().pow(2).mean(-1, keepdim=True) + EPS
y = x.float() * torch.pow(mean_squared, -0.5)
y = y * w.float()

assert x.shape == (ROWS, DIM)
assert w.shape == (DIM,)
assert y.shape == (ROWS, DIM)

def fmt(v):
    s = f"{float(v):.9g}"
    if "e" not in s and "." not in s:
        s += ".0"
    return s + "f"

def emit_float_array(name, vals, size_expr):
    vals = vals.reshape(-1).tolist()

    lines = [
        f"alignas(64) __global float {name}[{size_expr}] = {{"
    ]

    width = 8

    for i in range(0, len(vals), width):
        chunk = vals[i:i + width]
        lines.append(
            "  " + ", ".join(fmt(v) for v in chunk) + ","
        )

    lines.append("};")
    return "\n".join(lines)

text = f"""\
static constexpr uint32_t rows = {ROWS};
static constexpr uint32_t dim = {DIM};
static constexpr uint32_t total = {ROWS * DIM};

static constexpr float rms_eps = {fmt(EPS)};

{emit_float_array("X_raw", x, "total")}

{emit_float_array("W_raw", w, "dim")}

{emit_float_array("Y_expected", y, "total")}

alignas(64) __global float Y_raw[total] = {{0}};
"""

Path("data").write_text(text)

Path("expected").write_text(
    f"rows={ROWS}\n"
    f"dim={DIM}\n"
    f"elements={ROWS * DIM}\n"
    f"eps={EPS}\n"
    f"reference=PyTorch Gemma4 RMSNorm formula\n"
    f"checksum={float(y.sum()):.9f}\n"
    f"min={float(y.min()):.9f}\n"
    f"max={float(y.max()):.9f}\n"
)

print("===== PYTORCH RMSNORM REFERENCE =====")
print("shape:", tuple(y.shape))
print("eps:", EPS)
print("checksum:", float(y.sum()))
print("min:", float(y.min()))
print("max:", float(y.max()))
