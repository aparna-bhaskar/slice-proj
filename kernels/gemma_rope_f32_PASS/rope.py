from pathlib import Path
import torch

SEQ = 14
Q_HEADS = 8
K_HEADS = 1
DIM = 256
HALF = DIM // 2
THETA = 10000.0

torch.manual_seed(0)

# Deterministic Q/K inputs.
qi = torch.arange(SEQ * Q_HEADS * DIM, dtype=torch.int64)
ki = torch.arange(SEQ * K_HEADS * DIM, dtype=torch.int64)

Q = ((((qi * 17) % 257) - 128).float() / 64.0).reshape(
    SEQ, Q_HEADS, DIM
)

K = ((((ki * 23) % 251) - 125).float() / 64.0).reshape(
    SEQ, K_HEADS, DIM
)

# Default RoPE frequencies.
inv_freq = 1.0 / (
    THETA **
    (
        torch.arange(0, DIM, 2, dtype=torch.float32)
        / DIM
    )
)

positions = torch.arange(SEQ, dtype=torch.float32)

freqs = torch.outer(positions, inv_freq)

# HF-style embedding duplicates frequencies across the two halves.
emb = torch.cat([freqs, freqs], dim=-1)

COS = emb.cos()
SIN = emb.sin()


def rotate_half(x):
    x1 = x[..., :HALF]
    x2 = x[..., HALF:]

    return torch.cat((-x2, x1), dim=-1)


Q_ref = (
    Q * COS[:, None, :]
    + rotate_half(Q) * SIN[:, None, :]
)

K_ref = (
    K * COS[:, None, :]
    + rotate_half(K) * SIN[:, None, :]
)


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
            "  " +
            ", ".join(fmt(v) for v in chunk) +
            ","
        )

    lines.append("};")

    return "\n".join(lines)


Q_TOTAL = SEQ * Q_HEADS * DIM
K_TOTAL = SEQ * K_HEADS * DIM

text = f"""\
static constexpr uint32_t seq_len = {SEQ};
static constexpr uint32_t q_heads = {Q_HEADS};
static constexpr uint32_t k_heads = {K_HEADS};
static constexpr uint32_t head_dim = {DIM};
static constexpr uint32_t half_dim = {HALF};

static constexpr uint32_t q_total = {Q_TOTAL};
static constexpr uint32_t k_total = {K_TOTAL};

{emit("Q_raw", Q, "q_total")}

{emit("K_raw", K, "k_total")}

{emit("COS_raw", COS, "seq_len * head_dim")}

{emit("SIN_raw", SIN, "seq_len * head_dim")}

{emit("Q_expected", Q_ref, "q_total")}

{emit("K_expected", K_ref, "k_total")}

alignas(64) __global float Q_out[q_total] = {{0}};
alignas(64) __global float K_out[k_total] = {{0}};
"""

Path("data").write_text(text)

Path("expected").write_text(
    f"seq={SEQ}\n"
    f"q_heads={Q_HEADS}\n"
    f"k_heads={K_HEADS}\n"
    f"head_dim={DIM}\n"
    f"theta={THETA}\n"
    f"Q_elements={Q_TOTAL}\n"
    f"K_elements={K_TOTAL}\n"
    f"Q_checksum={float(Q_ref.sum()):.9f}\n"
    f"K_checksum={float(K_ref.sum()):.9f}\n"
)

print("===== PYTORCH RoPE REFERENCE =====")
print("Q:", tuple(Q_ref.shape))
print("K:", tuple(K_ref.shape))
print("theta:", THETA)
print("Q checksum:", float(Q_ref.sum()))
print("K checksum:", float(K_ref.sum()))
