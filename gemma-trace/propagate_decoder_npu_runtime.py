import re
import subprocess
import tempfile
from pathlib import Path
import numpy as np

BM = Path.home() / "chipyard/generators/gemmini/software/gemmini-rocc-tests/bareMetalC"

ATTN_H = BM / "gemma4_decoder_runtime_attention_full_data.h.new"
O_H    = BM / "gemma4_real_o_data.h"
OUT_H  = BM / "gemma4_decoder_runtime_o_data.h.new"

ATTN_SCALE = np.float32(7.874015719e-03)
V_NORM_SCALE = np.float32(1.077755913e-01)
O_X_SCALE = np.float32(1.077755913e-01)
O_C_SCALE = np.float32(1.414291561e-03)



def native_c_gate_up_product(gate_x, up_x):
    """Compute BF16-rounded GELU-tanh(gate) * up using native C tanhf()."""

    gate_x = np.ascontiguousarray(gate_x, dtype=np.float32)
    up_x = np.ascontiguousarray(up_x, dtype=np.float32)

    if gate_x.shape != up_x.shape:
        raise ValueError("gate_x and up_x shapes must match")

    count = gate_x.size

    c_source = r"""
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float round_to_bf16(float x)
{
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));

    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7FFFu + lsb;
    bits &= 0xFFFF0000u;

    memcpy(&x, &bits, sizeof(x));
    return x;
}

int main(int argc, char **argv)
{
    if (argc != 4)
        return 1;

    FILE *fg = fopen(argv[1], "rb");
    FILE *fu = fopen(argv[2], "rb");
    FILE *fo = fopen(argv[3], "wb");

    if (!fg || !fu || !fo)
        return 2;

    float gate;
    float up;

    while (
        fread(&gate, sizeof(float), 1, fg) == 1 &&
        fread(&up, sizeof(float), 1, fu) == 1
    ) {
        const float coeff = 0.7978845608028654f;

        float x3 = gate * gate * gate;
        float inner =
            coeff * (gate + 0.044715f * x3);

        float gelu =
            0.5f * gate * (1.0f + tanhf(inner));

        float product =
            round_to_bf16(gelu * up);

        if (fwrite(&product, sizeof(float), 1, fo) != 1)
            return 3;
    }

    fclose(fg);
    fclose(fu);
    fclose(fo);

    return 0;
}
"""

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        c_path = td / "gate_up.c"
        exe_path = td / "gate_up"
        gate_path = td / "gate.bin"
        up_path = td / "up.bin"
        out_path = td / "out.bin"

        c_path.write_text(c_source)
        gate_x.tofile(gate_path)
        up_x.tofile(up_path)

        subprocess.run(
            ["gcc", "-O2", str(c_path), "-lm", "-o", str(exe_path)],
            check=True,
        )

        subprocess.run(
            [
                str(exe_path),
                str(gate_path),
                str(up_path),
                str(out_path),
            ],
            check=True,
        )

        result = np.fromfile(out_path, dtype=np.float32)

    if result.size != count:
        raise RuntimeError(
            f"native C Gate*Up produced {result.size} values; "
            f"expected {count}"
        )

    return result.reshape(gate_x.shape)


def extract_int_array(path, symbol, shape):
    text = path.read_text()

    start = text.find(symbol)
    if start < 0:
        raise RuntimeError(f"{symbol} not found in {path}")

    eq = text.find("=", start)
    begin = text.find("{", eq)

    depth = 0
    end = None

    for i in range(begin, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end is None:
        raise RuntimeError(f"unterminated initializer for {symbol}")

    body = text[begin:end + 1]
    values = np.array(
        [int(x) for x in re.findall(r"-?\d+", body)],
        dtype=np.int64,
    )

    expected = int(np.prod(shape))
    if values.size != expected:
        raise RuntimeError(
            f"{symbol}: got {values.size} values, expected {expected}"
        )

    return values.reshape(shape)


def round_away(x):
    x = np.asarray(x, dtype=np.float32)

    return np.where(
        x >= 0,
        np.floor(x + np.float32(0.5)),
        np.ceil(x - np.float32(0.5)),
    )


def write_elem_array_2d(f, name, array):
    rows, cols = array.shape

    f.write(
        f"static const elem_t {name}"
        f"[{rows}][{cols}] = {{\n"
    )

    for r in range(rows):
        vals = ", ".join(str(int(v)) for v in array[r])
        f.write("{" + vals + "}")

        if r + 1 < rows:
            f.write(",")

        f.write("\n")

    f.write("};\n\n")


# ------------------------------------------------------------------
# Corrected Attention x V accumulator
# ------------------------------------------------------------------

attn_v = extract_int_array(
    ATTN_H,
    "DECODER_FULL_RUNTIME_ATTN_V_EXPECTED",
    (8, 14, 256),
)

# ------------------------------------------------------------------
# Runtime O input
#
# Mirrors prepare_runtime_o_input():
#
# real = attn_v * (ATTN_SCALE * V_NORM_SCALE)
# q    = roundf(real / O_X_SCALE)
# clamp [-127, 127]
#
# Layout:
# [head][token][dim] -> [token][head*256 + dim]
# ------------------------------------------------------------------

attn_v_scale = np.float32(
    ATTN_SCALE * V_NORM_SCALE
)

real = (
    attn_v.astype(np.float32)
    * attn_v_scale
)

q = round_away(
    real / O_X_SCALE
)

q = np.clip(q, -127, 127).astype(np.int8)

runtime_o_input = (
    q.transpose(1, 0, 2)
    .reshape(14, 2048)
)

print("O input checksum:", int(runtime_o_input.astype(np.int64).sum()))
print("O input range:", int(runtime_o_input.min()), int(runtime_o_input.max()))
print("O input first 10:", runtime_o_input.reshape(-1)[:10].tolist())

assert int(runtime_o_input.astype(np.int64).sum()) == 19523

# ------------------------------------------------------------------
# O projection
# ------------------------------------------------------------------

o_w = extract_int_array(
    O_H,
    "REAL_O_W_TRANSPOSED",
    (2048, 1536),
)

acc = (
    runtime_o_input.astype(np.int32)
    @ o_w.astype(np.int32)
)

scaled = (
    acc.astype(np.float32)
    * O_C_SCALE
)

# Gemmini ACC_SCALE: nearest-even
runtime_o = np.rint(scaled)
runtime_o = np.clip(runtime_o, -128, 127).astype(np.int8)

print("O output checksum:", int(runtime_o.astype(np.int64).sum()))
print("O output range:", int(runtime_o.min()), int(runtime_o.max()))
print("O output first 10:", runtime_o.reshape(-1)[:10].tolist())

assert int(runtime_o.astype(np.int64).sum()) == 820

# ------------------------------------------------------------------
# Write corrected runtime O header
# ------------------------------------------------------------------

with OUT_H.open("w") as f:
    f.write("#ifndef GEMMA4_DECODER_RUNTIME_O_DATA_H\n")
    f.write("#define GEMMA4_DECODER_RUNTIME_O_DATA_H\n\n")

    write_elem_array_2d(
        f,
        "DECODER_RUNTIME_O_INPUT_EXPECTED",
        runtime_o_input,
    )

    write_elem_array_2d(
        f,
        "DECODER_RUNTIME_O_EXPECTED",
        runtime_o,
    )

    f.write("#endif\n")

print("Wrote:", OUT_H)

# ==================================================================
# O output -> post-attention norm -> residual -> pre-FFN norm
#            -> quantized Gate/Up input
# ==================================================================

INPUT_H = BM / "gemma4_decoder_runtime_input_data.h"
FFN_NORM_H = BM / "gemma4_decoder_runtime_ffn_norm_data.h"
FFN_OUT_H = BM / "gemma4_decoder_runtime_ffn_input_data.h.new"

REAL_O_OUTPUT_SCALE = np.float32(2.519685030e-01)
REAL_GATE_X_SCALE = np.float32(1.082677171e-01)


def extract_float_array(path, symbol, shape):
    text = path.read_text()

    start = text.find(symbol)
    if start < 0:
        raise RuntimeError(f"{symbol} not found in {path}")

    eq = text.find("=", start)
    begin = text.find("{", eq)

    depth = 0
    end = None

    for i in range(begin, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end is None:
        raise RuntimeError(f"unterminated initializer for {symbol}")

    body = text[begin:end + 1]

    values = np.array(
        [
            float(x)
            for x in re.findall(
                r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?",
                body,
            )
        ],
        dtype=np.float32,
    )

    expected = int(np.prod(shape))
    if values.size != expected:
        raise RuntimeError(
            f"{symbol}: got {values.size}, expected {expected}"
        )

    return values.reshape(shape)


def bf16(x):
    x = np.asarray(x, dtype=np.float32)

    bits = x.view(np.uint32).copy()
    lsb = (bits >> np.uint32(16)) & np.uint32(1)

    bits += np.uint32(0x7FFF) + lsb
    bits &= np.uint32(0xFFFF0000)

    return bits.view(np.float32)


def write_float_array_2d(f, name, array):
    rows, cols = array.shape

    f.write(
        f"static const float {name}"
        f"[{rows}][{cols}] = {{\n"
    )

    for r in range(rows):
        values = ", ".join(
            f"{float(np.float32(v)):.9e}f"
            for v in array[r]
        )

        f.write("{" + values + "}")

        if r + 1 < rows:
            f.write(",")

        f.write("\n")

    f.write("};\n\n")


layer_input = extract_float_array(
    INPUT_H,
    "DECODER_LAYER_INPUT",
    (14, 1536),
)

post_attn_weight = extract_float_array(
    FFN_NORM_H,
    "DECODER_POST_ATTN_NORM_WEIGHT",
    (1536,),
)

pre_ffn_weight = extract_float_array(
    FFN_NORM_H,
    "DECODER_PRE_FFN_NORM_WEIGHT",
    (1536,),
)

# ---------------------------------------------------------------
# 1. Dequantize O and post-attention RMSNorm
# ---------------------------------------------------------------

runtime_post_attn_norm = np.empty(
    (14, 1536),
    dtype=np.float32,
)

for t in range(14):
    x = (
        runtime_o[t].astype(np.float32)
        * REAL_O_OUTPUT_SCALE
    )

    # C uses sequential double accumulation.
    sum_sq = 0.0
    for d in range(1536):
        xd = float(x[d])
        sum_sq += xd * xd

    mean_squared = np.float32(
        sum_sq / 1536.0
    )
    mean_squared = np.float32(
        mean_squared + np.float32(1.0e-6)
    )

    # Equivalent candidate for C powf(mean_squared, -0.5f).
    inv_rms = np.float32(
        np.float32(mean_squared) ** np.float32(-0.5)
    )

    y = (
        x
        * inv_rms
        * post_attn_weight
    ).astype(np.float32)

    runtime_post_attn_norm[t] = bf16(y)


# ---------------------------------------------------------------
# 2. First residual
# ---------------------------------------------------------------

runtime_first_residual = bf16(
    (
        layer_input
        + runtime_post_attn_norm
    ).astype(np.float32)
)


# ---------------------------------------------------------------
# 3. Pre-FFN RMSNorm
# ---------------------------------------------------------------

runtime_pre_ffn_norm = np.empty(
    (14, 1536),
    dtype=np.float32,
)

for t in range(14):
    x = runtime_first_residual[t]

    sum_sq = 0.0
    for d in range(1536):
        xd = float(x[d])
        sum_sq += xd * xd

    mean_squared = np.float32(
        sum_sq / 1536.0
    )
    mean_squared = np.float32(
        mean_squared + np.float32(1.0e-6)
    )

    inv_rms = np.float32(
        np.float32(mean_squared) ** np.float32(-0.5)
    )

    y = (
        x
        * inv_rms
        * pre_ffn_weight
    ).astype(np.float32)

    runtime_pre_ffn_norm[t] = bf16(y)


# ---------------------------------------------------------------
# 4. Quantize shared Gate/Up input
# ---------------------------------------------------------------

runtime_ffn_input = round_away(
    runtime_pre_ffn_norm
    / REAL_GATE_X_SCALE
)

runtime_ffn_input = np.clip(
    runtime_ffn_input,
    -127,
    127,
).astype(np.int8)


print(
    "Post-attn norm checksum:",
    float(runtime_post_attn_norm.astype(np.float64).sum()),
)
print(
    "First residual checksum:",
    float(runtime_first_residual.astype(np.float64).sum()),
)
print(
    "Pre-FFN norm checksum:",
    float(runtime_pre_ffn_norm.astype(np.float64).sum()),
)
print(
    "FFN input checksum:",
    int(runtime_ffn_input.astype(np.int64).sum()),
)
print(
    "FFN input range:",
    int(runtime_ffn_input.min()),
    int(runtime_ffn_input.max()),
)
print(
    "FFN input first 10:",
    runtime_ffn_input.reshape(-1)[:10].tolist(),
)

assert int(runtime_ffn_input.astype(np.int64).sum()) == 2151


# ---------------------------------------------------------------
# Write corrected FFN runtime header
# ---------------------------------------------------------------

with FFN_OUT_H.open("w") as f:
    f.write("#ifndef GEMMA4_DECODER_RUNTIME_FFN_INPUT_DATA_H\n")
    f.write("#define GEMMA4_DECODER_RUNTIME_FFN_INPUT_DATA_H\n\n")

    write_float_array_2d(
        f,
        "DECODER_RUNTIME_POST_ATTN_NORM_EXPECTED",
        runtime_post_attn_norm,
    )

    write_float_array_2d(
        f,
        "DECODER_RUNTIME_FIRST_RESIDUAL_EXPECTED",
        runtime_first_residual,
    )

    write_float_array_2d(
        f,
        "DECODER_RUNTIME_PRE_FFN_NORM_EXPECTED",
        runtime_pre_ffn_norm,
    )

    write_elem_array_2d(
        f,
        "DECODER_RUNTIME_FFN_INPUT_EXPECTED",
        runtime_ffn_input,
    )

    f.write("#endif\n")

print("Wrote:", FFN_OUT_H)

# ==================================================================
# FFN input -> Gemmini Gate + Up projections
# ==================================================================

GATE_H = BM / "gemma4_real_gate_data.h"
UP_H   = BM / "gemma4_real_up_data.h"

GATE_UP_OUT_H = (
    BM / "gemma4_decoder_runtime_gate_up_data.h.new"
)

REAL_GATE_C_SCALE = np.float32(5.167998374e-03)
REAL_UP_C_SCALE   = np.float32(2.652122639e-03)


gate_w = extract_int_array(
    GATE_H,
    "REAL_GATE_W_TRANSPOSED",
    (1536, 6144),
)

up_w = extract_int_array(
    UP_H,
    "REAL_UP_W_TRANSPOSED",
    (1536, 6144),
)


# ---------------------------------------------------------------
# Gate projection
# ---------------------------------------------------------------

gate_acc = (
    runtime_ffn_input.astype(np.int32)
    @ gate_w.astype(np.int32)
)

runtime_gate = np.rint(
    gate_acc.astype(np.float32)
    * REAL_GATE_C_SCALE
)

runtime_gate = np.clip(
    runtime_gate,
    -128,
    127,
).astype(np.int8)


# ---------------------------------------------------------------
# Up projection
# ---------------------------------------------------------------

up_acc = (
    runtime_ffn_input.astype(np.int32)
    @ up_w.astype(np.int32)
)

runtime_up = np.rint(
    up_acc.astype(np.float32)
    * REAL_UP_C_SCALE
)

runtime_up = np.clip(
    runtime_up,
    -128,
    127,
).astype(np.int8)


print(
    "Gate checksum:",
    int(runtime_gate.astype(np.int64).sum()),
)
print(
    "Gate range:",
    int(runtime_gate.min()),
    int(runtime_gate.max()),
)
print(
    "Gate first 10:",
    runtime_gate.reshape(-1)[:10].tolist(),
)

print(
    "Up checksum:",
    int(runtime_up.astype(np.int64).sum()),
)
print(
    "Up range:",
    int(runtime_up.min()),
    int(runtime_up.max()),
)
print(
    "Up first 10:",
    runtime_up.reshape(-1)[:10].tolist(),
)

assert int(runtime_gate.astype(np.int64).sum()) == -729534
assert int(runtime_up.astype(np.int64).sum()) == -3445


# ---------------------------------------------------------------
# Write corrected Gate/Up runtime header
# ---------------------------------------------------------------

with GATE_UP_OUT_H.open("w") as f:
    f.write("#ifndef GEMMA4_DECODER_RUNTIME_GATE_UP_DATA_H\n")
    f.write("#define GEMMA4_DECODER_RUNTIME_GATE_UP_DATA_H\n\n")

    write_elem_array_2d(
        f,
        "DECODER_RUNTIME_GATE_EXPECTED",
        runtime_gate,
    )

    write_elem_array_2d(
        f,
        "DECODER_RUNTIME_UP_EXPECTED",
        runtime_up,
    )

    f.write("#endif\n")

print("Wrote:", GATE_UP_OUT_H)

# ==================================================================
# Gate + Up -> GELU(Gate) * Up -> Down input
# ==================================================================

DOWN_INPUT_OUT_H = (
    BM / "gemma4_decoder_runtime_down_input_data.h.new"
)

REAL_GATE_OUTPUT_SCALE = np.float32(8.956693113e-02)
REAL_UP_OUTPUT_SCALE   = np.float32(1.525590569e-01)
REAL_DOWN_X_SCALE      = np.float32(1.732283473e+00)

GELU_COEFF = np.float32(0.7978845608028654)
GELU_CUBIC = np.float32(0.044715)


# ---------------------------------------------------------------
# Mirrors prepare_runtime_down_input()
# ---------------------------------------------------------------

gate_x = (
    runtime_gate.astype(np.float32)
    * REAL_GATE_OUTPUT_SCALE
)

up_x = (
    runtime_up.astype(np.float32)
    * REAL_UP_OUTPUT_SCALE
)

x3 = (
    gate_x * gate_x * gate_x
).astype(np.float32)

inner = (
    GELU_COEFF
    * (
        gate_x
        + GELU_CUBIC * x3
    )
).astype(np.float32)

gelu = (
    np.float32(0.5)
    * gate_x
    * (
        np.float32(1.0)
        + np.tanh(inner).astype(np.float32)
    )
).astype(np.float32)

runtime_gate_up_product = bf16(
    (
        gelu * up_x
    ).astype(np.float32)
)

# Use native-C tanhf() so the floating verifier golden exactly
# matches prepare_runtime_down_input().
runtime_gate_up_product = native_c_gate_up_product(
    gate_x,
    up_x,
)

runtime_down_input = round_away(
    runtime_gate_up_product
    / REAL_DOWN_X_SCALE
)

runtime_down_input = np.clip(
    runtime_down_input,
    -127,
    127,
).astype(np.int8)


print(
    "Gate*Up product checksum:",
    float(runtime_gate_up_product.astype(np.float64).sum()),
)
print(
    "Gate*Up product range:",
    float(runtime_gate_up_product.min()),
    float(runtime_gate_up_product.max()),
)

print(
    "Down input checksum:",
    int(runtime_down_input.astype(np.int64).sum()),
)
print(
    "Down input range:",
    int(runtime_down_input.min()),
    int(runtime_down_input.max()),
)
print(
    "Down input first 10:",
    runtime_down_input.reshape(-1)[:10].tolist(),
)


# ---------------------------------------------------------------
# Write candidate corrected Down-input header
# ---------------------------------------------------------------

with DOWN_INPUT_OUT_H.open("w") as f:
    f.write("#ifndef GEMMA4_DECODER_RUNTIME_DOWN_INPUT_DATA_H\n")
    f.write("#define GEMMA4_DECODER_RUNTIME_DOWN_INPUT_DATA_H\n\n")

    write_float_array_2d(
        f,
        "DECODER_RUNTIME_GATE_UP_PRODUCT_EXPECTED",
        runtime_gate_up_product,
    )

    write_elem_array_2d(
        f,
        "DECODER_RUNTIME_DOWN_INPUT_EXPECTED",
        runtime_down_input,
    )

    f.write("#endif\n")

print("Wrote:", DOWN_INPUT_OUT_H)

# ==================================================================
# Down input -> Gemmini Down projection
# ==================================================================

DOWN_H = BM / "gemma4_real_down_data.h"
DOWN_OUT_H = BM / "gemma4_decoder_runtime_down_data.h.new"

REAL_DOWN_C_SCALE = np.float32(2.074150369e-02)

down_w = extract_int_array(
    DOWN_H,
    "REAL_DOWN_W_TRANSPOSED",
    (6144, 1536),
)

down_acc = (
    runtime_down_input.astype(np.int32)
    @ down_w.astype(np.int32)
)

runtime_down = np.rint(
    down_acc.astype(np.float32)
    * REAL_DOWN_C_SCALE
)

runtime_down = np.clip(
    runtime_down,
    -128,
    127,
).astype(np.int8)

print(
    "Down output checksum:",
    int(runtime_down.astype(np.int64).sum()),
)
print(
    "Down output range:",
    int(runtime_down.min()),
    int(runtime_down.max()),
)
print(
    "Down output first 10:",
    runtime_down.reshape(-1)[:10].tolist(),
)

with DOWN_OUT_H.open("w") as f:
    f.write("#ifndef GEMMA4_DECODER_RUNTIME_DOWN_DATA_H\n")
    f.write("#define GEMMA4_DECODER_RUNTIME_DOWN_DATA_H\n\n")

    write_elem_array_2d(
        f,
        "DECODER_RUNTIME_DOWN_EXPECTED",
        runtime_down,
    )

    f.write("#endif\n")

print("Wrote:", DOWN_OUT_H)

# ==================================================================
# Down output -> post-FFN RMSNorm -> second residual
#             -> per-layer Gate input
# ==================================================================

POST_FFN_NORM_H = BM / "gemma4_decoder_post_ffn_norm_data.h"
POST_FFN_OUT_H = BM / "gemma4_decoder_runtime_post_ffn_data.h.new"

REAL_DOWN_OUTPUT_SCALE = np.float32(3.622047305e-01)
RUNTIME_PL_GATE_X_SCALE = np.float32(1.348031521e+01)

post_ffn_weight = extract_float_array(
    POST_FFN_NORM_H,
    "DECODER_POST_FFN_NORM_WEIGHT",
    (1536,),
)

runtime_post_ffn_norm = np.empty(
    (14, 1536),
    dtype=np.float32,
)

runtime_second_residual = np.empty(
    (14, 1536),
    dtype=np.float32,
)

runtime_per_layer_gate_x = np.empty(
    (14, 1536),
    dtype=np.int8,
)

for t in range(14):

    # Exact C-style sequential float32 accumulation.
    mean_sq = np.float32(0.0)

    for d in range(1536):
        x = np.float32(
            np.float32(runtime_down[t, d])
            * REAL_DOWN_OUTPUT_SCALE
        )

        term = np.float32(x * x)
        mean_sq = np.float32(mean_sq + term)

    mean_sq = np.float32(
        mean_sq / np.float32(1536.0)
    )

    inv_rms = np.float32(
        np.float32(1.0)
        / np.sqrt(
            np.float32(
                mean_sq + np.float32(1.0e-6)
            )
        ).astype(np.float32)
    )

    for d in range(1536):

        x = np.float32(
            np.float32(runtime_down[t, d])
            * REAL_DOWN_OUTPUT_SCALE
        )

        norm = np.float32(x * inv_rms)
        norm = np.float32(
            norm * post_ffn_weight[d]
        )
        norm = bf16(np.array([norm], dtype=np.float32))[0]

        runtime_post_ffn_norm[t, d] = norm

        residual = np.float32(
            runtime_first_residual[t, d] + norm
        )
        residual = bf16(
            np.array([residual], dtype=np.float32)
        )[0]

        runtime_second_residual[t, d] = residual

        q = round_away(
            np.array(
                [
                    np.float32(
                        residual / RUNTIME_PL_GATE_X_SCALE
                    )
                ],
                dtype=np.float32,
            )
        )[0]

        q = int(q)

        if q > 127:
            q = 127
        elif q < -127:
            q = -127

        runtime_per_layer_gate_x[t, d] = q


print(
    "Post-FFN norm checksum:",
    float(runtime_post_ffn_norm.astype(np.float64).sum()),
)

print(
    "Second residual checksum:",
    float(runtime_second_residual.astype(np.float64).sum()),
)

print(
    "Per-layer Gate X checksum:",
    int(runtime_per_layer_gate_x.astype(np.int64).sum()),
)

print(
    "Per-layer Gate X range:",
    int(runtime_per_layer_gate_x.min()),
    int(runtime_per_layer_gate_x.max()),
)

print(
    "Per-layer Gate X first 10:",
    runtime_per_layer_gate_x.reshape(-1)[:10].tolist(),
)


with POST_FFN_OUT_H.open("w") as f:
    f.write("#ifndef GEMMA4_DECODER_RUNTIME_POST_FFN_DATA_H\n")
    f.write("#define GEMMA4_DECODER_RUNTIME_POST_FFN_DATA_H\n\n")

    write_float_array_2d(
        f,
        "DECODER_RUNTIME_POST_FFN_NORM_EXPECTED",
        runtime_post_ffn_norm,
    )

    write_float_array_2d(
        f,
        "DECODER_RUNTIME_SECOND_RESIDUAL_EXPECTED",
        runtime_second_residual,
    )

    write_elem_array_2d(
        f,
        "DECODER_RUNTIME_PER_LAYER_GATE_X",
        runtime_per_layer_gate_x,
    )

    f.write("#endif\n")

print("Wrote:", POST_FFN_OUT_H)

# ==================================================================
# Per-layer Gate input -> Gemmini per-layer Gate projection
# ==================================================================

PL_GATE_H = BM / "gemma4_decoder_runtime_per_layer_gate_data.h"
PL_GATE_OUT_H = (
    BM / "gemma4_decoder_runtime_per_layer_gate_data.h.new"
)

RUNTIME_PL_GATE_C_SCALE = np.float32(2.533413284e-02)

pl_gate_w = extract_int_array(
    PL_GATE_H,
    "RUNTIME_PL_GATE_W_TRANSPOSED",
    (1536, 256),
)

pl_gate_acc = (
    runtime_per_layer_gate_x.astype(np.int32)
    @ pl_gate_w.astype(np.int32)
)

runtime_per_layer_gate = np.rint(
    pl_gate_acc.astype(np.float32)
    * RUNTIME_PL_GATE_C_SCALE
)

runtime_per_layer_gate = np.clip(
    runtime_per_layer_gate,
    -128,
    127,
).astype(np.int8)

print(
    "Per-layer Gate output checksum:",
    int(runtime_per_layer_gate.astype(np.int64).sum()),
)

print(
    "Per-layer Gate output range:",
    int(runtime_per_layer_gate.min()),
    int(runtime_per_layer_gate.max()),
)

print(
    "Per-layer Gate output first 10:",
    runtime_per_layer_gate.reshape(-1)[:10].tolist(),
)

with PL_GATE_OUT_H.open("w") as f:
    f.write("#ifndef GEMMA4_DECODER_RUNTIME_PER_LAYER_GATE_DATA_H\n")
    f.write("#define GEMMA4_DECODER_RUNTIME_PER_LAYER_GATE_DATA_H\n\n")

    f.write("#define RUNTIME_PL_GATE_I 14\n")
    f.write("#define RUNTIME_PL_GATE_K 1536\n")
    f.write("#define RUNTIME_PL_GATE_J 256\n\n")

    f.write(
        "#define RUNTIME_PL_GATE_X_SCALE "
        "((acc_scale_t)1.348031521e+01f)\n"
    )
    f.write(
        "#define RUNTIME_PL_GATE_W_SCALE "
        "((acc_scale_t)4.637710663e-05f)\n"
    )
    f.write(
        "#define RUNTIME_PL_GATE_OUTPUT_SCALE "
        "((acc_scale_t)2.467730269e-02f)\n"
    )
    f.write(
        "#define RUNTIME_PL_GATE_C_SCALE "
        "((acc_scale_t)2.533413284e-02f)\n\n"
    )

    write_elem_array_2d(
        f,
        "RUNTIME_PL_GATE_W_TRANSPOSED",
        pl_gate_w,
    )

    write_elem_array_2d(
        f,
        "DECODER_RUNTIME_PER_LAYER_GATE_EXPECTED",
        runtime_per_layer_gate,
    )

    f.write("#endif\n")

print("Wrote:", PL_GATE_OUT_H)

# ==================================================================
# Per-layer Gate -> GELU -> multiply per-layer input
#                -> BF16 -> final projection input
# ==================================================================

PL_INPUT_H = BM / "gemma4_decoder_per_layer_input_data.h"
PL_PRODUCT_OUT_H = (
    BM / "gemma4_decoder_runtime_per_layer_product_data.h.new"
)

RUNTIME_PL_GATE_OUTPUT_SCALE = np.float32(2.467730269e-02)
RUNTIME_PL_PROJ_X_SCALE = np.float32(1.181102395e-01)

per_layer_input = extract_float_array(
    PL_INPUT_H,
    "DECODER_PER_LAYER_INPUT",
    (14, 256),
)

pl_x = (
    runtime_per_layer_gate.astype(np.float32)
    * RUNTIME_PL_GATE_OUTPUT_SCALE
)

pl_x3 = (
    pl_x * pl_x * pl_x
).astype(np.float32)

pl_inner = (
    np.float32(0.7978845608028654)
    * (
        pl_x
        + np.float32(0.044715) * pl_x3
    )
).astype(np.float32)

pl_gelu = (
    np.float32(0.5)
    * pl_x
    * (
        np.float32(1.0)
        + np.tanh(pl_inner).astype(np.float32)
    )
).astype(np.float32)

runtime_per_layer_product = bf16(
    (
        pl_gelu * per_layer_input
    ).astype(np.float32)
)

runtime_per_layer_proj_x = round_away(
    runtime_per_layer_product
    / RUNTIME_PL_PROJ_X_SCALE
)

runtime_per_layer_proj_x = np.clip(
    runtime_per_layer_proj_x,
    -127,
    127,
).astype(np.int8)

print(
    "Per-layer product checksum:",
    float(runtime_per_layer_product.astype(np.float64).sum()),
)

print(
    "Per-layer product range:",
    float(runtime_per_layer_product.min()),
    float(runtime_per_layer_product.max()),
)

print(
    "Per-layer projection X checksum:",
    int(runtime_per_layer_proj_x.astype(np.int64).sum()),
)

print(
    "Per-layer projection X range:",
    int(runtime_per_layer_proj_x.min()),
    int(runtime_per_layer_proj_x.max()),
)

print(
    "Per-layer projection X first 10:",
    runtime_per_layer_proj_x.reshape(-1)[:10].tolist(),
)

with PL_PRODUCT_OUT_H.open("w") as f:
    f.write(
        "#ifndef GEMMA4_DECODER_RUNTIME_PER_LAYER_PRODUCT_DATA_H\n"
    )
    f.write(
        "#define GEMMA4_DECODER_RUNTIME_PER_LAYER_PRODUCT_DATA_H\n\n"
    )

    write_float_array_2d(
        f,
        "DECODER_RUNTIME_PER_LAYER_PRODUCT_EXPECTED",
        runtime_per_layer_product,
    )

    f.write(
        "#define RUNTIME_PL_PROJ_X_SCALE "
        "((acc_scale_t)1.181102395e-01f)\n\n"
    )

    write_elem_array_2d(
        f,
        "DECODER_RUNTIME_PER_LAYER_PROJ_X",
        runtime_per_layer_proj_x,
    )

    f.write("#endif\n")

print("Wrote:", PL_PRODUCT_OUT_H)

# ==================================================================
# Per-layer projection X -> final Gemmini per-layer projection
# ==================================================================

PL_PROJ_H = BM / "gemma4_decoder_runtime_per_layer_proj_data.h"
PL_PROJ_OUT_H = (
    BM / "gemma4_decoder_runtime_per_layer_proj_data.h.new"
)

RUNTIME_PL_PROJ_C_SCALE = np.float32(7.302627899e-03)

pl_proj_w = extract_int_array(
    PL_PROJ_H,
    "RUNTIME_PL_PROJ_W_TRANSPOSED",
    (256, 1536),
)

pl_proj_acc = (
    runtime_per_layer_proj_x.astype(np.int32)
    @ pl_proj_w.astype(np.int32)
)

runtime_per_layer_proj = np.rint(
    pl_proj_acc.astype(np.float32)
    * RUNTIME_PL_PROJ_C_SCALE
)

runtime_per_layer_proj = np.clip(
    runtime_per_layer_proj,
    -128,
    127,
).astype(np.int8)

print(
    "Per-layer projection output checksum:",
    int(runtime_per_layer_proj.astype(np.int64).sum()),
)

print(
    "Per-layer projection output range:",
    int(runtime_per_layer_proj.min()),
    int(runtime_per_layer_proj.max()),
)

print(
    "Per-layer projection output first 10:",
    runtime_per_layer_proj.reshape(-1)[:10].tolist(),
)

with PL_PROJ_OUT_H.open("w") as f:
    f.write("#ifndef GEMMA4_DECODER_RUNTIME_PER_LAYER_PROJ_DATA_H\n")
    f.write("#define GEMMA4_DECODER_RUNTIME_PER_LAYER_PROJ_DATA_H\n\n")

    f.write("#define RUNTIME_PL_PROJ_I 14\n")
    f.write("#define RUNTIME_PL_PROJ_K 256\n")
    f.write("#define RUNTIME_PL_PROJ_J 1536\n\n")

    f.write(
        "#define RUNTIME_PL_PROJ_X_SCALE "
        "((acc_scale_t)1.181102395e-01f)\n"
    )
    f.write(
        "#define RUNTIME_PL_PROJ_W_SCALE "
        "((acc_scale_t)6.182332523e-03f)\n"
    )
    f.write(
        "#define RUNTIME_PL_PROJ_OUTPUT_SCALE "
        "((acc_scale_t)9.999095649e-02f)\n"
    )
    f.write(
        "#define RUNTIME_PL_PROJ_C_SCALE "
        "((acc_scale_t)7.302627899e-03f)\n\n"
    )

    write_elem_array_2d(
        f,
        "RUNTIME_PL_PROJ_W_TRANSPOSED",
        pl_proj_w,
    )

    write_elem_array_2d(
        f,
        "DECODER_RUNTIME_PER_LAYER_PROJ_EXPECTED",
        runtime_per_layer_proj,
    )

    f.write("#endif\n")

print("Wrote:", PL_PROJ_OUT_H)

# ==================================================================
# Final projection -> post-per-layer RMSNorm -> residual
#                  -> layer scalar -> final output
# ==================================================================

FINAL_NORM_H = BM / "gemma4_decoder_final_norm_data.h"
OLD_FINAL_H = BM / "gemma4_decoder_runtime_final_data.h"
FINAL_OUT_H = BM / "gemma4_decoder_runtime_final_data.h.new"

RUNTIME_PL_PROJ_OUTPUT_SCALE = np.float32(9.999095649e-02)
DECODER_LAYER_SCALAR = np.float32(1.7822265625e-02)

post_pl_weight = extract_float_array(
    FINAL_NORM_H,
    "DECODER_POST_PER_LAYER_NORM_WEIGHT",
    (1536,),
)

# Keep the old final array only for comparison.
old_final_expected = extract_float_array(
    OLD_FINAL_H,
    "DECODER_RUNTIME_FINAL_EXPECTED",
    (14, 1536),
)

runtime_post_per_layer_norm = np.empty(
    (14, 1536),
    dtype=np.float32,
)

runtime_pre_scalar = np.empty(
    (14, 1536),
    dtype=np.float32,
)

runtime_final_output = np.empty(
    (14, 1536),
    dtype=np.float32,
)

for t in range(14):

    # Exact C-style sequential float32 accumulation.
    mean_sq = np.float32(0.0)

    for d in range(1536):
        x = np.float32(
            np.float32(runtime_per_layer_proj[t, d])
            * RUNTIME_PL_PROJ_OUTPUT_SCALE
        )

        term = np.float32(x * x)
        mean_sq = np.float32(mean_sq + term)

    mean_sq = np.float32(
        mean_sq / np.float32(1536.0)
    )

    inv_rms = np.float32(
        np.float32(1.0)
        / np.sqrt(
            np.float32(
                mean_sq + np.float32(1.0e-6)
            )
        ).astype(np.float32)
    )

    for d in range(1536):

        x = np.float32(
            np.float32(runtime_per_layer_proj[t, d])
            * RUNTIME_PL_PROJ_OUTPUT_SCALE
        )

        norm = np.float32(x * inv_rms)
        norm = np.float32(
            norm * post_pl_weight[d]
        )
        norm = bf16(
            np.array([norm], dtype=np.float32)
        )[0]

        runtime_post_per_layer_norm[t, d] = norm

        pre_scalar = np.float32(
            runtime_second_residual[t, d]
            + norm
        )

        pre_scalar = bf16(
            np.array([pre_scalar], dtype=np.float32)
        )[0]

        runtime_pre_scalar[t, d] = pre_scalar

        final = np.float32(
            pre_scalar * DECODER_LAYER_SCALAR
        )

        final = bf16(
            np.array([final], dtype=np.float32)
        )[0]

        runtime_final_output[t, d] = final


# ------------------------------------------------------------------
# Corrected propagated checksums
# ------------------------------------------------------------------

print(
    "Post-per-layer norm checksum:",
    float(runtime_post_per_layer_norm.astype(np.float64).sum()),
)

print(
    "Pre-scalar checksum:",
    float(runtime_pre_scalar.astype(np.float64).sum()),
)

print(
    "Final output checksum:",
    float(runtime_final_output.astype(np.float64).sum()),
)

print(
    "Final output range:",
    float(runtime_final_output.min()),
    float(runtime_final_output.max()),
)

print(
    "Final output first 10:",
    runtime_final_output.reshape(-1)[:10].tolist(),
)


# ------------------------------------------------------------------
# Compare corrected path with OLD stale runtime final reference.
# Diagnostic only.
# ------------------------------------------------------------------

old_diff = np.abs(
    runtime_final_output.astype(np.float32)
    - old_final_expected.astype(np.float32)
)

print(
    "Vs old runtime final mismatches:",
    int(np.count_nonzero(
        runtime_final_output != old_final_expected
    )),
    "/ 21504",
)

print(
    "Vs old runtime final max abs error:",
    float(old_diff.max()),
)

print(
    "Vs old runtime final mean abs error:",
    float(old_diff.astype(np.float64).mean()),
)


# ------------------------------------------------------------------
# Write corrected final runtime header
# ------------------------------------------------------------------

with FINAL_OUT_H.open("w") as f:
    f.write("#ifndef GEMMA4_DECODER_RUNTIME_FINAL_DATA_H\n")
    f.write("#define GEMMA4_DECODER_RUNTIME_FINAL_DATA_H\n\n")

    write_float_array_2d(
        f,
        "DECODER_RUNTIME_POST_PER_LAYER_NORM_EXPECTED",
        runtime_post_per_layer_norm,
    )

    write_float_array_2d(
        f,
        "DECODER_RUNTIME_PRE_SCALAR_EXPECTED",
        runtime_pre_scalar,
    )

    write_float_array_2d(
        f,
        "DECODER_RUNTIME_FINAL_EXPECTED",
        runtime_final_output,
    )

    f.write("#endif\n")

print("Wrote:", FINAL_OUT_H)

# ==================================================================
# Compare corrected NPU output against actual PyTorch layer output
# ==================================================================

PYTORCH_LAYER_OUTPUT = (
    Path("/home/aparna/gemma4-trace/decoder_cpu_export/layer_output.npy")
)

pytorch_layer_output = np.load(
    PYTORCH_LAYER_OUTPUT
).astype(np.float32).reshape(14, 1536)

diff = np.abs(
    runtime_final_output.astype(np.float32)
    - pytorch_layer_output
)

mismatches = int(np.count_nonzero(
    runtime_final_output != pytorch_layer_output
))

print()
print("=== Corrected NPU vs PyTorch layer output ===")
print(
    "PyTorch checksum:",
    float(pytorch_layer_output.astype(np.float64).sum()),
)
print(
    "NPU checksum:",
    float(runtime_final_output.astype(np.float64).sum()),
)
print(
    "Mismatches:",
    mismatches,
    "/ 21504",
)
print(
    "Max abs error:",
    float(diff.max()),
)
print(
    "Mean abs error:",
    float(diff.astype(np.float64).mean()),
)

# Same tolerance used by the CPU baseline.
tol = np.float32(0.5)
tol_mismatches = int(np.count_nonzero(diff > tol))

print(
    "Elements with abs error > 0.5:",
    tol_mismatches,
    "/ 21504",
)

print()
print("=== Final-stage error localization ===")

root = "/home/aparna/gemma4-trace/decoder_cpu_export"

for name, npu in [
    ("post_per_layer_norm_output.npy", runtime_post_per_layer_norm),
    ("second_residual_output.npy", runtime_second_residual),
    ("layer_output.npy", runtime_final_output),
]:
    ref = np.load(f"{root}/{name}").astype(np.float32).reshape(npu.shape)
    diff = np.abs(npu.astype(np.float32) - ref)

    print(name)
    print("  mean abs error:", float(diff.mean()))
    print("  max abs error: ", float(diff.max()))
    print("  > 0.5:", int(np.count_nonzero(diff > 0.5)))

print()
print("=== FFN error localization ===")

checks = [
    ("pre_ffn_norm_output.npy", runtime_pre_ffn_norm),
    ("second_residual_output.npy", runtime_second_residual),
]

for name, npu in checks:
    ref = np.load(f"{root}/{name}").astype(np.float32).reshape(npu.shape)
    diff = np.abs(npu.astype(np.float32) - ref)

    print(name)
    print("  NPU checksum: ", float(npu.astype(np.float64).sum()))
    print("  Ref checksum: ", float(ref.astype(np.float64).sum()))
    print("  mean abs err:", float(diff.mean()))
    print("  max abs err: ", float(diff.max()))

print()
print("=== Detailed FFN stage errors vs PyTorch ===")

ffn_checks = [
    (
        "Gate projection",
        runtime_gate.astype(np.float32) * REAL_GATE_OUTPUT_SCALE,
        "gate_proj_output.npy",
    ),
    (
        "Up projection",
        runtime_up.astype(np.float32) * REAL_UP_OUTPUT_SCALE,
        "up_proj_output.npy",
    ),
    (
        "Gate x Up",
        runtime_gate_up_product,
        "gate_up_output.npy",
    ),
    (
        "Down projection",
        runtime_down.astype(np.float32) * REAL_DOWN_OUTPUT_SCALE,
        "down_proj_output.npy",
    ),
    (
        "Post-FFN RMSNorm",
        runtime_post_ffn_norm,
        "post_ffn_norm_output.npy",
    ),
]

for label, npu, filename in ffn_checks:
    ref = np.load(f"{root}/{filename}").astype(np.float32).reshape(npu.shape)

    diff = np.abs(
        npu.astype(np.float32) -
        ref.astype(np.float32)
    )

    print(label)
    print("  NPU checksum: ", float(npu.astype(np.float64).sum()))
    print("  Ref checksum: ", float(ref.astype(np.float64).sum()))
    print("  mean abs err:", float(diff.mean()))
    print("  max abs err: ", float(diff.max()))
    print("  > 0.5:       ", int(np.count_nonzero(diff > 0.5)))

print()
print("=== Down-input quantization comparison ===")

ref_gate_up = np.load(
    f"{root}/gate_up_output.npy"
).astype(np.float32).reshape(14, 6144)

# Quantize the ideal PyTorch Gate×Up tensor using the exact
# Down-projection input scale and C roundf semantics.
ref_down_input_q = round_away(
    ref_gate_up / REAL_DOWN_X_SCALE
)
ref_down_input_q = np.clip(
    ref_down_input_q, -127, 127
).astype(np.int8)

qdiff = (
    runtime_down_input.astype(np.int16)
    - ref_down_input_q.astype(np.int16)
)

print(
    "Runtime Down X checksum:",
    int(runtime_down_input.astype(np.int64).sum()),
)
print(
    "PyTorch-quantized Down X checksum:",
    int(ref_down_input_q.astype(np.int64).sum()),
)
print(
    "Quantized X mismatches:",
    int(np.count_nonzero(runtime_down_input != ref_down_input_q)),
    "/",
    runtime_down_input.size,
)
print(
    "Mean |integer delta|:",
    float(np.abs(qdiff).mean()),
)
print(
    "Max |integer delta|:",
    int(np.abs(qdiff).max()),
)
print(
    "|integer delta| > 1:",
    int(np.count_nonzero(np.abs(qdiff) > 1)),
)
