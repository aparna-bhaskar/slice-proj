from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "google/gemma-4-E2B-it"
PROMPT = "The cat sat on the"

EXPORT_DIR = Path.home() / "gemma4-trace" / "real_v_export"

HEADER_PATH = (
    Path.home()
    / "chipyard"
    / "generators"
    / "gemmini"
    / "software"
    / "gemmini-rocc-tests"
    / "bareMetalC"
    / "gemma4_real_v_data.h"
)

MODULE_NAME = "model.language_model.layers.0.self_attn.v_proj"


def quantize_symmetric_int8(
    tensor: torch.Tensor,
) -> tuple[np.ndarray, np.float32]:
    """Per-tensor symmetric INT8 quantization."""

    tensor = tensor.detach().float().cpu()

    max_abs = float(tensor.abs().max())

    if max_abs == 0.0:
        scale = np.float32(1.0)
    else:
        scale = np.float32(max_abs / 127.0)

    quantized = torch.round(tensor / float(scale))
    quantized = torch.clamp(quantized, -127, 127)
    quantized = quantized.to(torch.int8).numpy()

    return quantized, scale


def c_float(value: float | np.float32) -> str:
    """Emit enough digits to preserve a float32 value in C."""

    return f"{float(np.float32(value)):.9e}f"


def write_c_array_2d(
    file,
    name: str,
    array: np.ndarray,
    values_per_line: int = 32,
) -> None:
    rows, cols = array.shape

    file.write(
        f"static const elem_t {name}"
        f"[{rows}][{cols}] row_align(1) = {{\n"
    )

    for row_index in range(rows):
        file.write("  {\n")

        row = array[row_index]

        for start in range(0, cols, values_per_line):
            end = min(start + values_per_line, cols)

            values = ", ".join(
                str(int(value))
                for value in row[start:end]
            )

            if end < cols:
                values += ","

            file.write(f"    {values}\n")

        suffix = "," if row_index + 1 < rows else ""
        file.write(f"  }}{suffix}\n")

    file.write("};\n\n")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    HEADER_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )

    model.eval()

    modules = dict(model.named_modules())

    v_proj = modules.get(MODULE_NAME)

    if v_proj is None:
        possible_names = [
            name
            for name in modules
            if name.endswith(
                "language_model.layers.0.self_attn.v_proj"
            )
        ]

        raise RuntimeError(
            f"Could not find {MODULE_NAME}. "
            f"Possible matches: {possible_names}"
        )

    if getattr(v_proj, "bias", None) is not None:
        raise RuntimeError(
            "Layer-0 v_proj has a bias. "
            "The C test currently assumes no bias."
        )

    messages = [
        {
            "role": "user",
            "content": PROMPT,
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
    )

    input_device = model.get_input_embeddings().weight.device

    inputs = {
        name: tensor.to(input_device)
        for name, tensor in inputs.items()
    }

    captured: dict[str, torch.Tensor] = {}

    def k_hook(module, module_inputs, module_output):
        captured["x"] = (
            module_inputs[0]
            .detach()
            .float()
            .cpu()
        )

        captured["v_reference"] = (
            module_output
            .detach()
            .float()
            .cpu()
        )

    hook_handle = v_proj.register_forward_hook(k_hook)

    print("Running one prefill forward pass...")

    try:
        with torch.inference_mode():
            model(
                **inputs,
                use_cache=False,
                return_dict=True,
            )
    finally:
        hook_handle.remove()

    if "x" not in captured or "v_reference" not in captured:
        raise RuntimeError("V-projection hook did not run")

    # Remove batch dimension.
    x_float = captured["x"][0].contiguous()
    v_reference = captured["v_reference"][0].contiguous()

    # PyTorch stores Linear weights as [out_features, in_features].
    wv_float = (
        v_proj.weight
        .detach()
        .float()
        .cpu()
        .contiguous()
    )

    # Gemmini expects B as [K, J].
    wv_transposed_float = wv_float.T.contiguous()

    print("X shape:", tuple(x_float.shape))
    print("WV PyTorch shape:", tuple(wv_float.shape))
    print(
        "WV transposed shape:",
        tuple(wv_transposed_float.shape),
    )
    print(
        "V reference shape:",
        tuple(v_reference.shape),
    )

    assert tuple(x_float.shape) == (14, 1536)
    assert tuple(wv_float.shape) == (256, 1536)
    assert tuple(wv_transposed_float.shape) == (1536, 256)
    assert tuple(v_reference.shape) == (14, 256)

    # Quantize the real activation and weight.
    x_int8, x_scale = quantize_symmetric_int8(x_float)

    wv_transposed_int8, w_scale = quantize_symmetric_int8(
        wv_transposed_float
    )

    v_reference_numpy = v_reference.numpy().astype(
        np.float32
    )

    v_reference_max = float(
        np.max(np.abs(v_reference_numpy))
    )

    if v_reference_max == 0.0:
        k_output_scale = np.float32(1.0)
    else:
        k_output_scale = np.float32(
            v_reference_max / 127.0
        )

    # Integer accumulator:
    # [14,1536] @ [1536,256] -> [14,256]
    accumulator = (
        x_int8.astype(np.int32)
        @ wv_transposed_int8.astype(np.int32)
    )

    # Gemmini C_scale:
    # k_int8 = round(acc * x_scale * w_scale / k_scale)
    c_scale = np.float32(
        (np.float32(x_scale) * np.float32(w_scale))
        / np.float32(k_output_scale)
    )

    scaled = (
        accumulator.astype(np.float32)
        * np.float32(c_scale)
    )

    # np.rint performs round-to-nearest-even, matching
    # the generated Gemmini ACC_SCALE macro.
    v_expected_int8 = np.rint(scaled)
    v_expected_int8 = np.clip(
        v_expected_int8,
        -128,
        127,
    ).astype(np.int8)

    v_dequantized = (
        v_expected_int8.astype(np.float32)
        * np.float32(k_output_scale)
    )

    difference = v_dequantized - v_reference_numpy

    max_absolute_error = float(
        np.max(np.abs(difference))
    )

    mean_absolute_error = float(
        np.mean(np.abs(difference))
    )

    rmse = float(
        np.sqrt(np.mean(difference**2))
    )

    reference_flat = v_reference_numpy.reshape(-1)
    dequantized_flat = v_dequantized.reshape(-1)

    cosine_similarity = float(
        np.dot(reference_flat, dequantized_flat)
        / (
            np.linalg.norm(reference_flat)
            * np.linalg.norm(dequantized_flat)
        )
    )

    saturated_min = int(
        np.count_nonzero(v_expected_int8 == -128)
    )

    saturated_max = int(
        np.count_nonzero(v_expected_int8 == 127)
    )

    checksum = int(
        v_expected_int8.astype(np.int64).sum()
    )

    metadata = {
        "model_id": MODEL_ID,
        "prompt": PROMPT,
        "formatted_prompt": formatted_prompt,
        "module_name": MODULE_NAME,
        "input_ids_shape": list(
            inputs["input_ids"].shape
        ),
        "x_shape": list(x_int8.shape),
        "wv_pytorch_shape": list(wv_float.shape),
        "wv_transposed_shape": list(
            wv_transposed_int8.shape
        ),
        "k_shape": list(v_expected_int8.shape),
        "x_scale": float(x_scale),
        "w_scale": float(w_scale),
        "k_output_scale": float(k_output_scale),
        "gemmini_c_scale": float(c_scale),
        "max_absolute_error": max_absolute_error,
        "mean_absolute_error": mean_absolute_error,
        "rmse": rmse,
        "cosine_similarity": cosine_similarity,
        "saturated_min": saturated_min,
        "saturated_max": saturated_max,
        "expected_checksum": checksum,
    }

    # Save readable/reference files.
    np.save(EXPORT_DIR / "x_float.npy", x_float.numpy())
    np.save(EXPORT_DIR / "x_int8.npy", x_int8)

    np.save(
        EXPORT_DIR / "wv_float.npy",
        wv_float.numpy(),
    )

    np.save(
        EXPORT_DIR / "wv_transposed_int8.npy",
        wv_transposed_int8,
    )

    np.save(
        EXPORT_DIR / "v_reference_float.npy",
        v_reference_numpy,
    )

    np.save(
        EXPORT_DIR / "v_expected_int8.npy",
        v_expected_int8,
    )

    np.save(
        EXPORT_DIR / "v_dequantized.npy",
        v_dequantized,
    )

    with open(
        EXPORT_DIR / "metadata.json",
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    # Generate a C header containing actual quantized tensors.
    with open(
        HEADER_PATH,
        "w",
        encoding="utf-8",
    ) as header:
        header.write(
            "#ifndef GEMMA4_REAL_V_DATA_H\n"
            "#define GEMMA4_REAL_V_DATA_H\n\n"
        )

        header.write("#define REAL_V_I 14\n")
        header.write("#define REAL_V_K 1536\n")
        header.write("#define REAL_V_J 256\n\n")

        header.write(
            "#define REAL_V_X_SCALE "
            f"((acc_scale_t){c_float(x_scale)})\n"
        )

        header.write(
            "#define REAL_V_W_SCALE "
            f"((acc_scale_t){c_float(w_scale)})\n"
        )

        header.write(
            "#define REAL_V_OUTPUT_SCALE "
            f"((acc_scale_t){c_float(k_output_scale)})\n"
        )

        header.write(
            "#define REAL_V_C_SCALE "
            f"((acc_scale_t){c_float(c_scale)})\n\n"
        )

        write_c_array_2d(
            header,
            "REAL_V_X",
            x_int8,
        )

        write_c_array_2d(
            header,
            "REAL_V_W_TRANSPOSED",
            wv_transposed_int8,
        )

        write_c_array_2d(
            header,
            "REAL_V_EXPECTED",
            v_expected_int8,
        )

        header.write("#endif\n")

    print("\nExport completed")
    print("Header:", HEADER_PATH)
    print("Metadata:", EXPORT_DIR / "metadata.json")

    print("\nQuantization scales")
    print("X scale:", float(x_scale))
    print("W scale:", float(w_scale))
    print("V output scale:", float(k_output_scale))
    print("Gemmini C scale:", float(c_scale))

    print("\nPyTorch comparison")
    print("Maximum absolute error:", max_absolute_error)
    print("Mean absolute error:", mean_absolute_error)
    print("RMSE:", rmse)
    print("Cosine similarity:", cosine_similarity)

    print("\nExpected INT8 output")
    print("Minimum:", int(v_expected_int8.min()))
    print("Maximum:", int(v_expected_int8.max()))
    print("Saturated at -128:", saturated_min)
    print("Saturated at 127:", saturated_max)
    print("Checksum:", checksum)


if __name__ == "__main__":
    main()