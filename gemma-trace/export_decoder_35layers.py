from pathlib import Path
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM

MODEL_ID = "google/gemma-4-E2B-it"
OUTPUT_DIR = Path.home() / "gemma4-trace" / "decoder_35layers_weights"


def save_tensor(directory, name, tensor):
    array = tensor.detach().to(
        device="cpu", dtype=torch.float32
    ).contiguous().numpy()

    path = directory / f"{name}.bin"
    array.tofile(path)

    print(
        f"  {name:36s} "
        f"shape={str(array.shape):20s} "
        f"checksum={float(array.sum()):.6f}"
    )


def export_layer(layer, index, output_root):
    directory = output_root / f"layer_{index:02d}"
    directory.mkdir(parents=True, exist_ok=True)

    tensors = {
        "input_norm_weight": layer.input_layernorm.weight,
        "post_attention_norm_weight": layer.post_attention_layernorm.weight,
        "pre_ffn_norm_weight": layer.pre_feedforward_layernorm.weight,
        "post_ffn_norm_weight": layer.post_feedforward_layernorm.weight,
        "post_per_layer_input_norm_weight":
            layer.post_per_layer_input_norm.weight,
        "q_proj_weight": layer.self_attn.q_proj.weight,
        "o_proj_weight": layer.self_attn.o_proj.weight,
        "q_norm_weight": layer.self_attn.q_norm.weight,
        "gate_proj_weight": layer.mlp.gate_proj.weight,
        "up_proj_weight": layer.mlp.up_proj.weight,
        "down_proj_weight": layer.mlp.down_proj.weight,
        "per_layer_gate_weight": layer.per_layer_input_gate.weight,
        "per_layer_projection_weight": layer.per_layer_projection.weight,
        "layer_scalar": layer.layer_scalar,
    }

    attn = layer.self_attn

    if not attn.is_kv_shared_layer:
        tensors["k_proj_weight"] = attn.k_proj.weight
        tensors["k_norm_weight"] = attn.k_norm.weight

        if attn.v_proj is not None:
            tensors["v_proj_weight"] = attn.v_proj.weight

    print(f"\n===== Exporting layer {index:02d} =====")

    for name, tensor in tensors.items():
        save_tensor(directory, name, tensor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--end",
        type=int,
        default=35,
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading Gemma-4 model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        local_files_only=True,
    )

    model.eval()

    text_model = model.model.language_model
    layers = text_model.layers

    print("Decoder layers:", len(layers))

    if len(layers) != 35:
        raise RuntimeError(
            f"Expected 35 layers, got {len(layers)}"
        )

    if not (0 <= args.start < args.end <= len(layers)):
        raise ValueError("Invalid layer range")

    with torch.no_grad():
        for index in range(args.start, args.end):
            export_layer(layers[index], index, args.output)

    print("\nExport complete.")
    print("Output:", args.output)


if __name__ == "__main__":
    main()
