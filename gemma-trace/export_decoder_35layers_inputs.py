from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-4-E2B-it"
PROMPT = "The cat sat on the"

OUTPUT_DIR = Path("/scratch/aparna/gemma4_35layers/inputs")
REFERENCE_DIR = Path.home() / "gemma4-trace/decoder_cpu_export"

NUM_LAYERS = 35
SEQ_LEN = 14
PER_LAYER_DIM = 256


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        local_files_only=True,
    )

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

    layers = model.model.language_model.layers
    assert len(layers) == NUM_LAYERS

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

    print("Input token shape:", tuple(inputs["input_ids"].shape))

    captured = {}
    handles = []

    def make_hook(index):
        def hook(module, args, kwargs):
            hidden_states = args[0]
            per_layer_input = args[1]

            assert per_layer_input.shape == (
                1, SEQ_LEN, PER_LAYER_DIM
            ), (
                f"Layer {index}: unexpected shape "
                f"{tuple(per_layer_input.shape)}"
            )

            array = (
                per_layer_input[0]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
                .numpy()
                .copy()
            )

            captured[index] = array

            print(
                f"Captured layer {index:02d}: "
                f"shape={array.shape} "
                f"checksum={float(array.sum()):.9g}"
            )

        return hook

    for index, layer in enumerate(layers):
        handle = layer.register_forward_pre_hook(
            make_hook(index),
            with_kwargs=True,
        )
        handles.append(handle)

    print("Running full 35-layer forward pass...")

    try:
        with torch.inference_mode():
            model(
                **inputs,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    assert len(captured) == NUM_LAYERS, (
        f"Captured {len(captured)} layers, expected 35"
    )

    print("\nValidating layer 0...")

    reference = np.fromfile(
        REFERENCE_DIR / "per_layer_input.bin",
        dtype=np.float32,
    ).reshape(SEQ_LEN, PER_LAYER_DIM)

    actual = captured[0]

    max_error = float(np.max(np.abs(actual - reference)))
    mean_error = float(np.mean(np.abs(actual - reference)))

    print("Layer 0 max error:", max_error)
    print("Layer 0 mean error:", mean_error)

    if not np.array_equal(actual, reference):
        raise RuntimeError(
            "Layer-0 input differs from original trace. "
            "Inputs were not saved."
        )

    print("Layer-0 input matches original trace exactly.")

    for index in range(NUM_LAYERS):
        path = OUTPUT_DIR / f"per_layer_input_{index:02d}.bin"
        captured[index].tofile(path)

    print("\nExported", NUM_LAYERS, "layer-specific inputs")
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
