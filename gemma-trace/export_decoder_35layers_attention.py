from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-4-E2B-it"
PROMPT = "The cat sat on the"

ROOT = Path("/scratch/aparna/gemma4_35layers")
REFERENCE = Path.home() / "gemma4-trace/decoder_cpu_export"


def main():
    output_dir = ROOT / "attention"
    reference_dir = ROOT / "reference"

    output_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, local_files_only=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        local_files_only=True,
    )
    model.eval()

    messages = [{"role": "user", "content": PROMPT}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(formatted, return_tensors="pt")
    device = model.get_input_embeddings().weight.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    layers = model.model.language_model.layers
    assert len(layers) == 35
    assert inputs["input_ids"].shape[1] == 14

    captured = {}
    handles = []

    def save_array(path, tensor):
        array = (
            tensor.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
            .numpy()
        )
        array.tofile(path)
        return array

    def make_pre_hook(i):
        def hook(module, args, kwargs):
            cos, sin = kwargs["position_embeddings"]
            mask = kwargs["attention_mask"]

            captured[i] = {
                "cos": cos.detach().float().cpu(),
                "sin": sin.detach().float().cpu(),
                "mask": (
                    mask.detach().float().cpu()
                    if mask is not None else None
                ),
            }
        return hook

    def make_post_hook(i):
        def hook(module, args, kwargs, output):
            array = save_array(
                reference_dir / f"layer_output_{i:02d}.bin",
                output[0],
            )
            print(
                f"Layer {i:02d} output: "
                f"shape={array.shape}, "
                f"checksum={float(array.sum()):.9g}"
            )
        return hook

    for i, layer in enumerate(layers):
        handles.append(
            layer.register_forward_pre_hook(
                make_pre_hook(i), with_kwargs=True
            )
        )
        handles.append(
            layer.register_forward_hook(
                make_post_hook(i), with_kwargs=True
            )
        )

    try:
        with torch.inference_mode():
            model(**inputs, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()

    for i in range(35):
        data = captured[i]
        cos = data["cos"]
        sin = data["sin"]
        mask = data["mask"]

        print(
            f"Layer {i:02d}: "
            f"cos={tuple(cos.shape)}, "
            f"sin={tuple(sin.shape)}, "
            f"mask={None if mask is None else tuple(mask.shape)}"
        )

        save_array(output_dir / f"rope_cos_{i:02d}.bin", cos)
        save_array(output_dir / f"rope_sin_{i:02d}.bin", sin)

        if mask is not None:
            save_array(
                output_dir / f"attention_mask_{i:02d}.bin",
                mask,
            )

    original = np.fromfile(
        REFERENCE / "layer_output.bin", dtype=np.float32
    )
    new = np.fromfile(
        reference_dir / "layer_output_00.bin", dtype=np.float32
    )

    if not np.array_equal(original, new):
        print(
            "WARNING: layer-0 output differs. "
            "Max error:",
            np.max(np.abs(original - new)),
        )
    else:
        print("Layer-0 output matches original reference exactly.")

    print("Export complete:", ROOT)


if __name__ == "__main__":
    main()
