from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "google/gemma-4-E2B-it"
PROMPT = "The cat sat on the"

EXPORT_DIR = Path.home() / "gemma4-trace" / "decoder_cpu_export"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

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

    text_model = model.model.language_model
    layer0 = text_model.layers[0]

    # Layer-0 RMSNorm weights
    norm_weights = {
        "input_norm_weight":
            layer0.input_layernorm.weight.detach().float().cpu().numpy(),

        "post_attention_norm_weight":
            layer0.post_attention_layernorm.weight.detach().float().cpu().numpy(),

        "pre_ffn_norm_weight":
            layer0.pre_feedforward_layernorm.weight.detach().float().cpu().numpy(),

        "post_ffn_norm_weight":
            layer0.post_feedforward_layernorm.weight.detach().float().cpu().numpy(),

        "post_per_layer_input_norm_weight":
            layer0.post_per_layer_input_norm.weight.detach().float().cpu().numpy(),
    }

    for name, array in norm_weights.items():
        np.save(EXPORT_DIR / f"{name}.npy", array)
        print(name, array.shape, "checksum:", float(array.sum()))

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

    captured = {}

    # Capture raw Q/K/V projection outputs for C validation
    def q_proj_hook(module, module_inputs, module_output):
        captured["q_proj_output"] = module_output.detach().float().cpu()

    def k_proj_hook(module, module_inputs, module_output):
        captured["k_proj_output"] = module_output.detach().float().cpu()

    def v_proj_hook(module, module_inputs, module_output):
        captured["v_proj_output"] = module_output.detach().float().cpu()

    def o_proj_hook(module, module_inputs, module_output):
        captured["o_proj_output"] = module_output.detach().float().cpu()

    q_handle = layer0.self_attn.q_proj.register_forward_hook(q_proj_hook)
    k_handle = layer0.self_attn.k_proj.register_forward_hook(k_proj_hook)
    v_handle = layer0.self_attn.v_proj.register_forward_hook(v_proj_hook)
    o_handle = layer0.self_attn.o_proj.register_forward_hook(o_proj_hook)

    # Capture Q/K/V RMSNorm outputs
    def q_norm_hook(module, module_inputs, module_output):
        captured["q_norm_output"] = module_output.detach().float().cpu()

    def k_norm_hook(module, module_inputs, module_output):
        captured["k_norm_output"] = module_output.detach().float().cpu()

    def v_norm_hook(module, module_inputs, module_output):
        captured["v_norm_output"] = module_output.detach().float().cpu()

    q_norm_handle = layer0.self_attn.q_norm.register_forward_hook(q_norm_hook)
    k_norm_handle = layer0.self_attn.k_norm.register_forward_hook(k_norm_hook)
    v_norm_handle = layer0.self_attn.v_norm.register_forward_hook(v_norm_hook)

    # Capture output of the first RMSNorm for C validation
    def input_norm_hook(module, module_inputs, module_output):
        captured["input_norm_output"] = (
            module_output.detach().float().cpu()
        )

    input_norm_handle = layer0.input_layernorm.register_forward_hook(
        input_norm_hook
    )

    # Capture post-attention RMSNorm output.
    def post_attention_norm_hook(module, module_inputs, module_output):
        captured["post_attention_norm_output"] = (
            module_output.detach().float().cpu()
        )

    post_attention_norm_handle = (
        layer0.post_attention_layernorm.register_forward_hook(
            post_attention_norm_hook
        )
    )

    # Capture the true PyTorch result after the first residual add.
    # This is the input to pre_feedforward_layernorm.
    def pre_ffn_norm_pre_hook(module, args):
        captured["first_residual_output_actual"] = (
            args[0].detach().float().cpu()
        )

    pre_ffn_norm_pre_handle = (
        layer0.pre_feedforward_layernorm.register_forward_pre_hook(
            pre_ffn_norm_pre_hook
        )
    )

    # Capture MLP / feed-forward boundaries.
    def pre_ffn_norm_hook(module, module_inputs, module_output):
        captured["pre_ffn_norm_output"] = (
            module_output.detach().float().cpu()
        )

    def gate_proj_hook(module, module_inputs, module_output):
        captured["gate_proj_output"] = (
            module_output.detach().float().cpu()
        )

    def up_proj_hook(module, module_inputs, module_output):
        captured["up_proj_output"] = (
            module_output.detach().float().cpu()
        )

    def down_proj_pre_hook(module, args):
        captured["gate_up_output"] = (
            args[0].detach().float().cpu()
        )

    def down_proj_hook(module, module_inputs, module_output):
        captured["down_proj_output"] = (
            module_output.detach().float().cpu()
        )

    def post_ffn_norm_hook(module, module_inputs, module_output):
        captured["post_ffn_norm_output"] = (
            module_output.detach().float().cpu()
        )

    pre_ffn_norm_handle = (
        layer0.pre_feedforward_layernorm.register_forward_hook(
            pre_ffn_norm_hook
        )
    )

    gate_proj_handle = layer0.mlp.gate_proj.register_forward_hook(
        gate_proj_hook
    )

    up_proj_handle = layer0.mlp.up_proj.register_forward_hook(
        up_proj_hook
    )

    down_proj_pre_handle = (
        layer0.mlp.down_proj.register_forward_pre_hook(
            down_proj_pre_hook
        )
    )

    down_proj_handle = layer0.mlp.down_proj.register_forward_hook(
        down_proj_hook
    )

    post_ffn_norm_handle = (
        layer0.post_feedforward_layernorm.register_forward_hook(
            post_ffn_norm_hook
        )
    )

    # Capture actual PyTorch output after the MLP residual add.
    # This is the input to per_layer_input_gate.
    def per_layer_gate_pre_hook(module, args):
        captured["second_residual_output"] = (
            args[0].detach().float().cpu()
        )

    per_layer_gate_pre_handle = (
        layer0.per_layer_input_gate.register_forward_pre_hook(
            per_layer_gate_pre_hook
        )
    )

    # ---- Gemma 4 per-layer-input tail ----
    def per_layer_gate_hook(module, module_inputs, module_output):
        captured["per_layer_gate_output"] = (
            module_output.detach().float().cpu()
        )

    def per_layer_projection_pre_hook(module, args):
        # Exact GELU(gate) * per_layer_input tensor.
        captured["per_layer_product"] = (
            args[0].detach().float().cpu()
        )

    def per_layer_projection_hook(module, module_inputs, module_output):
        captured["per_layer_projection_output"] = (
            module_output.detach().float().cpu()
        )

    def post_per_layer_norm_hook(module, module_inputs, module_output):
        captured["post_per_layer_norm_output"] = (
            module_output.detach().float().cpu()
        )

    per_layer_gate_handle = (
        layer0.per_layer_input_gate.register_forward_hook(
            per_layer_gate_hook
        )
    )

    per_layer_projection_pre_handle = (
        layer0.per_layer_projection.register_forward_pre_hook(
            per_layer_projection_pre_hook
        )
    )

    per_layer_projection_handle = (
        layer0.per_layer_projection.register_forward_hook(
            per_layer_projection_hook
        )
    )

    post_per_layer_norm_handle = (
        layer0.post_per_layer_input_norm.register_forward_hook(
            post_per_layer_norm_hook
        )
    )

    def layer0_pre_hook(module, args, kwargs):
        captured["hidden_states"] = (
            args[0].detach().float().cpu()
        )

        per_layer_input = args[1]
        captured["per_layer_input"] = (
            per_layer_input.detach().float().cpu()
        )

        captured["position_ids"] = (
            kwargs["position_ids"].detach().cpu()
        )

        captured["attention_mask"] = (
            kwargs["attention_mask"].detach().float().cpu()
        )

        cos, sin = kwargs["position_embeddings"]

        captured["rope_cos"] = (
            cos.detach().float().cpu()
        )

        captured["rope_sin"] = (
            sin.detach().float().cpu()
        )

    def layer0_hook(module, args, kwargs, output):
        captured["layer_output"] = (
            output.detach().float().cpu()
        )

    pre_handle = layer0.register_forward_pre_hook(
        layer0_pre_hook,
        with_kwargs=True,
    )

    post_handle = layer0.register_forward_hook(
        layer0_hook,
        with_kwargs=True,
    )

    print("Running one prefill forward pass...")

    try:
        with torch.inference_mode():
            model(
                **inputs,
                use_cache=False,
                return_dict=True,
            )
    finally:
        pre_handle.remove()
        post_handle.remove()
        input_norm_handle.remove()
        q_handle.remove()
        k_handle.remove()
        v_handle.remove()
        q_norm_handle.remove()
        k_norm_handle.remove()
        v_norm_handle.remove()

    print("\nCaptured tensors:")

    for name, tensor in captured.items():
        print(
            f"{name:20s}",
            tuple(tensor.shape),
            tensor.dtype,
        )

    hidden_states = captured["hidden_states"][0].numpy()
    per_layer_input = captured["per_layer_input"][0].numpy()
    layer_output = captured["layer_output"][0].numpy()
    input_norm_output = captured["input_norm_output"][0].numpy()
    position_ids = captured["position_ids"][0].numpy()
    attention_mask = captured["attention_mask"]

    assert hidden_states.shape == (14, 1536)
    assert per_layer_input.shape == (14, 256)
    assert layer_output.shape == (14, 1536)
    assert position_ids.shape == (14,)

    np.save(EXPORT_DIR / "hidden_states.npy", hidden_states)
    np.save(EXPORT_DIR / "per_layer_input.npy", per_layer_input)
    np.save(EXPORT_DIR / "layer_output.npy", layer_output)
    np.save(EXPORT_DIR / "input_norm_output.npy", input_norm_output)
    np.save(EXPORT_DIR / "position_ids.npy", position_ids)
    np.save(EXPORT_DIR / "attention_mask.npy", attention_mask.numpy())

    # Export Q/K/V projection weights.
    q_weight = layer0.self_attn.q_proj.weight.detach().float().cpu().numpy()
    k_weight = layer0.self_attn.k_proj.weight.detach().float().cpu().numpy()
    v_weight = layer0.self_attn.v_proj.weight.detach().float().cpu().numpy()
    o_weight = layer0.self_attn.o_proj.weight.detach().float().cpu().numpy()

    q_output = captured["q_proj_output"][0].numpy()
    k_output = captured["k_proj_output"][0].numpy()
    v_output = captured["v_proj_output"][0].numpy()
    o_output = captured["o_proj_output"][0].numpy()

    q_norm_weight = layer0.self_attn.q_norm.weight.detach().float().cpu().numpy()
    k_norm_weight = layer0.self_attn.k_norm.weight.detach().float().cpu().numpy()

    q_norm_output = captured["q_norm_output"][0].numpy()
    k_norm_output = captured["k_norm_output"][0].numpy()
    v_norm_output = captured["v_norm_output"][0].numpy()

    rope_cos = captured["rope_cos"].numpy()
    rope_sin = captured["rope_sin"].numpy()

    post_attention_norm_output = (
        captured["post_attention_norm_output"][0].numpy()
    )

    # First residual connection:
    # hidden_states = residual + post_attention_norm_output
    first_residual_output = (
        captured["first_residual_output_actual"][0].numpy()
    )

    gate_proj_weight = (
        layer0.mlp.gate_proj.weight.detach().float().cpu().numpy()
    )
    up_proj_weight = (
        layer0.mlp.up_proj.weight.detach().float().cpu().numpy()
    )
    down_proj_weight = (
        layer0.mlp.down_proj.weight.detach().float().cpu().numpy()
    )

    pre_ffn_norm_output = captured["pre_ffn_norm_output"][0].numpy()
    gate_proj_output = captured["gate_proj_output"][0].numpy()
    up_proj_output = captured["up_proj_output"][0].numpy()
    gate_up_output = captured["gate_up_output"][0].numpy()
    down_proj_output = captured["down_proj_output"][0].numpy()
    post_ffn_norm_output = captured["post_ffn_norm_output"][0].numpy()
    second_residual_output = (
        captured["second_residual_output"][0].numpy()
    )

    per_layer_gate_weight = (
        layer0.per_layer_input_gate.weight.detach().float().cpu().numpy()
    )

    per_layer_projection_weight = (
        layer0.per_layer_projection.weight.detach().float().cpu().numpy()
    )

    per_layer_gate_output = (
        captured["per_layer_gate_output"][0].numpy()
    )

    per_layer_product = (
        captured["per_layer_product"][0].numpy()
    )

    per_layer_projection_output = (
        captured["per_layer_projection_output"][0].numpy()
    )

    post_per_layer_norm_output = (
        captured["post_per_layer_norm_output"][0].numpy()
    )

    layer_scalar = (
        layer0.layer_scalar.detach().float().cpu().numpy()
    )

    arrays = {
        "pre_ffn_norm_output": pre_ffn_norm_output,
        "gate_proj_weight": gate_proj_weight,
        "up_proj_weight": up_proj_weight,
        "down_proj_weight": down_proj_weight,
        "gate_proj_output": gate_proj_output,
        "up_proj_output": up_proj_output,
        "gate_up_output": gate_up_output,
        "down_proj_output": down_proj_output,
        "post_ffn_norm_output": post_ffn_norm_output,
        "second_residual_output": second_residual_output,
        "per_layer_gate_weight": per_layer_gate_weight,
        "per_layer_projection_weight": per_layer_projection_weight,
        "per_layer_gate_output": per_layer_gate_output,
        "per_layer_product": per_layer_product,
        "per_layer_projection_output": per_layer_projection_output,
        "post_per_layer_norm_output": post_per_layer_norm_output,
        "layer_scalar": layer_scalar,
        "post_attention_norm_output": post_attention_norm_output,
        "first_residual_output": first_residual_output,
        "rope_cos": rope_cos,
        "rope_sin": rope_sin,
        "q_norm_weight": q_norm_weight,
        "k_norm_weight": k_norm_weight,
        "q_norm_output": q_norm_output,
        "k_norm_output": k_norm_output,
        "v_norm_output": v_norm_output,
        "q_proj_weight": q_weight,
        "k_proj_weight": k_weight,
        "v_proj_weight": v_weight,
        "o_proj_weight": o_weight,
        "q_proj_output": q_output,
        "k_proj_output": k_output,
        "v_proj_output": v_output,
        "o_proj_output": o_output,
    }

    for name, array in arrays.items():
        array = np.asarray(array, dtype=np.float32)
        np.save(EXPORT_DIR / f"{name}.npy", array)
        array.tofile(EXPORT_DIR / f"{name}.bin")
        print(name, array.shape, "checksum:", float(array.sum()))

    print("\nSaved boundary tensors to:")
    print(EXPORT_DIR)

    print("\nLayer input checksum:",
          float(hidden_states.sum()))
    print("Per-layer input checksum:",
          float(per_layer_input.sum()))
    print("Layer output checksum:",
          float(layer_output.sum()))


if __name__ == "__main__":
    main()
