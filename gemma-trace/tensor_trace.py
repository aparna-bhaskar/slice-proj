import inspect
import json
import re
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.models.gemma4.modeling_gemma4 as gemma4_source


MODEL_ID = "google/gemma-4-E2B-it"
PROMPT = "The cat sat on the"


def describe(value: Any, depth: int = 0) -> Any:
    """Return tensor metadata without printing large tensor contents."""
    if depth > 4:
        return type(value).__name__

    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }

    if isinstance(value, (tuple, list)):
        return [describe(item, depth + 1) for item in value]

    if isinstance(value, dict):
        return {
            str(key): describe(item, depth + 1)
            for key, item in value.items()
        }

    return type(value).__name__


def module_weight_shapes(module: torch.nn.Module) -> dict[str, list[int]]:
    """Get direct learned parameter shapes for a module."""
    result = {}

    for name, parameter in module.named_parameters(recurse=False):
        result[name] = list(parameter.shape)

    # Some Gemma modules wrap nn.Linear inside .linear.
    linear = getattr(module, "linear", None)

    if linear is not None:
        if getattr(linear, "weight", None) is not None:
            result["linear.weight"] = list(linear.weight.shape)

        if getattr(linear, "bias", None) is not None:
            result["linear.bias"] = list(linear.bias.shape)

    return result


print("=" * 88)
print("GEMMA 4 E2B TENSOR TRACE")
print("=" * 88)
print("Model:", MODEL_ID)
print("Prompt:", repr(PROMPT))
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable.")

print("GPU:", torch.cuda.get_device_name(0))
print("BF16 supported:", torch.cuda.is_bf16_supported())
print(
    "Installed Gemma source:",
    inspect.getsourcefile(gemma4_source),
)


print("\n" + "=" * 88)
print("1. LOAD TOKENIZER AND MODEL")
print("=" * 88)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,

    # Exposes the attention calculations as ordinary PyTorch operations.
    attn_implementation="eager",
)

model.eval()

device = model.get_input_embeddings().weight.device

print("Model class:", type(model).__name__)
print("Embedding device:", device)
print("Model dtype:", model.dtype)
print(
    "Parameter count:",
    f"{sum(parameter.numel() for parameter in model.parameters()):,}",
)
print(
    "GPU allocated:",
    round(torch.cuda.memory_allocated() / 2**30, 3),
    "GiB",
)


print("\n" + "=" * 88)
print("2. TOKENIZATION")
print("=" * 88)

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

inputs = {
    name: tensor.to(device)
    for name, tensor in inputs.items()
}

token_ids = inputs["input_ids"][0].tolist()
tokens = tokenizer.convert_ids_to_tokens(token_ids)

print("Formatted prompt:", repr(formatted_prompt))
print("input_ids shape:", tuple(inputs["input_ids"].shape))
print("attention_mask shape:", tuple(inputs["attention_mask"].shape))

print("\nPosition | Token ID | Token")
print("-" * 62)

for position, (token_id, token) in enumerate(zip(token_ids, tokens)):
    print(f"{position:8d} | {token_id:8d} | {token!r}")


print("\n" + "=" * 88)
print("3. TEXT CONFIGURATION")
print("=" * 88)

if hasattr(model.config, "get_text_config"):
    text_config = model.config.get_text_config()
else:
    text_config = getattr(model.config, "text_config", model.config)

config_fields = [
    "vocab_size",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "global_head_dim",
    "intermediate_size",
    "sliding_window",
    "max_position_embeddings",
    "hidden_size_per_layer_input",
    "vocab_size_per_layer_input",
    "final_logit_softcapping",
]

for field in config_fields:
    print(f"{field}: {getattr(text_config, field, None)}")

layer_types = list(getattr(text_config, "layer_types", []))

print("\nLayer sequence:")

for index, layer_type in enumerate(layer_types):
    print(f"layer {index:2d}: {layer_type}")


print("\n" + "=" * 88)
print("4. FIND DECODER LAYERS")
print("=" * 88)

all_modules = dict(model.named_modules())

# Works with module names such as:
# model.language_model.layers.0.self_attn
# model.layers.0.self_attn
attention_modules = {}

pattern = re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn$")

for name, module in all_modules.items():
    # Only trace the text decoder. Exclude the vision and audio towers.
    if not name.startswith("model.language_model.layers."):
        continue

    match = pattern.search(name)

    if match and type(module).__name__ == "Gemma4TextAttention":
        layer_index = int(match.group(1))
        attention_modules[layer_index] = (name, module)

if not attention_modules:
    print("Could not automatically find decoder attention modules.")
    print("Module names containing 'self_attn':")

    for name in all_modules:
        if "self_attn" in name:
            print(name)

    raise RuntimeError("No text self-attention modules were found.")

print("Found decoder attention layers:", sorted(attention_modules))


# Choose the first layer of every distinct attention type.
selected_layers = []
seen_types = set()

for layer_index in sorted(attention_modules):
    name, module = attention_modules[layer_index]

    layer_type = getattr(module, "layer_type", None)

    if layer_type is None and layer_index < len(layer_types):
        layer_type = layer_types[layer_index]

    layer_type = str(layer_type)

    if layer_type not in seen_types:
        selected_layers.append(layer_index)
        seen_types.add(layer_type)

# Safety fallback.
if not selected_layers:
    selected_layers = [min(attention_modules)]

print("Representative layers:", selected_layers)

for layer_index in selected_layers:
    name, module = attention_modules[layer_index]

    print(
        f"layer {layer_index}: "
        f"name={name}, "
        f"type={getattr(module, 'layer_type', None)}, "
        f"class={type(module).__name__}, "
        f"heads={getattr(module, 'num_heads', None)}, "
        f"KV heads={getattr(module, 'num_key_value_heads', None)}, "
        f"head_dim={getattr(module, 'head_dim', None)}, "
        f"KV groups={getattr(module, 'num_key_value_groups', None)}"
    )


print("\n" + "=" * 88)
print("5. REGISTER TENSOR HOOKS")
print("=" * 88)

records = []
matmul_records = []
hook_handles = []

current_phase = "not_started"
active_attention = None


def make_module_hook(module_name: str):
    def hook(module, args, output):
        record = {
            "phase": current_phase,
            "module": module_name,
            "class": type(module).__name__,
            "weights": module_weight_shapes(module),
            "input": describe(args),
            "output": describe(output),
        }

        records.append(record)

        print("\n[MODULE]")
        print("phase:", current_phase)
        print("name:", module_name)
        print("class:", record["class"])
        print("weights:", record["weights"])
        print("input:", record["input"])
        print("output:", record["output"])

    return hook


def make_attention_pre_hook(module_name: str):
    def hook(module, args):
        global active_attention
        active_attention = module_name

        print("\n" + "*" * 88)
        print("ENTER ATTENTION:", module_name)
        print("PHASE:", current_phase)
        print("*" * 88)

    return hook


def make_attention_post_hook(module_name: str):
    def hook(module, args, output):
        global active_attention

        print("*" * 88)
        print("EXIT ATTENTION:", module_name)
        print("*" * 88)

        active_attention = None

    return hook


# Temporarily wrap torch.matmul so QK^T and attention-probabilities × V
# are recorded while a selected attention module is executing.
original_matmul = torch.matmul


def traced_matmul(left, right, *args, **kwargs):
    output = original_matmul(left, right, *args, **kwargs)

    if (
        active_attention is not None
        and isinstance(left, torch.Tensor)
        and isinstance(right, torch.Tensor)
    ):
        record = {
            "phase": current_phase,
            "attention": active_attention,
            "left": list(left.shape),
            "right": list(right.shape),
            "output": list(output.shape),
            "dtype": str(output.dtype),
        }

        matmul_records.append(record)

        print("\n[ATTENTION MATMUL]")
        print("phase:", current_phase)
        print("attention:", active_attention)
        print("left:", tuple(left.shape))
        print("right:", tuple(right.shape))
        print("output:", tuple(output.shape))
        print("dtype:", output.dtype)

    return output


torch.matmul = traced_matmul


# Token embedding.
embedding_module = model.get_input_embeddings()

hook_handles.append(
    embedding_module.register_forward_hook(
        make_module_hook("token_embedding")
    )
)


# Final vocabulary projection.
output_embedding = model.get_output_embeddings()

if output_embedding is not None:
    hook_handles.append(
        output_embedding.register_forward_hook(
            make_module_hook("lm_head")
        )
    )


# Find and hook useful endpoint/per-layer embedding modules.
for name, module in all_modules.items():
    useful_endpoint = (
        "embed_tokens_per_layer" in name
        or "per_layer_model_projection" in name
        or "per_layer_projection_norm" in name
        or name.endswith("language_model.norm")
        or name.endswith("model.norm")
    )

    if useful_endpoint:
        print("Hooking endpoint:", name)

        hook_handles.append(
            module.register_forward_hook(
                make_module_hook(name)
            )
        )


# Hook representative local/global decoder layers.
projection_suffixes = (
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
    "self_attn",
    "q_proj",
    "k_proj",
    "v_proj",
    "q_norm",
    "k_norm",
    "v_norm",
    "o_proj",
    "mlp",
    "gate_proj",
    "up_proj",
    "down_proj",
    "per_layer_input_gate",
    "per_layer_projection",
    "post_per_layer_input_norm",
)

for layer_index in selected_layers:
    attention_name, attention_module = attention_modules[layer_index]
    layer_prefix = attention_name.removesuffix(".self_attn")

    print("\nHooking representative layer:", layer_prefix)

    for name, module in all_modules.items():
        if not name.startswith(layer_prefix + "."):
            continue

        relative_name = name[len(layer_prefix) + 1 :]

        if relative_name.endswith(projection_suffixes):
            print("  ", name)

            hook_handles.append(
                module.register_forward_hook(
                    make_module_hook(name)
                )
            )

    hook_handles.append(
        attention_module.register_forward_pre_hook(
            make_attention_pre_hook(attention_name)
        )
    )

    hook_handles.append(
        attention_module.register_forward_hook(
            make_attention_post_hook(attention_name)
        )
    )


try:
    print("\n" + "=" * 88)
    print("6. PREFILL FORWARD PASS")
    print("=" * 88)

    current_phase = "prefill"

    with torch.inference_mode():
        prefill = model(
            **inputs,
            use_cache=True,
            output_hidden_states=True,
            output_attentions=True,
            logits_to_keep=1,
            return_dict=True,
        )

    print("\nPREFILL RESULTS")
    print("Logits shape:", tuple(prefill.logits.shape))
    print(
        "Hidden-state count:",
        len(prefill.hidden_states)
        if prefill.hidden_states is not None
        else None,
    )

    if prefill.hidden_states is not None:
        print(
            "Initial hidden-state shape:",
            tuple(prefill.hidden_states[0].shape),
        )
        print(
            "Final hidden-state shape:",
            tuple(prefill.hidden_states[-1].shape),
        )

    print(
        "Attention tensor count:",
        len(prefill.attentions)
        if prefill.attentions is not None
        else None,
    )

    if prefill.attentions is not None:
        for layer_index in selected_layers:
            attention = prefill.attentions[layer_index]

            print(
                f"Layer {layer_index} attention probabilities:",
                None if attention is None else tuple(attention.shape),
            )

    next_token = prefill.logits[:, -1, :].argmax(dim=-1)

    print("First generated token ID:", next_token.item())
    print(
        "First generated token:",
        repr(tokenizer.decode(next_token)),
    )


    print("\n" + "=" * 88)
    print("7. KV CACHE")
    print("=" * 88)

    cache = prefill.past_key_values

    print("Cache class:", type(cache).__name__)

    if hasattr(cache, "get_seq_length"):
        print("Cache sequence length:", cache.get_seq_length())

    # Transformers cache representations vary by version.
    if hasattr(cache, "layers"):
        for layer_index in selected_layers:
            if layer_index >= len(cache.layers):
                continue

            layer_cache = cache.layers[layer_index]

            print(f"\nLayer {layer_index} cache class:", type(layer_cache).__name__)

            for attribute in [
                "keys",
                "values",
                "key_cache",
                "value_cache",
            ]:
                value = getattr(layer_cache, attribute, None)

                if isinstance(value, torch.Tensor):
                    print(attribute, tuple(value.shape))

    elif hasattr(cache, "key_cache"):
        for layer_index in selected_layers:
            if layer_index < len(cache.key_cache):
                print(
                    f"Layer {layer_index} K:",
                    tuple(cache.key_cache[layer_index].shape),
                )
                print(
                    f"Layer {layer_index} V:",
                    tuple(cache.value_cache[layer_index].shape),
                )

    shared_kv = getattr(prefill, "shared_kv_states", None)

    if shared_kv is not None:
        print("\nShared KV states:")

        for key, value in shared_kv.items():
            print(key, describe(value))


    print("\n" + "=" * 88)
    print("8. ONE-TOKEN DECODE PASS")
    print("=" * 88)

    current_phase = "decode"

    decode_input_ids = next_token.unsqueeze(1)

    decode_attention_mask = torch.cat(
        [
            inputs["attention_mask"],
            torch.ones(
                (inputs["attention_mask"].shape[0], 1),
                dtype=inputs["attention_mask"].dtype,
                device=inputs["attention_mask"].device,
            ),
        ],
        dim=1,
    )

    print("Decode input shape:", tuple(decode_input_ids.shape))
    print(
        "Decode attention-mask shape:",
        tuple(decode_attention_mask.shape),
    )

    with torch.inference_mode():
        decode = model(
            input_ids=decode_input_ids,
            attention_mask=decode_attention_mask,
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
            output_attentions=True,
            logits_to_keep=1,
            return_dict=True,
        )

    print("\nDECODE RESULTS")
    print("Decode logits shape:", tuple(decode.logits.shape))

    second_token = decode.logits[:, -1, :].argmax(dim=-1)

    print("Second generated token ID:", second_token.item())
    print(
        "Second generated token:",
        repr(tokenizer.decode(second_token)),
    )

    if hasattr(decode.past_key_values, "get_seq_length"):
        print(
            "Cache sequence length after decode:",
            decode.past_key_values.get_seq_length(),
        )


    print("\n" + "=" * 88)
    print("9. ATTENTION MATRIX MULTIPLICATIONS")
    print("=" * 88)

    for index, record in enumerate(matmul_records, start=1):
        print(
            f"{index}. "
            f"phase={record['phase']} "
            f"attention={record['attention']} "
            f"{tuple(record['left'])} @ "
            f"{tuple(record['right'])} -> "
            f"{tuple(record['output'])}"
        )


    print("\n" + "=" * 88)
    print("10. SAVE MACHINE-READABLE RESULTS")
    print("=" * 88)

    with open("tensor_trace_records.json", "w") as file:
        json.dump(records, file, indent=2)

    with open("attention_matmuls.json", "w") as file:
        json.dump(matmul_records, file, indent=2)

    with open("gemma4_text_config.json", "w") as file:
        json.dump(text_config.to_dict(), file, indent=2)

    with open("module_tree.txt", "w") as file:
        for name, module in all_modules.items():
            file.write(f"{name}\t{type(module).__name__}\n")

    print("Created tensor_trace_records.json")
    print("Created attention_matmuls.json")
    print("Created gemma4_text_config.json")
    print("Created module_tree.txt")

    print(
        "Peak GPU memory:",
        round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "GiB",
    )

    print("\nTENSOR TRACE PASSED")

finally:
    torch.matmul = original_matmul

    for handle in hook_handles:
        handle.remove()
