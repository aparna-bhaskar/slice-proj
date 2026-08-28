import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-4-E2B-it"
PROMPT = "The cat sat on the"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

print("=" * 70)
print("GEMMA 4 E2B SMOKE TEST")
print("=" * 70)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Loading model weights...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
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
    "GPU memory allocated:",
    round(torch.cuda.memory_allocated() / 2**30, 2),
    "GiB",
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

inputs = {
    name: tensor.to(device)
    for name, tensor in inputs.items()
}

token_ids = inputs["input_ids"][0].tolist()
tokens = tokenizer.convert_ids_to_tokens(token_ids)

print("\nOriginal prompt:", repr(PROMPT))
print("Formatted prompt:", repr(formatted_prompt))
print("Input IDs shape:", tuple(inputs["input_ids"].shape))

print("\nPosition | Token ID | Token")
print("-" * 55)

for position, (token_id, token) in enumerate(zip(token_ids, tokens)):
    print(f"{position:8d} | {token_id:8d} | {token!r}")

print("\nRunning direct forward pass...")

with torch.inference_mode():
    outputs = model(
        **inputs,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )

print("Logits shape:", tuple(outputs.logits.shape))
print("Hidden-state count:", len(outputs.hidden_states))
print("First hidden-state shape:", tuple(outputs.hidden_states[0].shape))
print("Last hidden-state shape:", tuple(outputs.hidden_states[-1].shape))
print("Cache type:", type(outputs.past_key_values).__name__)

if hasattr(outputs.past_key_values, "get_seq_length"):
    print(
        "Cache sequence length:",
        outputs.past_key_values.get_seq_length(),
    )

last_logits = outputs.logits[:, -1, :].float()
probabilities = torch.softmax(last_logits, dim=-1)

top_probabilities, top_ids = torch.topk(
    probabilities,
    k=5,
    dim=-1,
)

print("\nTop-five predicted tokens:")

for rank, (token_id, probability) in enumerate(
    zip(top_ids[0].tolist(), top_probabilities[0].tolist()),
    start=1,
):
    print(
        f"{rank}. "
        f"id={token_id:7d} "
        f"token={tokenizer.decode([token_id])!r} "
        f"probability={probability:.8f}"
    )

print("\nRunning generate() for one token...")

with torch.inference_mode():
    generated = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
    )

input_length = inputs["input_ids"].shape[1]
new_token_ids = generated[0, input_length:]

print("Generated token IDs:", new_token_ids.tolist())
print("Generated text:", repr(tokenizer.decode(new_token_ids)))
print(
    "Peak GPU memory:",
    round(torch.cuda.max_memory_allocated() / 2**30, 2),
    "GiB",
)

print("\nSMOKE TEST PASSED")
