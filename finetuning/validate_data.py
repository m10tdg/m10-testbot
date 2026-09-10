import json
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
DATA_FILE = "dataset/all_data.jsonl"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

print("=" * 80)
print("DATASET VALIDATION")
print("=" * 80)

print("Examples:", len(data))

required = {"instruction", "input", "output"}

for i, example in enumerate(data):
    assert required.issubset(example), (
        f"Example {i} missing fields"
    )

    user_text = (
        example["instruction"]
        + "\n\n"
        + example["input"]
    )

    messages = [
        {
            "role": "user",
            "content": user_text
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    full_messages = [
        {
            "role": "user",
            "content": user_text
        },
        {
            "role": "assistant",
            "content": example["output"]
        }
    ]

    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False
    )

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False
    )["input_ids"]

    full_ids = tokenizer(
        full_text,
        add_special_tokens=False
    )["input_ids"]

    response_tokens = len(full_ids) - len(prompt_ids)

    if response_tokens <= 0:
        raise RuntimeError(
            f"Example {i} has no response tokens"
        )

    if len(full_ids) > 2048:
        print(
            f"WARNING example {i}: "
            f"{len(full_ids)} tokens"
        )

print("\n✓ Dataset structure valid")
print("✓ All examples contain response tokens")