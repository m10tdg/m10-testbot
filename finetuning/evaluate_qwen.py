# evaluate_qwen.py
import json
from pathlib import Path
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_DIR = Path("/project/project_465003167/m10-testbot/finetuning")
BASE_MODEL = "Qwen/Qwen2.5-14B-Instruct"
ADAPTER_DIR = PROJECT_DIR / "qwen-finetuned-final"
VAL_FILE = PROJECT_DIR / "dataset" / "validation_data.jsonl"
OUTPUT_FILE = PROJECT_DIR / "training_results" / "evaluation_finetuned.json"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)

# Load base model directly to CUDA (matches finetune_qwen.py pattern for LUMI/ROCm)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to("cuda")

model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()

dataset = load_dataset("json", data_files=str(VAL_FILE))["train"]
predictions = []

system_message = {
    "role": "system",
    "content": (
        "You are a Playwright test automation expert. "
        "Given a test instruction, URL, and DOM structure, "
        "generate a concise executable Playwright Python script. "
        "Output ONLY Python code using page.locator(), page.fill(), page.click(), "
        "page.wait_for_url(), and similar Playwright methods. "
        "Use the exact selector IDs from the DOM structure. "
        "Include try/except error handling. "
        "No explanations, no markdown, only Python code."
    ),
}

for i, example in enumerate(dataset):
    user_content = example["instruction"]
    if example.get("input"):
        user_content += f"\n\n{example['input']}"

    messages = [system_message, {"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only generated tokens
    gen_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    pred_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    predictions.append(
        {
            "index": i,
            "instruction": example.get("instruction", ""),
            "input": example.get("input", ""),
            "reference": example.get("output", ""),
            "prediction": pred_text,
        }
    )

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump({"predictions": predictions}, f, indent=2)

print(f"Predictions saved to {OUTPUT_FILE}")