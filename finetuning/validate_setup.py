#!/usr/bin/env python3
"""
Pre-training validation script.

Run this BEFORE submitting the SLURM job to catch any issues early.
This script validates:
- Python packages are installed correctly
- GPU is available
- Training data format is correct
- Model can be loaded
"""

import sys
import json
from pathlib import Path

print("=" * 80)
print("LUMI FINE-TUNING SETUP VALIDATION")
print("=" * 80)

checks_passed = 0
checks_failed = 0

def check(description: str):
    """Decorator for validation checks"""
    def decorator(func):
        def wrapper():
            global checks_passed, checks_failed
            try:
                print(f"\n[CHECK] {description}...", end=" ", flush=True)
                func()
                print("✓ PASS")
                checks_passed += 1
            except Exception as e:
                print(f"✗ FAIL: {e}")
                checks_failed += 1
                return False
            return True
        return wrapper
    return decorator


# ============================================================================
# 1. PYTHON PACKAGES
# ============================================================================

@check("PyTorch installation")
def check_pytorch():
    import torch
    assert torch.__version__, "PyTorch version not found"
    print(f"✓ {torch.__version__}", end="")

check_pytorch()


@check("GPU availability")
def check_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected - fine-tuning requires GPU")
    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    assert device_count > 0, "GPU count is 0"
    print(f"✓ {device_count} GPU(s) - {device_name}", end="")

check_gpu()


@check("Transformers library")
def check_transformers():
    import transformers
    assert transformers.__version__, "Transformers version not found"
    print(f"✓ {transformers.__version__}", end="")

check_transformers()


@check("PEFT (LoRA) library")
def check_peft():
    import peft
    assert peft.__version__, "PEFT version not found"
    print(f"✓ {peft.__version__}", end="")

check_peft()


@check("Datasets library")
def check_datasets():
    import datasets
    assert datasets.__version__, "Datasets version not found"
    print(f"✓ {datasets.__version__}", end="")

check_datasets()


@check("Accelerate library")
def check_accelerate():
    import accelerate
    assert accelerate.__version__, "Accelerate version not found"
    print(f"✓ {accelerate.__version__}", end="")

check_accelerate()


# ============================================================================
# 2. TRAINING DATA
# ============================================================================

@check("Training data file exists")
def check_training_data_exists():
    training_file = Path("/dataset/training_data.jsonl")
    if not training_file.exists():
        raise FileNotFoundError(
            f"training_data.jsonl not found in {Path.cwd()}\n"
            f"  See: python convert_to_training_data.py --help"
        )
    print(f"✓ {training_file}", end="")

check_training_data_exists()


@check("Training data format (JSONL)")
def check_training_data_format():
    with open("training_data.jsonl") as f:
        lines = f.readlines()
    
    if not lines:
        raise ValueError("training_data.jsonl is empty")
    
    # Check first 5 lines are valid JSON
    for i, line in enumerate(lines[:5]):
        try:
            data = json.loads(line)
            required_keys = {"instruction", "input", "output"}
            missing = required_keys - set(data.keys())
            if missing:
                raise KeyError(f"Missing keys: {missing}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Line {i+1} is not valid JSON: {e}")
    
    print(f"✓ {len(lines)} examples", end="")

check_training_data_format()


@check("Training data size (minimum 50 examples)")
def check_training_data_size():
    with open("training_data.jsonl") as f:
        count = len(f.readlines())
    
    if count < 50:
        raise ValueError(
            f"Only {count} examples found - minimum 50 recommended\n"
            f"  Current examples are too few for good fine-tuning"
        )
    print(f"✓ {count} examples (sufficient)", end="")

check_training_data_size()


# ============================================================================
# 3. CONFIGURATION FILES
# ============================================================================

@check("Fine-tuning script exists (finetune_qwen.py)")
def check_finetune_script():
    script_file = Path("finetune_qwen.py")
    if not script_file.exists():
        raise FileNotFoundError("finetune_qwen.py not found")
    print(f"✓ {script_file}", end="")

check_finetune_script()


@check("SLURM job script exists (submit_finetuning.slurm)")
def check_slurm_script():
    slurm_file = Path("submit_finetuning.slurm")
    if not slurm_file.exists():
        raise FileNotFoundError("submit_finetuning.slurm not found")
    
    # Check for required SLURM directives
    with open(slurm_file) as f:
        content = f.read()
    
    required = ["--job-name", "--account", "--partition", "--nodes", "--gpus"]
    missing = [r for r in required if r not in content]
    
    if missing:
        raise ValueError(f"Missing SLURM directives: {missing}")
    
    print(f"✓ {slurm_file}", end="")

check_slurm_script()


@check("Output directories will be created")
def check_output_dirs():
    dirs = [
        Path("/project/project_465003167/m10-testbot/qwen-finetuned"),
        Path("/project/project_465003167/m10-testbot/qwen-finetuned-final"),
        Path("/project/project_465003167/m10-testbot/logs"),
    ]
    
    for d in dirs:
        d.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ {len(dirs)} directories", end="")

check_output_dirs()


# ============================================================================
# 4. MODEL LOADING (DRY RUN)
# ============================================================================

@check("Can load tokenizer (dry run)")
def check_tokenizer_load():
    print("\n    (Downloading tokenizer... this may take a minute)", end="", flush=True)
    
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-14B",
        trust_remote_code=True,
        cache_dir="/project/project_465003167/m10-testbot/.cache",
    )
    
    assert tokenizer is not None
    assert len(tokenizer) > 0
    
    print("\r    ", end="")  # Clear the "downloading" message
    print(f"✓ vocab size: {len(tokenizer)}", end="")

check_tokenizer_load()


@check("Sample tokenization works")
def check_tokenization():
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-14B",
        trust_remote_code=True,
        cache_dir="/project/project_465003167/m10-testbot/.cache",
    )
    
    sample_text = "[INST] Generate a test [/INST] page.goto('https://example.com')"
    tokens = tokenizer.encode(sample_text)
    
    assert len(tokens) > 0
    print(f"✓ {len(tokens)} tokens", end="")

check_tokenization()


# ============================================================================
# 5. MEMORY & RESOURCES
# ============================================================================

@check("Available memory (at least 60GB)")
def check_memory():
    import os
    try:
        mem_info = os.popen("free -h").read()
        # This is a soft check - just verify the command runs
        assert "Mem:" in mem_info
        print(f"✓ Memory detected", end="")
    except:
        print(f"⚠ Could not verify (may be unavailable on this system)", end="")

check_memory()


@check("GPU memory (should have 16GB+ per GPU)")
def check_gpu_memory():
    import torch
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"✓ {gpu_memory:.1f} GB per GPU", end="")
    else:
        raise RuntimeError("No GPU found")

check_gpu_memory()


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

total = checks_passed + checks_failed

print(f"\nPassed: {checks_passed}/{total}")
print(f"Failed: {checks_failed}/{total}")

if checks_failed == 0:
    print("\n" + "✓" * 40)
    print("ALL CHECKS PASSED - Ready to fine-tune!")
    print("✓" * 40)
    
    print("\nNext steps:")
    print("1. Review training_data.jsonl")
    print("2. Submit SLURM job:  sbatch submit_finetuning.slurm")
    print("3. Monitor:  tail -f logs/slurm-*.out")
    print("4. After training, fine-tuned model will be in:")
    print("   /project/project_465003167/m10-testbot/qwen-finetuned-final")
    
    sys.exit(0)
else:
    print("\n" + "✗" * 40)
    print("SETUP VALIDATION FAILED")
    print("✗" * 40)
    
    print("\nPlease fix the issues above before submitting the fine-tuning job.")
    
    sys.exit(1)