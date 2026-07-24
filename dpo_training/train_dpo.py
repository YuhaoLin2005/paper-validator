"""DPO training on causal preference dataset with QLoRA.

Gate 2: Train Qwen2.5-3B to prefer causal-reasoning responses over direct-action.
Hardware: RTX3060 Laptop 6GB VRAM — 3B model + 4-bit QLoRA fits comfortably.

Usage:
    python dpo_training/train_dpo.py                # full train
    python dpo_training/train_dpo.py --dry-run      # verify setup without training
    python dpo_training/train_dpo.py --val-split 0.1 # 10% val split
"""

import json, os, sys, argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = MODEL_DIR / "causal-dpo-qwen1.5b"

# ── QLoRA config ──────────────────────────────────
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

LORA_CONFIG = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
)

TRAINING_ARGS = DPOConfig(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=5e-5,
    warmup_steps=100,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    bf16=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    remove_unused_columns=False,
    report_to="none",
    dataloader_num_workers=0,
    beta=0.1,
    max_length=1024,
)


def load_dataset(split="train"):
    path = DATA_DIR / ("causal_pairs_train.jsonl" if split == "train" else "causal_pairs_test.jsonl")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return Dataset.from_list(rows)


def format_dpo(example):
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Verify setup, no training")
    ap.add_argument("--val-split", type=float, default=0.0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    out_dir = Path(args.output) if args.output else OUTPUT_DIR

    print("=" * 60)
    print("Gate 2: DPO Training — Causal Preference Optimization")
    print("=" * 60)

    # Hardware check
    print(f"PyTorch: {torch.__version__}  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {torch.cuda.get_device_name(0)}  VRAM: {gb:.1f} GB")
        if gb < 5.5:
            print("WARNING: <5.5GB VRAM — consider reducing max_length.")
    else:
        print("WARNING: No CUDA GPU. Training requires GPU.")
        if not args.dry_run:
            sys.exit(1)

    # Load data
    ds = load_dataset("train").map(format_dpo)
    print(f"Train pairs: {len(ds)}")

    if args.val_split > 0:
        split = ds.train_test_split(test_size=args.val_split, seed=42)
        train_ds, val_ds = split["train"], split["test"]
        print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")
    else:
        train_ds, val_ds = ds, None

    sample = train_ds[0]
    print(f"Sample prompt: {sample['prompt'][:80]}...")
    print(f"Sample chosen ({len(sample['chosen'])}c): {sample['chosen'][:120]}...")
    print(f"Sample rejected ({len(sample['rejected'])}c): {sample['rejected'][:80]}...")

    if args.dry_run:
        print("\n[Dry run] Setup verified. No training.")
        return

    # Load model
    print(f"\nLoading {MODEL_NAME} (4-bit QLoRA)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    # Train
    print("\nStarting DPO training...")
    targs = TRAINING_ARGS
    if args.max_steps:
        targs.max_steps = args.max_steps
    targs.output_dir = str(out_dir)

    trainer = DPOTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    trainer.train()

    # Save
    print(f"\nSaving adapter to {out_dir}...")
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print("Done. Gate 2 complete.")


if __name__ == "__main__":
    main()
