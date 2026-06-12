"""
train_student.py
-----------------
Fine-tunes a student language model on synthetic teacher outputs.
Primary model used in the paper: OPT-1.3B (facebook/opt-1.3b), full fine-tuning
in bfloat16 with AdamW (lr=1e-5), 3 epochs per generation.

Usage:
    # OPT-1.3B, GPT-4o teacher lineage, generation 0
    python train_student.py --model opt --teacher gpt4o --generation 0

    # OPT-1.3B, Claude teacher lineage, generation 2
    python train_student.py --model opt --teacher claude --generation 2

    # GPT-2 Medium + LoRA (lighter alternative, not used in paper)
    python train_student.py --model gpt2 --teacher gpt4o --generation 0
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)

# Paths
ROOT          = Path(__file__).resolve().parents[1]
SYNTHETIC_DIR = ROOT / "data" / "synthetic"
CKPT_DIR      = ROOT / "outputs" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Model configs
# "opt" matches the actual experiment in the paper.
# "gpt2" and "mistral" are provided as lighter/heavier alternatives.
MODEL_CONFIGS = {
    "opt": {
        "hf_id":         "facebook/opt-1.3b",
        "use_lora":      False,    # full fine-tuning as used in paper
        "max_length":    512,
        "batch_size":    4,
        "grad_accum":    4,
        "lr":            1e-5,     # AdamW lr from paper
        "epochs":        3,
        "dtype":         torch.bfloat16,
    },
    "gpt2": {
        "hf_id":         "gpt2-medium",
        "use_lora":      True,
        "lora_r":        8,
        "lora_alpha":    16,
        "lora_dropout":  0.05,
        "target_modules": ["c_attn", "c_proj"],
        "max_length":    256,
        "batch_size":    8,
        "grad_accum":    2,
        "lr":            2e-4,
        "epochs":        3,
        "dtype":         torch.float16,
    },
    "mistral": {
        "hf_id":         "mistralai/Mistral-7B-v0.1",
        "use_lora":      True,
        "use_qlora":     True,     # 4-bit quantization
        "lora_r":        8,
        "lora_alpha":    16,
        "lora_dropout":  0.05,
        "target_modules": ["q_proj", "v_proj"],
        "max_length":    512,
        "batch_size":    4,
        "grad_accum":    4,
        "lr":            2e-4,
        "epochs":        3,
        "dtype":         torch.bfloat16,
    },
}


class DistillationDataset(Dataset):
    """
    Wraps teacher or student-generated outputs for causal language modeling.
    Each example is formatted as:
        "[PROMPT] {prompt} [RESPONSE] {response} [END]"
    """

    def __init__(self, records: list, tokenizer, max_length: int):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.examples   = []

        for record in records:
            text = (
                f"[PROMPT] {record['prompt'].strip()} "
                f"[RESPONSE] {record['response'].strip()} [END]"
            )
            self.examples.append(text)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.examples[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        # Labels: same as input_ids, with padding masked out
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


def load_synthetic_data(teacher: str, generation: int) -> list:
    """Load records for the given teacher lineage and generation number."""
    records = []

    if teacher == "both":
        teachers = ["gpt4o", "claude"]
    else:
        teachers = [teacher]

    for t in teachers:
        path = SYNTHETIC_DIR / f"gen{generation}_{t}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Data not found: {path}\n"
                f"Run generate_teacher_outputs.py first for teacher={t}, generation={generation}"
            )
        with open(path) as f:
            data = json.load(f)
        records.extend(data)
        print(f"  Loaded {len(data)} records from {path.name}")

    print(f"  Total: {len(records)} training examples")
    return records


def load_tokenizer(config: dict):
    tokenizer = AutoTokenizer.from_pretrained(config["hf_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    special_tokens = ["[PROMPT]", "[RESPONSE]", "[END]"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    print(f"  Tokenizer: {config['hf_id']} | vocab size: {len(tokenizer)}")
    return tokenizer


def load_model(model_name: str, config: dict, tokenizer):
    """Load the model. OPT uses full fine-tuning; GPT-2 and Mistral use LoRA/QLoRA."""
    use_lora   = config.get("use_lora",  False)
    use_qlora  = config.get("use_qlora", False)

    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config["hf_id"],
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
        print(f"  Loaded {config['hf_id']} in 4-bit NF4 (QLoRA)")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config["hf_id"],
            torch_dtype=config["dtype"],
            device_map="auto" if torch.cuda.is_available() else None,
        )
        print(f"  Loaded {config['hf_id']} ({'full fine-tuning' if not use_lora else 'LoRA'})")

    model.resize_token_embeddings(len(tokenizer))

    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=config["target_modules"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Trainable parameters: {total_params:,} (all)")

    return model


def train(model_name: str, teacher: str, generation: int, run_id: str = None):
    """
    Full training pipeline for one generation.

    Args:
        model_name  : 'opt' (paper default), 'gpt2', or 'mistral'
        teacher     : 'gpt4o', 'claude', or 'both'
        generation  : which generation's data to train on
        run_id      : optional string appended to checkpoint folder name
    """
    config = MODEL_CONFIGS[model_name]

    if run_id is None:
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    ckpt_name = f"{model_name}_gen{generation}_{teacher}_{run_id}"
    ckpt_path = CKPT_DIR / ckpt_name

    print(f"\n{'='*60}")
    print(f"Student : {model_name} ({config['hf_id']})")
    print(f"Teacher : {teacher} | Generation: {generation}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"{'='*60}\n")

    print("Loading data...")
    records = load_synthetic_data(teacher, generation)

    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer(config)

    print("\nLoading model...")
    model = load_model(model_name, config, tokenizer)

    dataset = DistillationDataset(records, tokenizer, config["max_length"])
    print(f"\nDataset: {len(dataset)} examples")

    use_bf16 = config["dtype"] == torch.bfloat16 and torch.cuda.is_available()
    use_fp16 = config["dtype"] == torch.float16 and torch.cuda.is_available() and not use_bf16

    training_args = TrainingArguments(
        output_dir=str(ckpt_path),
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["grad_accum"],
        learning_rate=config["lr"],
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=use_fp16,
        bf16=use_bf16,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("\nStarting training...\n")
    trainer.train()

    model.save_pretrained(str(ckpt_path / "final"))
    tokenizer.save_pretrained(str(ckpt_path / "final"))

    metadata = {
        "model_name":  model_name,
        "hf_id":       config["hf_id"],
        "teacher":     teacher,
        "generation":  generation,
        "run_id":      run_id,
        "ckpt_path":   str(ckpt_path / "final"),
        "timestamp":   datetime.utcnow().isoformat(),
        "train_size":  len(dataset),
    }
    with open(ckpt_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nTraining complete. Checkpoint: {ckpt_path / 'final'}")
    return str(ckpt_path / "final")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune student model on teacher outputs.")
    parser.add_argument(
        "--model",
        choices=["opt", "gpt2", "mistral"],
        default="opt",
        help="Student model. 'opt' = OPT-1.3B full fine-tuning (paper default). "
             "'gpt2' = GPT-2 Medium + LoRA. 'mistral' = Mistral 7B + QLoRA.",
    )
    parser.add_argument(
        "--teacher",
        choices=["gpt4o", "claude", "both"],
        default="gpt4o",
        help="Which teacher lineage's data to train on",
    )
    parser.add_argument(
        "--generation",
        type=int,
        default=0,
        help="Generation number of data to train on",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Optional identifier appended to checkpoint folder name",
    )
    args = parser.parse_args()

    train(
        model_name=args.model,
        teacher=args.teacher,
        generation=args.generation,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
