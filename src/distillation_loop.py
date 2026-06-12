"""
distillation_loop.py
---------------------
Orchestrates the full multi-generation distillation loop:

    Gen 0:  Teacher (GPT-4o or Claude) generates responses to original prompts
            -> Student trained on teacher outputs
    Gen 1:  Trained student generates new responses
            -> New student trained on gen-1 outputs
    Gen N:  Repeat until max_generations

The loop evaluates metrics at each generation and saves a full run manifest.

Primary setup used in the paper:
    OPT-1.3B student, GPT-4o and Claude as separate teacher lineages, 5 generations.

Usage:
    # Paper setup: OPT-1.3B, GPT-4o lineage, 5 generations
    python distillation_loop.py --model opt --teacher gpt4o --generations 5

    # Paper setup: OPT-1.3B, Claude lineage, 5 generations
    python distillation_loop.py --model opt --teacher claude --generations 5

    # Lighter alternative: GPT-2 Medium + LoRA
    python distillation_loop.py --model gpt2 --teacher gpt4o --generations 5
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[1]
SYNTHETIC_DIR = ROOT / "data" / "synthetic"
CKPT_DIR      = ROOT / "outputs" / "checkpoints"
RESULTS_DIR   = ROOT / "outputs" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SRC_DIR          = Path(__file__).resolve().parent
GENERATE_SCRIPT  = SRC_DIR / "generate_teacher_outputs.py"
TRAIN_SCRIPT     = SRC_DIR / "train_student.py"
METRICS_SCRIPT   = SRC_DIR / "evaluate_metrics.py"

# ─────────────────────────────────────────────────────────────────────────────
# Subprocess helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_subprocess(cmd: list, desc: str):
    """Run a subprocess and stream output. Raises on non-zero exit."""
    print(f"\n{'─'*60}")
    print(f"▶ {desc}")
    print(f"  Command: {' '.join(str(c) for c in cmd)}")
    print(f"{'─'*60}")

    result = subprocess.run(cmd, check=True)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Student generation: use the trained student to generate outputs
# ─────────────────────────────────────────────────────────────────────────────

def generate_student_outputs(
    ckpt_path: str,
    generation: int,
    teacher_label: str,
    model_name: str,
    prompts_path: Path,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    batch_size: int = 8,
):
    """
    Use the trained student model to generate responses to the original prompts.
    Saves output to data/synthetic/gen{generation}_{teacher_label}_student.json
    so the next generation's training script can pick it up.

    Note: teacher_label here refers to the teacher lineage (e.g. 'gpt4o'),
    not the model that generated this particular generation's outputs.
    We track the full provenance in the metadata.
    """
    print(f"\nGenerating student outputs for generation {generation}...")

    # Load prompts
    with open(prompts_path) as f:
        prompts = json.load(f)

    # Load model + tokenizer
    print(f"  Loading checkpoint: {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_config = _get_base_config(model_name)
    if base_config["use_qlora"]:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_config["hf_id"],
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_config["hf_id"],
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )

    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, ckpt_path)
    model.eval()

    results = []
    errors  = []

    for i, item in enumerate(prompts):
        prompt_text = (
            f"[PROMPT] {item['prompt'].strip()} [RESPONSE]"
        )
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=base_config["max_length"],
        ).to(device if not base_config["use_qlora"] else "cuda")

        print(f"  [{i+1}/{len(prompts)}] {item['id']} ... ", end="", flush=True)

        try:
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.convert_tokens_to_ids("[END]"),
                )

            # Decode only the newly generated tokens
            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            response   = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            # Remove trailing [END] if present
            response   = response.replace("[END]", "").strip()

            results.append({
                "prompt_id":   item["id"],
                "category":    item["category"],
                "prompt":      item["prompt"],
                "response":    response,
                "teacher":     f"student_gen{generation}",
                "teacher_lineage": teacher_label,
                "model_id":    base_config["hf_id"],
                "ckpt_path":   ckpt_path,
                "generation":  generation,
                "timestamp":   datetime.utcnow().isoformat(),
            })
            print("✓")
        except Exception as e:
            print(f"✗ {e}")
            errors.append({"prompt_id": item["id"], "error": str(e)})

    # Save — next generation's train_student.py will load this as "gen{N}_{teacher}"
    out_path = SYNTHETIC_DIR / f"gen{generation}_{teacher_label}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Saved {len(results)} student responses → {out_path}")
    if errors:
        err_path = SYNTHETIC_DIR / f"gen{generation}_{teacher_label}_errors.json"
        with open(err_path, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"  ⚠️  {len(errors)} errors → {err_path}")

    return results


def _get_base_config(model_name: str) -> dict:
    """Return the base model config dict (mirrors MODEL_CONFIGS in train_student.py)."""
    configs = {
        "opt": {
            "hf_id":      "facebook/opt-1.3b",
            "use_qlora":  False,
            "max_length": 512,
        },
        "gpt2": {
            "hf_id":      "gpt2-medium",
            "use_qlora":  False,
            "max_length": 256,
        },
        "mistral": {
            "hf_id":      "mistralai/Mistral-7B-v0.1",
            "use_qlora":  True,
            "max_length": 512,
        },
    }
    return configs[model_name]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate a generation (calls evaluate_metrics.py)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_generation(teacher: str, generation: int) -> dict:
    """
    Call evaluate_metrics.py for this generation and return the metrics dict.
    Returns empty dict if metrics script fails (non-blocking).
    """
    try:
        result = subprocess.run(
            [
                sys.executable, str(METRICS_SCRIPT),
                "--teacher",    teacher,
                "--generation", str(generation),
                "--output_json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        # evaluate_metrics.py prints a JSON line to stdout when --output_json is set
        for line in result.stdout.splitlines():
            if line.startswith("{"):
                return json.loads(line)
    except Exception as e:
        print(f"  ⚠️  Metrics evaluation failed (non-fatal): {e}")
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def distillation_loop(
    model_name: str,
    teacher: str,
    max_generations: int,
    run_id: str = None,
    skip_initial_generation: bool = False,
):
    """
    Full multi-generation distillation loop.

    Generation 0:
        - Teacher (GPT-4o / Claude) generates responses (via generate_teacher_outputs.py)
        - Student trained on those responses (via train_student.py)
        - Metrics computed on gen-0 data

    Generation N (N >= 1):
        - Trained student from gen N-1 generates new responses
        - New student trained on those responses
        - Metrics computed
        - Loop continues until max_generations
    """
    if run_id is None:
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    prompts_path = ROOT / "data" / "prompts" / "prompts.json"
    manifest = {
        "run_id":            run_id,
        "model_name":        model_name,
        "teacher":           teacher,
        "max_generations":   max_generations,
        "started_at":        datetime.utcnow().isoformat(),
        "generations":       [],
    }

    last_ckpt = None

    print(f"\n{'='*60}")
    print(f"DISTILLATION LOOP")
    print(f"Student model  : {model_name}")
    print(f"Teacher        : {teacher}")
    print(f"Max generations: {max_generations}")
    print(f"Run ID         : {run_id}")
    print(f"{'='*60}\n")

    for gen in range(max_generations + 1):
        print(f"\n{'#'*60}")
        print(f"# GENERATION {gen}")
        print(f"{'#'*60}")

        gen_entry = {"generation": gen, "steps": []}

        # ── Step 1: Generate responses ──────────────────────────────────────────
        if gen == 0:
            if not skip_initial_generation:
                # Teacher model generates gen-0 data
                teachers = ["gpt4o", "claude"] if teacher == "both" else [teacher]
                for t in teachers:
                    data_path = SYNTHETIC_DIR / f"gen0_{t}.json"
                    if data_path.exists():
                        print(f"  Gen-0 data already exists for {t}, skipping generation.")
                    else:
                        run_subprocess(
                            [
                                sys.executable, str(GENERATE_SCRIPT),
                                "--teacher",    t,
                                "--generation", "0",
                            ],
                            desc=f"Generate gen-0 data (teacher: {t})",
                        )
                gen_entry["steps"].append("teacher_generation")
            else:
                print("  Skipping initial teacher generation (--skip_initial_generation set).")
        else:
            # Student from previous generation generates outputs
            assert last_ckpt is not None, "No checkpoint from previous generation."
            generate_student_outputs(
                ckpt_path=last_ckpt,
                generation=gen,
                teacher_label=teacher if teacher != "both" else "combined",
                model_name=model_name,
                prompts_path=prompts_path,
            )
            gen_entry["steps"].append("student_generation")

        # ── Step 2: Evaluate current generation's data ──────────────────────────
        print(f"\nEvaluating generation {gen} outputs...")
        metrics = evaluate_generation(
            teacher=teacher if teacher != "both" else "combined",
            generation=gen,
        )
        gen_entry["metrics"] = metrics
        if metrics:
            print(f"  Metrics: {json.dumps(metrics, indent=4)}")

        # ── Step 3: Train student on current generation ─────────────────────────
        train_teacher_arg = teacher if teacher != "both" or gen > 0 else "both"
        if gen > 0:
            # For gen > 0, the "teacher" file is the student-generated file
            # (already saved as gen{N}_{teacher_label}.json by generate_student_outputs)
            train_teacher_arg = teacher if teacher != "both" else "combined"

        run_subprocess(
            [
                sys.executable, str(TRAIN_SCRIPT),
                "--model",      model_name,
                "--teacher",    train_teacher_arg,
                "--generation", str(gen),
                "--run_id",     f"{run_id}_gen{gen}",
            ],
            desc=f"Train student (gen {gen})",
        )
        gen_entry["steps"].append("student_training")

        # Find the checkpoint just written
        ckpt_name  = f"{model_name}_gen{gen}_{train_teacher_arg}_{run_id}_gen{gen}"
        last_ckpt  = str(CKPT_DIR / ckpt_name / "final")
        gen_entry["checkpoint"] = last_ckpt

        manifest["generations"].append(gen_entry)

        # ── Save manifest after each generation ─────────────────────────────────
        manifest_path = RESULTS_DIR / f"run_{run_id}_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\n  📋 Manifest updated → {manifest_path}")

    manifest["completed_at"] = datetime.utcnow().isoformat()
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n\n{'='*60}")
    print(f"✅ DISTILLATION LOOP COMPLETE")
    print(f"   Generations: {max_generations}")
    print(f"   Manifest   : {manifest_path}")
    print(f"{'='*60}\n")

    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run multi-generation distillation loop.")
    parser.add_argument(
        "--model",
        choices=["opt", "gpt2", "mistral"],
        default="opt",
        help="Student model. 'opt' = OPT-1.3B (paper default). 'gpt2' = GPT-2 Medium + LoRA. 'mistral' = Mistral 7B + QLoRA.",
    )
    parser.add_argument(
        "--teacher",
        choices=["gpt4o", "claude", "both"],
        default="gpt4o",
        help="Teacher model(s) to use for generation 0",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=5,
        help="Number of distillation generations to run (default: 5)",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Optional run identifier for checkpoint naming",
    )
    parser.add_argument(
        "--skip_initial_generation",
        action="store_true",
        help="Skip gen-0 teacher generation if data already exists",
    )
    args = parser.parse_args()

    distillation_loop(
        model_name=args.model,
        teacher=args.teacher,
        max_generations=args.generations,
        run_id=args.run_id,
        skip_initial_generation=args.skip_initial_generation,
    )


if __name__ == "__main__":
    main()
