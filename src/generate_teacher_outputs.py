"""
generate_teacher_outputs.py
----------------------------
Queries the teacher models (GPT-4o and/or Claude) on the prompt dataset
and saves their responses as Gen-0 training data for the student.

Run this once before starting the distillation loop.

Usage:
    python generate_teacher_outputs.py --teacher gpt4o --generation 0
    python generate_teacher_outputs.py --teacher claude --generation 0
    python generate_teacher_outputs.py --teacher both --generation 0

Requires:
    OPENAI_API_KEY    environment variable (for GPT-4o)
    ANTHROPIC_API_KEY environment variable (for Claude)
"""

import os
import json
import argparse
import time
from datetime import datetime
from pathlib import Path

# ── API clients (install: pip install openai anthropic) ──────────────────────
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai not installed. Run: pip install openai")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic not installed. Run: pip install anthropic")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
PROMPTS_PATH  = ROOT / "data" / "prompts" / "prompts.json"
SYNTHETIC_DIR = ROOT / "data" / "synthetic"
SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

# ── Model configs ─────────────────────────────────────────────────────────────
TEACHER_CONFIGS = {
    "gpt4o": {
        "model_id":    "gpt-4o",
        "provider":    "openai",
        "max_tokens":  512,
        "temperature": 0.7,
    },
    "claude": {
        "model_id":    "claude-opus-4-6",
        "provider":    "anthropic",
        "max_tokens":  512,
        "temperature": 0.7,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Generation functions
# ─────────────────────────────────────────────────────────────────────────────

def generate_openai(client, prompt: str, config: dict) -> str:
    """Call GPT-4o and return the response text."""
    response = client.chat.completions.create(
        model=config["model_id"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=config["max_tokens"],
        temperature=config["temperature"],
    )
    return response.choices[0].message.content.strip()


def generate_anthropic(client, prompt: str, config: dict) -> str:
    """Call Claude and return the response text."""
    message = client.messages.create(
        model=config["model_id"],
        max_tokens=config["max_tokens"],
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def generate_response(client, prompt: str, config: dict) -> str:
    """Dispatch to the correct provider."""
    if config["provider"] == "openai":
        return generate_openai(client, prompt, config)
    elif config["provider"] == "anthropic":
        return generate_anthropic(client, prompt, config)
    else:
        raise ValueError(f"Unknown provider: {config['provider']}")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_client(teacher_name: str):
    """Build the API client for the chosen teacher."""
    config = TEACHER_CONFIGS[teacher_name]
    if config["provider"] == "openai":
        assert OPENAI_AVAILABLE, "openai package not installed."
        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key, "Set OPENAI_API_KEY environment variable."
        return OpenAI(api_key=api_key)
    elif config["provider"] == "anthropic":
        assert ANTHROPIC_AVAILABLE, "anthropic package not installed."
        api_key = os.getenv("ANTHROPIC_API_KEY")
        assert api_key, "Set ANTHROPIC_API_KEY environment variable."
        return anthropic.Anthropic(api_key=api_key)


def run_generation(teacher_name: str, generation: int,
                   limit: int = None, delay: float = 0.5,
                   retry_errors: bool = False):
    """
    Generate responses for all prompts using the specified teacher.

    Args:
        teacher_name  : 'gpt4o' or 'claude'
        generation    : generation number (0 = human prompts, 1+ = synthetic)
        limit         : only process first N prompts (useful for testing)
        delay         : seconds to wait between API calls (rate limiting)
        retry_errors  : if True, only retry prompts listed in the errors file
                        and MERGE results into the existing output file
    """
    config = TEACHER_CONFIGS[teacher_name]
    client = build_client(teacher_name)

    out_path = SYNTHETIC_DIR / f"gen{generation}_{teacher_name}.json"
    err_path = SYNTHETIC_DIR / f"gen{generation}_{teacher_name}_errors.json"

    if retry_errors:
        # Load only the failed prompt IDs
        if not err_path.exists():
            print(f"No errors file found at {err_path}. Nothing to retry.")
            return []
        with open(err_path) as f:
            error_records = json.load(f)
        failed_ids = {r["prompt_id"] for r in error_records}

        # Load full prompt list and filter to only failed ones
        with open(PROMPTS_PATH) as f:
            all_prompts = json.load(f)
        prompts = [p for p in all_prompts if p["id"] in failed_ids]

        # Load existing successful results to merge into
        existing_results = []
        if out_path.exists():
            with open(out_path) as f:
                existing_results = json.load(f)

        print(f"\nRetrying {len(prompts)} failed prompts (will merge into existing {len(existing_results)} results)...")
    else:
        # Normal run — load all prompts
        with open(PROMPTS_PATH) as f:
            prompts = json.load(f)
        existing_results = []

    if limit:
        prompts = prompts[:limit]

    print(f"\n{'='*60}")
    print(f"Teacher: {teacher_name} ({config['model_id']})")
    print(f"Generation: {generation}")
    print(f"Prompts: {len(prompts)}")
    print(f"{'='*60}\n")

    results = list(existing_results)  # start with what we already have
    errors  = []

    for i, item in enumerate(prompts):
        prompt_id = item["id"]
        category  = item["category"]
        prompt    = item["prompt"]

        print(f"[{i+1}/{len(prompts)}] {prompt_id} ({category}) ... ", end="", flush=True)

        try:
            response = generate_response(client, prompt, config)
            results.append({
                "prompt_id":   prompt_id,
                "category":    category,
                "prompt":      prompt,
                "response":    response,
                "teacher":     teacher_name,
                "model_id":    config["model_id"],
                "generation":  generation,
                "timestamp":   datetime.utcnow().isoformat(),
            })
            print("✓")
        except Exception as e:
            print(f"✗ ERROR: {e}")
            errors.append({"prompt_id": prompt_id, "error": str(e)})

        time.sleep(delay)

    # Save results
    out_filename = f"gen{generation}_{teacher_name}.json"
    out_path = SYNTHETIC_DIR / out_filename
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved {len(results)} responses → {out_path}")
    if errors:
        err_path = SYNTHETIC_DIR / f"gen{generation}_{teacher_name}_errors.json"
        with open(err_path, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"⚠️  {len(errors)} errors saved → {err_path}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset from teacher models.")
    parser.add_argument("--teacher",    choices=["gpt4o", "claude", "both"], default="both",
                        help="Which teacher model to use")
    parser.add_argument("--generation", type=int, default=0,
                        help="Generation number (0 = from original prompts)")
    parser.add_argument("--limit",      type=int, default=None,
                        help="Limit number of prompts (for testing)")
    parser.add_argument("--delay",        type=float, default=0.5,
                        help="Seconds between API calls")
    parser.add_argument("--retry_errors", action="store_true",
                        help="Only retry failed prompts and merge into existing output file")
    args = parser.parse_args()

    teachers = ["gpt4o", "claude"] if args.teacher == "both" else [args.teacher]

    for teacher in teachers:
        run_generation(
            teacher_name=teacher,
            generation=args.generation,
            limit=args.limit,
            delay=args.delay,
            retry_errors=args.retry_errors,
        )


if __name__ == "__main__":
    main()
