"""
evaluate_metrics.py
--------------------
Computes all evaluation metrics for a given generation of synthetic data.

Metrics:
  - TTR                 : Type-Token Ratio (vocabulary diversity per response)
  - Repetition Rate     : fraction of repeated n-grams within a response
  - BLEU-4              : n-gram overlap with Gen-0 baseline
  - ROUGE-L             : longest common subsequence with Gen-0 baseline
  - Cosine Similarity   : intra-generation semantic similarity (homogenization signal)
  - Perplexity          : fluency under a frozen GPT-2 reference model
  - Novel N-gram Ratio  : fraction of bigrams not seen in Gen-0 baseline
  - BERTScore F1        : contextual semantic similarity with Gen-0 baseline

Usage:
    # Evaluate gen-1, GPT-4o lineage
    python evaluate_metrics.py --teacher gpt4o --generation 1

    # Evaluate all generations and save summary CSV
    python evaluate_metrics.py --teacher gpt4o --all_generations --max_gen 5

    # Print JSON to stdout (used internally by distillation_loop.py)
    python evaluate_metrics.py --teacher gpt4o --generation 1 --output_json
"""

import json
import math
import argparse
import warnings
from pathlib import Path
from collections import Counter

import numpy as np

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[1]
SYNTHETIC_DIR = ROOT / "data" / "synthetic"
RESULTS_DIR   = ROOT / "outputs" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Lazy imports — only load heavy libraries when needed
# ─────────────────────────────────────────────────────────────────────────────

def _import_nltk():
    import nltk
    for resource in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)
    return nltk


def _import_rouge():
    from rouge_score import rouge_scorer
    return rouge_scorer


def _import_bertscore():
    from bert_score import score as bert_score
    return bert_score


def _import_sentence_transformers():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer


def _import_transformers_perplexity():
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    return torch, GPT2LMHeadModel, GPT2TokenizerFast


# ─────────────────────────────────────────────────────────────────────────────
# Tokenization utility
# ─────────────────────────────────────────────────────────────────────────────

def simple_tokenize(text: str) -> list:
    """Word-level tokenize — lowercase, split on whitespace/punctuation."""
    import re
    return re.findall(r"\b\w+\b", text.lower())


# ─────────────────────────────────────────────────────────────────────────────
# Individual metric functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_bleu(hypotheses: list, references: list) -> float:
    """
    Corpus-level BLEU-4.
    hypotheses, references: lists of strings (one per example).
    """
    nltk = _import_nltk()
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    tokenized_hyps = [simple_tokenize(h) for h in hypotheses]
    tokenized_refs = [[simple_tokenize(r)] for r in references]

    smoother = SmoothingFunction().method1
    score    = corpus_bleu(tokenized_refs, tokenized_hyps, smoothing_function=smoother)
    return round(float(score), 4)


def compute_rouge_l(hypotheses: list, references: list) -> float:
    """
    Mean ROUGE-L F1 across all examples.
    """
    rouge_scorer = _import_rouge()
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for hyp, ref in zip(hypotheses, references):
        result = scorer.score(ref, hyp)
        scores.append(result["rougeL"].fmeasure)
    return round(float(np.mean(scores)), 4)


def compute_bertscore(hypotheses: list, references: list, lang: str = "en") -> float:
    """
    Mean BERTScore F1 (uses DeBERTa-xlarge-mnli by default via bert_score library).
    Falls back gracefully if GPU unavailable.
    """
    bert_score = _import_bertscore()
    P, R, F1 = bert_score(hypotheses, references, lang=lang, verbose=False)
    return round(float(F1.mean().item()), 4)


def compute_ttr(texts: list) -> float:
    """
    Mean Type-Token Ratio across all texts.
    TTR = unique_tokens / total_tokens (per response, then averaged).
    Higher TTR = more diverse vocabulary.
    """
    ratios = []
    for text in texts:
        tokens = simple_tokenize(text)
        if len(tokens) == 0:
            continue
        ttr = len(set(tokens)) / len(tokens)
        ratios.append(ttr)
    return round(float(np.mean(ratios)), 4) if ratios else 0.0


def compute_cosine_similarity(texts: list, model_name: str = "all-MiniLM-L6-v2") -> float:
    """
    Mean pairwise cosine similarity of sentence embeddings within the generation.
    High value = homogenization (outputs are converging).
    Uses all-MiniLM-L6-v2 (fast, ~80MB) from sentence-transformers.

    Note: Computing all pairs is O(n²). For n=200 this is fine.
    """
    SentenceTransformer = _import_sentence_transformers()
    model       = SentenceTransformer(model_name)
    embeddings  = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    n = len(embeddings)
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            sim    = float(np.dot(embeddings[i], embeddings[j]))  # cosine (already normalized)
            total += sim
            count += 1

    return round(total / count, 4) if count > 0 else 0.0


def compute_perplexity(texts: list, max_length: int = 128) -> float:
    """
    Mean perplexity of each text under a frozen GPT-2 base model.
    Lower = more fluent/predictable. Rising perplexity across generations = collapse.
    """
    torch, GPT2LMHeadModel, GPT2TokenizerFast = _import_transformers_perplexity()

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model     = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    perplexities = []
    for text in texts:
        encoding = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            loss = model(**encoding, labels=encoding["input_ids"]).loss
        perplexities.append(math.exp(loss.item()))

    return round(float(np.mean(perplexities)), 2)


def compute_repetition_rate(texts: list, n: int = 3) -> float:
    """
    Mean fraction of repeated n-grams within each response.
    repetition_rate = 1 - (unique_ngrams / total_ngrams)
    High value = responses are repetitive (a collapse signal).
    """
    rates = []
    for text in texts:
        tokens = simple_tokenize(text)
        if len(tokens) < n:
            rates.append(0.0)
            continue
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        if len(ngrams) == 0:
            rates.append(0.0)
            continue
        unique_fraction = len(set(ngrams)) / len(ngrams)
        rates.append(1.0 - unique_fraction)
    return round(float(np.mean(rates)), 4)


def compute_novel_ngram_ratio(texts: list, baseline_texts: list, n: int = 2) -> float:
    """
    Fraction of n-grams in `texts` that do NOT appear in `baseline_texts`.
    Higher = model is generating more novel content (good early on; monitors divergence).
    Falling toward 0 = model is staying close to baseline.
    Shooting to 1 = model has drifted away from coherent language.
    """
    baseline_ngrams = set()
    for text in baseline_texts:
        tokens = simple_tokenize(text)
        for i in range(len(tokens) - n + 1):
            baseline_ngrams.add(tuple(tokens[i:i+n]))

    novel_count, total_count = 0, 0
    for text in texts:
        tokens = simple_tokenize(text)
        for i in range(len(tokens) - n + 1):
            ng = tuple(tokens[i:i+n])
            total_count += 1
            if ng not in baseline_ngrams:
                novel_count += 1

    return round(novel_count / total_count, 4) if total_count > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Load data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_responses(teacher: str, generation: int) -> list:
    """Load synthetic responses for a given teacher/generation."""
    path = SYNTHETIC_DIR / f"gen{generation}_{teacher}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No data at {path}. "
            f"Run generate_teacher_outputs.py or distillation_loop.py first."
        )
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    teacher: str,
    generation: int,
    skip_bertscore: bool = False,
    skip_cosine: bool = False,
    skip_perplexity: bool = False,
) -> dict:
    """
    Compute all metrics for a given teacher/generation pair.
    Returns a dict of metric name → value.
    """
    print(f"\nEvaluating: teacher={teacher}, generation={generation}")

    # Load this generation's responses
    records   = load_responses(teacher, generation)
    responses = [r["response"] for r in records]

    # Load gen-0 baseline (always the teacher's original outputs for BLEU/ROUGE/novel-ngram)
    if generation == 0:
        baseline_records = records
    else:
        baseline_records = load_responses(teacher, 0)
    baselines = [r["response"] for r in baseline_records]

    # Align lengths (in case gen-0 has different count due to errors)
    min_len   = min(len(responses), len(baselines))
    responses = responses[:min_len]
    baselines = baselines[:min_len]

    print(f"  Comparing {min_len} responses against gen-0 baseline...")

    metrics = {
        "teacher":    teacher,
        "generation": generation,
        "n_samples":  min_len,
    }

    # BLEU-4
    print("  Computing BLEU-4...")
    metrics["bleu4"] = compute_bleu(responses, baselines)

    # ROUGE-L
    print("  Computing ROUGE-L...")
    metrics["rouge_l"] = compute_rouge_l(responses, baselines)

    # TTR
    print("  Computing TTR...")
    metrics["ttr"] = compute_ttr(responses)

    # Repetition rate
    print("  Computing repetition rate...")
    metrics["repetition_rate"] = compute_repetition_rate(responses)

    # Novel n-gram ratio
    print("  Computing novel bigram ratio...")
    metrics["novel_bigram_ratio"] = compute_novel_ngram_ratio(responses, baselines)

    # BERTScore (can be slow on CPU)
    if not skip_bertscore:
        print("  Computing BERTScore (may take a minute)...")
        try:
            metrics["bertscore_f1"] = compute_bertscore(responses, baselines)
        except Exception as e:
            print(f"    ⚠️  BERTScore failed: {e}")
            metrics["bertscore_f1"] = None
    else:
        metrics["bertscore_f1"] = None

    # Cosine similarity (intra-generation)
    if not skip_cosine:
        print("  Computing cosine similarity (intra-generation)...")
        try:
            metrics["cosine_similarity"] = compute_cosine_similarity(responses)
        except Exception as e:
            print(f"    ⚠️  Cosine similarity failed: {e}")
            metrics["cosine_similarity"] = None
    else:
        metrics["cosine_similarity"] = None

    # Perplexity (reference GPT-2)
    if not skip_perplexity:
        print("  Computing perplexity (reference GPT-2)...")
        try:
            metrics["perplexity"] = compute_perplexity(responses)
        except Exception as e:
            print(f"    ⚠️  Perplexity failed: {e}")
            metrics["perplexity"] = None
    else:
        metrics["perplexity"] = None

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Multi-generation sweep
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all_generations(
    teacher: str,
    max_gen: int,
    skip_bertscore: bool = False,
    skip_cosine: bool = False,
    skip_perplexity: bool = False,
) -> list:
    """
    Evaluate all generations from 0 to max_gen.
    Returns list of metric dicts. Saves CSV summary to outputs/results/.
    """
    all_metrics = []

    for gen in range(max_gen + 1):
        try:
            m = evaluate(
                teacher=teacher,
                generation=gen,
                skip_bertscore=skip_bertscore,
                skip_cosine=skip_cosine,
                skip_perplexity=skip_perplexity,
            )
            all_metrics.append(m)
        except FileNotFoundError as e:
            print(f"  Skipping gen {gen}: {e}")

    if all_metrics:
        # Save JSON
        out_path = RESULTS_DIR / f"metrics_{teacher}_gen0to{max_gen}.json"
        with open(out_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\n✅ Metrics saved → {out_path}")

        # Save CSV
        _save_csv(all_metrics, RESULTS_DIR / f"metrics_{teacher}_gen0to{max_gen}.csv")

    return all_metrics


def _save_csv(metrics_list: list, path: Path):
    """Save metrics list to a CSV file."""
    import csv
    if not metrics_list:
        return
    fieldnames = list(metrics_list[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_list)
    print(f"  CSV saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute evaluation metrics for distillation generations.")
    parser.add_argument("--teacher",         choices=["gpt4o", "claude", "both", "combined"],
                        default="gpt4o",     help="Teacher lineage label")
    parser.add_argument("--generation",      type=int, default=0,
                        help="Specific generation to evaluate")
    parser.add_argument("--all_generations", action="store_true",
                        help="Evaluate all generations from 0 to --max_gen")
    parser.add_argument("--max_gen",         type=int, default=5,
                        help="Maximum generation to include when using --all_generations")
    parser.add_argument("--output_json",     action="store_true",
                        help="Print metrics as a JSON line to stdout (for distillation_loop.py)")
    parser.add_argument("--skip_bertscore",  action="store_true",
                        help="Skip BERTScore (slow on CPU)")
    parser.add_argument("--skip_cosine",     action="store_true",
                        help="Skip cosine similarity (requires sentence-transformers)")
    parser.add_argument("--skip_perplexity", action="store_true",
                        help="Skip perplexity computation")
    args = parser.parse_args()

    if args.all_generations:
        evaluate_all_generations(
            teacher=args.teacher,
            max_gen=args.max_gen,
            skip_bertscore=args.skip_bertscore,
            skip_cosine=args.skip_cosine,
            skip_perplexity=args.skip_perplexity,
        )
    else:
        metrics = evaluate(
            teacher=args.teacher,
            generation=args.generation,
            skip_bertscore=args.skip_bertscore,
            skip_cosine=args.skip_cosine,
            skip_perplexity=args.skip_perplexity,
        )

        # Print summary
        print(f"\n{'─'*40}")
        print(f"Teacher: {metrics['teacher']} | Generation: {metrics['generation']}")
        print(f"{'─'*40}")
        for k, v in metrics.items():
            if k not in ("teacher", "generation", "n_samples"):
                print(f"  {k:<25} {v}")

        if args.output_json:
            # Single JSON line to stdout for programmatic use
            print(json.dumps(metrics))

        # Save to file
        out_path = RESULTS_DIR / f"metrics_{args.teacher}_gen{args.generation}.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✅ Metrics saved → {out_path}")


if __name__ == "__main__":
    main()
