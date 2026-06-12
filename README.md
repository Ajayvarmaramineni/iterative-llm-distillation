# Teaching Themselves to Forget: Convergent Collapse in Iterative LLM Distillation

Independent research by Ajay Ramineni, WPI  
Paper targeting ICLR 2027

---

## What This Is

This repo contains the full experiment code for a study on what happens when a language model is trained repeatedly on its own outputs across multiple generations.

The setup: two independent teacher lineages (GPT-4o and Claude Opus) each generate responses to 178 prompts across four task categories. A student model (OPT-1.3B) is fine-tuned on those outputs. Then the student's own outputs become the training data for the next generation. Repeat for 5 generations.

The finding: both lineages collapse to the same degraded state by Generation 5, regardless of which teacher they started from. Vocabulary diversity collapses, repetition climbs, and text coherence drops. We call this **convergent collapse**.

---

## Repo Structure

```
├── data/
│   └── prompts/
│       └── prompts.json          # 178 prompts across 4 categories
├── src/
│   ├── generate_teacher_outputs.py   # Step 1: get Gen-0 data from GPT-4o / Claude
│   ├── train_student.py              # Step 2: fine-tune OPT-1.3B on teacher outputs
│   ├── distillation_loop.py          # Step 3: run the full multi-generation loop
│   └── evaluate_metrics.py           # Step 4: compute all metrics per generation
└── figures/                          # Paper figures (PNG)
```

---

## Experiment Setup

| Component         | Detail                                      |
|-------------------|---------------------------------------------|
| Student model     | OPT-1.3B (`facebook/opt-1.3b`)              |
| Teacher 1         | GPT-4o (OpenAI API)                         |
| Teacher 2         | Claude Opus (`claude-opus-4-6`)             |
| Fine-tuning       | Full fine-tuning, bfloat16, no GradScaler   |
| Optimizer         | AdamW, lr=1e-5, weight decay=0.01           |
| Training          | 3 epochs per generation                     |
| Generations       | 5 (Gen 0 = teacher outputs, Gen 1-5 = student outputs) |
| Prompts           | 178 across QA, logical reasoning, creative writing, domain knowledge |
| Metrics           | TTR, repetition rate, BLEU-4, ROUGE-L, cosine similarity, perplexity |

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API keys

```bash
export OPENAI_API_KEY=your_key_here
export ANTHROPIC_API_KEY=your_key_here
```

### 3. Generate Gen-0 teacher outputs

```bash
python src/generate_teacher_outputs.py --teacher gpt4o --generation 0
python src/generate_teacher_outputs.py --teacher claude --generation 0
```

### 4. Run the full distillation loop (5 generations)

```bash
# GPT-4o lineage
python src/distillation_loop.py --teacher gpt4o --generations 5

# Claude lineage
python src/distillation_loop.py --teacher claude --generations 5
```

### 5. Evaluate all generations

```bash
python src/evaluate_metrics.py --teacher gpt4o --all_generations --max_gen 5
python src/evaluate_metrics.py --teacher claude --all_generations --max_gen 5
```

---

## Metrics

| Metric             | What it measures                                      | Collapse signal       |
|--------------------|-------------------------------------------------------|----------------------|
| TTR                | Type-Token Ratio — vocabulary diversity per response  | Drops toward 0       |
| Repetition Rate    | Fraction of repeated n-grams within a response        | Rises toward 1       |
| BLEU-4             | N-gram overlap with Gen-0 baseline                    | Drops                |
| ROUGE-L            | Longest common subsequence with Gen-0 baseline        | Drops                |
| Cosine Similarity  | Semantic similarity across responses (intra-gen)      | Rises (homogenization) |
| Perplexity         | Fluency under frozen GPT-2 reference model            | Rises                |

---

## Prompt Categories

| Category              | Count |
|-----------------------|-------|
| Question answering    | ~45   |
| Logical reasoning     | ~45   |
| Creative writing      | ~44   |
| Domain knowledge      | ~44   |
| **Total**             | **178** |

---

## Known Gaps

This is independent research done on a personal laptop with limited compute. A full list of what the study is missing and what would strengthen it is documented in `research_gaps.pdf` (attached separately with the paper).

Key gaps: single student architecture, single run per generation (no seeds), small prompt set, no human evaluation, no mitigation experiment, closed-source teachers only.

---

## Requirements

- Python 3.9+
- CUDA GPU recommended (CPU possible but very slow for training)
- OpenAI API key (for GPT-4o teacher)
- Anthropic API key (for Claude teacher)

See `requirements.txt` for full dependency list.

---

## Citation

If you use this code or build on this work, please reach out first.  
Contact: aramineni@wpi.edu
