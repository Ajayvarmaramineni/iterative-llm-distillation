# Iterative LLM Distillation

> **What happens when a language model trains on its own outputs, generation after generation?**

This project studies multi-generation knowledge distillation in large language models. Two independent teacher lineages (GPT-4o and Claude Opus) each generate responses to a shared prompt set. A student model (OPT-1.3B) is fine-tuned on those outputs. Then the student's own outputs become the training data for the next generation. This loop repeats for five generations.

The result: both lineages collapse to the same degraded state by Generation 5, regardless of which teacher they started from. Vocabulary diversity collapses, repetition climbs, and coherence drops. We call this **convergent collapse**.

---

## Results at a Glance

![Headline Results](figures/fig1_headline.png)

*Six quality metrics tracked across five generations for both teacher lineages. All metrics degrade monotonically.*

---

## How Collapse Unfolds

<table>
  <tr>
    <td><img src="figures/fig3_radar.png" alt="Radar Chart" /></td>
    <td><img src="figures/fig5_heatmap.png" alt="Metric Heatmap" /></td>
  </tr>
  <tr>
    <td align="center"><em>Radar: Gen-0 vs Gen-5 across all metrics</em></td>
    <td align="center"><em>Heatmap: metric values across all generations</em></td>
  </tr>
</table>

---

## Faithfulness & Category Breakdown

<table>
  <tr>
    <td><img src="figures/fig2_faithfulness.png" alt="Faithfulness" /></td>
    <td><img src="figures/fig4_category_breakdown.png" alt="Category Breakdown" /></td>
  </tr>
  <tr>
    <td align="center"><em>Semantic faithfulness drift from Gen-0 baseline</em></td>
    <td align="center"><em>Collapse rate by prompt category</em></td>
  </tr>
</table>

---

## Collapse Delta

![Collapse Delta](figures/fig6_collapse_delta.png)

*Per-metric percentage change from Gen-0 to Gen-5. GPT-4o and Claude lineages converge to nearly identical degradation.*

---

## Experiment Setup

| Component | Detail |
|---|---|
| Student model | OPT-1.3B (`facebook/opt-1.3b`) |
| Teacher 1 | GPT-4o (OpenAI API) |
| Teacher 2 | Claude Opus (`claude-opus-4-6`) |
| Fine-tuning | Full fine-tuning, bfloat16 |
| Optimizer | AdamW, lr=1e-5, weight decay=0.01 |
| Epochs | 3 per generation |
| Generations | 5 (Gen 0 = teacher outputs, Gen 1-5 = student outputs) |
| Prompts | 178 across 4 categories |
| Metrics | TTR, repetition rate, BLEU-4, ROUGE-L, cosine similarity, perplexity |

---

## Repo Structure

```
iterative-llm-distillation/
├── data/
│   └── prompts/
│       └── prompts.json              # 178 prompts across 4 categories
├── src/
│   ├── generate_teacher_outputs.py   # Step 1: query GPT-4o / Claude for Gen-0 data
│   ├── train_student.py              # Step 2: fine-tune OPT-1.3B on teacher outputs
│   ├── distillation_loop.py          # Step 3: orchestrate the full multi-gen loop
│   └── evaluate_metrics.py           # Step 4: compute all metrics per generation
├── figures/                          # Result visualizations (PNG)
└── requirements.txt
```

---

## Quickstart

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

### 4. Run the full distillation loop

```bash
# GPT-4o lineage
python src/distillation_loop.py --model opt --teacher gpt4o --generations 5

# Claude lineage
python src/distillation_loop.py --model opt --teacher claude --generations 5
```

### 5. Evaluate all generations

```bash
python src/evaluate_metrics.py --teacher gpt4o --all_generations --max_gen 5
python src/evaluate_metrics.py --teacher claude --all_generations --max_gen 5
```

---

## Metrics

| Metric | What it measures | Collapse signal |
|---|---|---|
| TTR | Type-Token Ratio — vocabulary diversity per response | Drops toward 0 |
| Repetition Rate | Fraction of repeated n-grams within a response | Rises toward 1 |
| BLEU-4 | N-gram overlap with Gen-0 baseline | Drops |
| ROUGE-L | Longest common subsequence with Gen-0 baseline | Drops |
| Cosine Similarity | Intra-generation semantic homogenization | Rises |
| Perplexity | Fluency under a frozen GPT-2 reference model | Rises |

---

## Prompt Categories

| Category | Count |
|---|---|
| Question answering | ~45 |
| Logical reasoning | ~45 |
| Creative writing | ~44 |
| Domain knowledge | ~44 |
| **Total** | **178** |

---

## Requirements

- Python 3.9+
- CUDA GPU recommended (CPU works but training is slow)
- OpenAI API key (for GPT-4o teacher)
- Anthropic API key (for Claude teacher)

---

## Known Limitations

This study was run with limited compute on a personal machine. Notable constraints:

- Single student architecture (OPT-1.3B only)
- Single run per generation — no random seed variance
- 178 prompts is a relatively small evaluation set
- No human evaluation of output quality
- No mitigation experiments (e.g. data mixing, regularization)
- Closed-source teachers only; open-source teacher lineages not tested

---

## Contact

Ajay Ramineni — aramineni@wpi.edu
