# Judge vs human agreement

File: `data/judge_human_validation.csv` (n=40).

## Sampling

After `python scripts/evaluate.py`, `scripts/annotate_judge.py --size 40 --seed 42` sampled proposed-agent golden predictions uniformly (shuffled with seed 42). The automatic (heuristic) judge filled `llm_judge_score` and `llm_grounded`. Human cells were empty until scored.

## Who labeled

**One developer annotator** (the same person who wrote the generator and the heuristic). This is not independent dual annotation and will *overstate* agreement relative to an ops team. Scores were assigned by reading the customer message, the draft reply, and the top retrieved evidence — not by looking at the heuristic score first (the CSV was filled from notes taken while reading the printed replies).

Scale: 0 = unusable, 1 = poor / wrong issue, 2 = acceptable but weak, 3 = good, 4 = excellent. `human_grounded` is true only if the reply addresses **this** customer's issue using evidence that is actually about that issue.

## Measured agreement

From `reports/human_agreement.json` after those 40 scores existed:

| Statistic | Value |
|---|---:|
| n | 40 |
| Pearson | 0.361 |
| Spearman | 0.369 |
| Exact agreement | 0.275 |
| Within-1 agreement | 0.800 |
| Cohen's κ (0–4 scores) | 0.032 |
| Grounded agreement | 0.750 |
| Grounded κ | 0.000 |
| Human mean | 2.375 |
| Judge mean | 2.975 |

## Disagreement pattern

The heuristic is **optimistic**, especially on groundedness. Typical miss: retrieve a same-intent but wrong-script neighbor (Watch pairing → CarPlay cable; order status → address change) and copy it. Token overlap with evidence is high, so the judge says 3/grounded; a human says 0–1/not grounded.

κ ≈ 0 on the 0–4 scale means exact-score agreement is barely above chance. Within-1 is 80%, so the judge is a coarse ranking signal, not a substitute for human review.

## Limitations

- Single rater, rater wrote the system.
- n=40.
- Heuristic judge, not an LLM (no `XAI_API_KEY` in the measurement environment).
- Synthetic tickets make some replies look more "on-template" than real 2017 tweets would.

Do not cite these numbers as "the LLM judge agrees with humans." They are developer vs heuristic-judge agreement on this demo.
