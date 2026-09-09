# Golden set

- Files: `data/golden_set.csv`, `data/golden_set.jsonl`
- Target size: **200** (assignment range 150–250)
- Held out from training, calibration, retrieval indexing, and threshold choices

## Sampling strategy

1. Reconstruct customer→brand pairs.
2. Split **by conversation_id** 70/10/20 (seed 42).
3. From **test only**, take a stratified sample (`min_per_intent` from `configs/config.yaml`).
4. Those conversation IDs never appear in train/val files.

This is not a random tweet sample from 3M rows. It is stratified on the demo corpus so minority intents are visible.

## Labeling methodology

**Single developer annotator.** There was no second rater and no crowd worker.

For the bundled sample, each conversation is constructed with a source intent. The golden label is that source intent after applying the written disambiguation rules in `docs/intent_taxonomy.md` (cause vs symptom, billing vs Apple ID, third-party → other). Escalation labels are **not** copied from the model; they come from `src/data/labeling.py`, which is the protocol:

- Legal / fraud / safety language → ESCALATE
- Apple ID / billing / order → ESCALATE
- Hardware that won't power on / swollen battery / replacement ask → ESCALATE
- Severe anger or "still not fixed" → ESCALATE
- How-to and generic iOS/connectivity troubleshooting → AUTO_HANDLE
- `other` → ESCALATE

That protocol is conservative on purpose (false auto-handle is the costly error).

When you switch to real `twcs.csv`, **do not reuse these 200 rows**. Sample test conversations and label them with `python scripts/annotate_golden.py`.

## Label definitions

See `configs/intents.yaml` and `docs/intent_taxonomy.md`. `expected_action` ∈ {`AUTO_HANDLE`, `ESCALATE`}. `escalation_reason` is filled only when the action is ESCALATE.

## Ambiguous-label policy

If unsure between two intents, prefer the one that changes the action. If still unsure, `other` + ESCALATE. Notes field records the dilemma.

## Inter-rater methodology

**Not performed.** One annotator (the developer). We do not report Cohen's kappa for intent labels because there is no second human. Judge-vs-human agreement (response quality) is a separate file: `data/judge_human_validation.csv`.

## Limitations

- Developer-labeled, so it can agree with the escalation policy that the same developer wrote. That inflates escalation accuracy relative to an independent ops team.
- Synthetic sample language is cleaner than 2017 Twitter.
- n=200 is small; per-intent estimates are noisy.
- Not a temporal split of the real dump (Hardalov et al. used last-five-days). We used a conversation-level random split on the sample, which is the right leakage control for this corpus.
