# What is misleading about my headline number?

**Headline reported in README / REPORT:** proposed-agent intent **macro-F1 = 0.660** on the 200-example golden set.

That number is a real measurement from `python scripts/evaluate.py`. It is also easy to over-read.

## Class imbalance

Golden intent counts:

{
  "apple_id_icloud": 20,
  "backup_storage": 23,
  "billing_subscription": 25,
  "connectivity": 22,
  "hardware_device": 21,
  "how_to_feature": 22,
  "ios_update_bug": 24,
  "order_purchase": 22,
  "other": 21
}

Macro-F1 treats a 16-example intent the same as a 30-example intent. Accuracy would look friendlier if the majority class is easy. We report both; neither is the whole story. Per-class precision/recall in `reports/results.json` is the honest view.

## Easy examples

The sample corpus is template-generated. Many how-to questions contain the phrase "how do I", which both the silver labeler and the classifier pick up. Those examples inflate every system, including the TF-IDF baseline. They are still valid support tickets, but they are not the tickets that decide whether you trust auto-handle.

## Golden set size

n = 200. A 3-point macro-F1 gap can be a handful of items. We do not bootstrap confidence intervals in the demo path; do not treat the third decimal as stable.

## Sampling bias

The golden set is stratified by intent from the **test conversations of a synthetic Apple Support–style sample**, not a random draw from 2017 Twitter. Real AppleSupport volume in the Kaggle dump is dominated by iOS 11 / iPhone issues in a specific month. Temporal and product mix will shift.

## Temporal drift

The official corpus is from 2017 (iOS 11, iPhone X launch). An agent trained on it would be wrong about later products. We did not pretend otherwise.

## Conversation leakage

We split on `conversation_id` and index **train only**. Tests assert no ID overlap. If someone rebuilt the index on all pairs, retrieval metrics would be fictionally high. The current Recall@k is same-intent recall, not "did we retrieve this exact tweet" — the exact tweet is held out on purpose.

## Judge bias

The default judge is a **heuristic** (token overlap + promise regex + Apple-style cues). It is reproducible and strict about invented refunds, but it can reward replies that share words with the ticket without being useful. Copying a *wrong* historical neighbor still scores as "grounded" because the reply text *is* the evidence.

That is not a guess: on 40 developer-scored replies (`data/judge_human_validation.csv`), Pearson vs the heuristic is **0.36**, exact agreement **0.28**, within-1 **0.80**, Cohen's κ on the 0–4 score **0.03**. Human mean 2.38 vs judge mean 2.98. Grounded-label agreement is 0.75 with κ ≈ 0 because the judge almost always says grounded. An optional xAI judge exists; we do not mix those scales into the headline.

## Retrieval overlap

Because the sample uses shared templates, lexical retrieval can look strong without "understanding." That is a dataset artifact. On real tweets, expect Recall@5 to drop.

## Distribution mismatch

Train labels are **silver** (keywords). Golden labels follow the developer protocol. Some train items are intentionally mislabeled. Headline F1 is "system vs developer gold," not "system vs silver."

## Aggregate metrics hide minority-intent failures

A solid macro-F1 can still mean `other` or `order_purchase` is unusable. Escalation **false auto-handle rate** (0.241) matters more for trust than intent F1. We would rather over-escalate.

## Escalation class mix

Golden expected actions: {"ESCALATE": 112, "AUTO_HANDLE": 88}.
If most tickets are ESCALATE, a trivial "always escalate" baseline gets a high escalation F1. That is why we also report the majority baseline and false auto-handle rate.

## Bottom line

Use the headline to compare **proposed vs two baselines on the same frozen golden set**. Do not use it as a production go-live number.
