# Hiver SDE Intern Assignment

## 1. Executive Summary

I built a **draft-and-triage** agent for **Apple Support** tweets. It classifies a customer message into 9 data-derived intents, retrieves how Apple historically answered similar issues, drafts a reply from that evidence, and decides **AUTO_HANDLE** vs **ESCALATE** with a written reason.

It cannot reset Apple IDs, issue refunds, or see orders. On a frozen **200-example golden set**:

| System | Intent Macro-F1 | Escalation F1 | False auto-handle | Retrieval Recall@5 | Judge overall |
|---|---:|---:|---:|---:|---:|
| Majority baseline | 0.022 | 0.718 | 0.000 | 0.000 | 2.000 |
| TF-IDF baseline | 0.656 | 0.715 | 0.339 | 0.640 | 2.805 |
| **Proposed agent** | **0.660** | **0.752** | **0.241** | **0.645** | **2.845** |

The headline **intent macro-F1 = 0.660** is a real measurement and a weak reason to trust the bot. The proposed model is only barely better than unigram TF-IDF because both train on the same noisy silver labels. The actual gain is a **lower false auto-handle rate** (0.241 vs 0.339) from a conservative policy. "Always escalate" still wins raw escalation F1 — which is why that metric is a trap.

## 2. Problem Framing

**Good** for this brand means: (1) the intent is specific enough to choose a playbook, (2) the draft sounds like public Apple Support and does not invent a refund/unlock/ETA, (3) account, billing, hardware-coverage, and legal tickets go to a human.

The agent **does** draft troubleshooting and how-to replies, and it **does** say why it escalated.

The agent **does not** authenticate users, look up orders, book Genius Bar, or send DMs.

## 3. Dataset

Official corpus: Kaggle `thoughtvector/customer-support-on-twitter` (~3M tweets, 2017). Not committed. `python scripts/prepare_data.py --kaggle` downloads it when credentials exist.

This repo runs on a **schema-compatible Apple Support–style sample** (2,780 usable pairs; train 1,945 / val 289 / test 546) so a reviewer finishes in minutes. Construction-time intents are **not** used as training labels.

## 4. Brand Selection

**AppleSupport.** Largest (or second-largest) brand in published counts of this corpus; Hardalov et al. 2018 used 49,626 Apple dialog tuples. Diverse issues and a distinctive public style (Settings > General > About, iforgot.apple.com, "we cannot refund on Twitter"). See `docs/brand_selection.md`.

## 5. Intent Taxonomy

Nine intents from Apple themes, not Banking77: `ios_update_bug`, `hardware_device`, `apple_id_icloud`, `billing_subscription`, `connectivity`, `backup_storage`, `order_purchase`, `how_to_feature`, `other`. Ambiguity rule: label the **stated cause**, not the loudest symptom. `other` is first-class and currently the failure sink.

## 6. System Architecture

```
message → normalize → TF-IDF+LR intent (Platt-scaled on val)
        → train-only TF-IDF retrieval (k=5, overlap/intent rerank)
        → copy/sanitize historical reply (optional xAI rewrite)
        → multi-signal escalation policy → {intent, decision, reason, draft, evidence}
```

Offline by default. `XAI_API_KEY` enables Grok drafts and an LLM judge.

## 7. Evaluation Methodology

Conversation-level split; tests forbid ID overlap. Retrieval index is train-only. Golden **n=200**, stratified from **test**, developer-labeled with `src/data/labeling.py` (single annotator). Train uses silver keywords only.

Metrics: intent accuracy / macro-F1 / per-class / confusion; escalation P/R/F1 and **false auto-handle rate**; same-intent Recall@k and MRR; heuristic judge 0–4.

## 8. Baselines

1. **Trivial:** majority train intent (`how_to_feature`), always ESCALATE, canned "we'll look into this."
2. **Simple:** unigram TF-IDF+LR, copy nearest neighbor, escalate a short risky-intent list.

## 9. Results

Measured by `python scripts/evaluate.py` on the committed golden set.

**Intent (proposed).** Accuracy 0.645. Per-class F1: billing 0.98, Apple ID 0.98, iOS update 0.79, connectivity 0.78, order 0.74, how-to 0.62, backup 0.61, hardware 0.39, **other 0.07**. Hardware recall is 0.24; how-to precision is 0.47 (`other` leaks in).

**Escalation (proposed).** Accuracy 0.72, F1 0.752, false auto-handle **0.241** (27/112 gold-escalate cases). Baseline 2 is 0.339. Majority is 0.000 because it never auto-handles.

**Retrieval.** Recall@5 = 0.645 ≈ Recall@1. Neighbors are near-duplicates.

**Judge.** Mean overall 2.845 / 4. Hallucination rate 0.0 on the regex (no invented `$` / "refund has been issued"). Groundedness is **inflated** — see §11.

**Human vs judge (n=40, developer rater):** Pearson 0.36, exact 0.28, within-1 0.80, κ 0.03. Human mean 2.38 vs judge 2.98.

## 10. Failure Analysis

Real examples from `artifacts/predictions/proposed.jsonl`:

1. **Intent misclassification (24)** — `gold_020` photos not uploading → `other` / ESCALATE. Silver features missed "photos"+"iCloud".
2. **Symptom dumped to `other` (20)** — `gold_051` hotspot drops → `other`.
3. **`other` absorbed into how-to (19)** — `gold_129` "love the packaging" → how-to / AUTO_HANDLE. Off-topic tweets look like support questions.
4. **False auto-handle (8 remaining after #3)** — `gold_067` dropped iPad, "covered?" → how-to / AUTO_HANDLE. The *draft* correctly refused to guess coverage; the *decision* should still be ESCALATE.
5. **Extra escalation (7)** — `gold_018` backup failed nightly, gold AUTO_HANDLE, policy ESCALATE. Conservative on purpose.

## 11. What Is Misleading About My Headline Number?

See `reports/headline_number.md`. Short version: n=200; synthetic templates; silver vs gold mismatch; `other` F1 0.07 hidden by macro average; always-escalate looks strong on F1; heuristic judge scores copied wrong neighbors as grounded; no temporal test on the real dump.

## 12. Limitations

- Sample ≠ 2017 Twitter. Expect retrieval and how-to F1 to drop on `twcs.csv`.
- One annotator wrote the policy and the gold.
- False auto-handle 24% is not production-safe.
- No live account tools — by design.
- Optional LLM path untested here (no API key).

## 13. What I Would Do With One More Week

1. Relabel 200 **real** AppleSupport test threads from `twcs.csv` (temporal holdout).
2. A first-pass `other` detector so off-topic never AUTO_HANDLEs.
3. Intra-intent rerank so "Watch pairing" cannot fetch "CarPlay cable."
4. Second rater on 50 golden rows (report κ).
5. Sweep confidence/similarity floors on **val only**, freeze, then re-score gold once.
