# Project Plan — Hiver SDE Intern Take-Home

## Objectives

Build a **single-brand** AI support agent on Customer Support on Twitter data that:

1. Classifies an inbound customer message into a **small, data-derived** intent taxonomy.
2. Drafts a reply **grounded in historical Apple Support resolutions** (retrieval, not free-form policy invention).
3. Decides **AUTO_HANDLE** vs **ESCALATE**, with a stated reason.
4. **Proves** the system is good enough to trust — evaluation quality over architecture theatre.

The agent is a **drafting and triage** tool. It cannot reset Apple IDs, issue refunds, look up orders, or replace hardware.

## Architecture

```
incoming message
    → preprocess / normalize
    → intent classifier   (TF-IDF + calibrated logistic regression)
    → historical retrieval (TF-IDF cosine over train conversations, k=5, rerank)
    → response generator   (adapt top evidence; refuse to invent policy)
    → escalation policy    (transparent multi-signal rules)
    → final {intent, reply, decision, reason, evidence}
```

Offline-first. Optional xAI (`XAI_API_KEY`) can draft replies and score them as an LLM judge. Without a key, generation and judging use deterministic local components so a reviewer can reproduce headline numbers in under 15 minutes.

No multi-agent graph, no fine-tuned LLM, no FAISS dependency. sklearn is enough for a laptop demo and for an interview explanation.

## Datasets

| Source | Role |
|---|---|
| Kaggle `thoughtvector/customer-support-on-twitter` | Official corpus. Downloadable via `src/data/download.py`. **Not committed.** |
| Bundled Apple Support–style sample | Schema-compatible subsample generated deterministically (`seed=42`) so the repo runs without Kaggle credentials. |
| Banking77 | **Not** used as the production taxonomy. Optional contrast only. |

**Selected brand: AppleSupport.** Criterion and published corpus evidence: `docs/brand_selection.md`.

## Evaluation strategy

- **Conversation-level split** (70/10/20). No conversation ID in more than one split.
- **Golden set (200)** sampled from the **test** conversations only, developer-labeled under a written protocol (`docs/golden_set.md`). Held out from training and threshold selection.
- Train labels are **silver** (keyword rules). Golden labels are **not** used to fit the classifier.
- Retrieval index is **train-only**.
- Metrics: intent accuracy / macro-F1 / per-class P/R / confusion; escalation accuracy / F1 / **false auto-handle rate**; retrieval Recall@k and MRR (same-intent relevance, because the exact test conversation is held out); response quality via a structured judge.
- LLM-as-judge with a local heuristic fallback. Human agreement is measured on a 40-example sample **after** replies exist — never fabricated.

## Baselines

1. **Trivial:** majority intent + always ESCALATE + canned reply.
2. **Simple:** unigram TF-IDF + logistic regression; nearest-neighbor reply copy; escalate on a short high-risk intent list.

The proposed system adds bigrams, class weights, validation-set probability calibration, intent-aware reranking, safety-filtered grounded generation, and a multi-signal escalation policy.

## Deliverables

Runnable git repository with CLI, FastAPI `/predict`, tests, golden set, evaluation reports, decision log, and a ≤6-page `REPORT.md`.

## Milestones

0. Plan and brand selection (this document).
1. Ingestion, thread reconstruction, leakage-aware splits.
2. Intent taxonomy from Apple Support themes + sample inspection.
3. Golden set and labeling protocol.
4. Baselines + proposed agent.
5. Evaluation harness, judge, failure analysis, headline-number caveats.
6. Tests, README quickstart, demo script.

## Assumptions

- Kaggle credentials may be absent; the sample must be sufficient to reproduce the demo.
- The sample is **synthetic and schema-compatible**, not a dump of copyrighted tweet text. Taxonomy and brand choice are grounded in published analyses of the real corpus (Hardalov et al., 2018; public Kaggle EDA).
- Golden labels are **single-annotator developer labels**, not independent crowd ratings.
- Confidence scores are Platt-scaled on validation data; they are not claimed as perfectly calibrated frequencies.
- Prefer extra escalations over false auto-handles.
- No network is required for the graded quickstart path.
