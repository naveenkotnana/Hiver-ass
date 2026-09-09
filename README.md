# Apple Support AI agent (Hiver SDE intern take-home)

Draft-and-triage agent for **one brand** from the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) corpus: **@AppleSupport**.

It classifies an inbound message, retrieves how Apple historically answered similar issues, drafts a **grounded** reply, and decides **AUTO_HANDLE** vs **ESCALATE** with a reason. Evaluation quality is the point; the model is deliberately boring.

```
customer message
    → preprocess
    → intent (TF-IDF + calibrated logistic regression)
    → historical retrieval (train-only TF-IDF, k=5)
    → grounded draft (copy/sanitize historical reply)
    → escalation policy
    → intent + decision + reason + draft + evidence
```

The agent **cannot** unlock Apple IDs, issue refunds, or look up orders.

## Quickstart (under 15 minutes)

Python 3.10+. From the repo root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_demo.py
```

That command:

1. Builds the Apple Support sample (or uses Kaggle if configured)
2. Fits the classifier and the train-only retrieval index
3. Evaluates majority / TF-IDF / proposed systems on the **200-example golden set**
4. Prints headline metrics and five sample predictions

One-command evaluation after the demo:

```bash
python scripts/evaluate.py
```

Example prediction:

```bash
python -m src.cli --message "I was charged twice for my order"
# or
python scripts/predict.py --message "how do I turn off Wi-Fi Assist on iOS 11?"
```

Optional API:

```bash
uvicorn app:app --port 8000
# POST /predict  {"message": "I was charged twice for my order"}
```

Optional UI (assignment demo screen: message → intent → evidence → draft → AUTO_HANDLE/ESCALATE → reason):

```bash
streamlit run streamlit_app.py
```

Requires the models from `python scripts/run_demo.py` first. Not required for evaluation.

## Dataset setup

| What | How |
|---|---|
| Demo sample (default) | `python scripts/prepare_data.py` — synthetic, schema-compatible, seed 42 |
| Official Kaggle dump | Set `KAGGLE_USERNAME` / `KAGGLE_KEY` or `~/.kaggle/kaggle.json`, then `python scripts/prepare_data.py --kaggle` |

**Do not commit `twcs.csv`.** See `data/README.md`. Brand choice: `docs/brand_selection.md`.

## Environment variables

Copy `.env.example`. Nothing is required for the graded path.

| Variable | Purpose |
|---|---|
| `XAI_API_KEY` | Optional SpaceXAI/xAI drafts + LLM judge (`https://api.x.ai/v1`, default model `grok-4.5`) |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | Optional full corpus |
| `RANDOM_SEED` | Default 42 |

No API keys are hardcoded.

## Expected outputs

`python scripts/run_demo.py` should print a table close to:

| System | Intent acc | Intent macro-F1 | Escalation F1 | False auto-handle | Recall@5 | Judge mean |
|---|---:|---:|---:|---:|---:|---:|
| majority | 0.110 | 0.022 | 0.718 | 0.000 | 0.000 | 2.000 |
| tfidf_baseline | 0.645 | 0.656 | 0.715 | 0.339 | 0.640 | 2.805 |
| **proposed** | **0.645** | **0.660** | **0.752** | **0.241** | **0.645** | **2.845** |

CLI shape:

```
Intent:
billing_subscription

Decision:
ESCALATE

Reason:
Intent 'billing_subscription' requires an authenticated specialist...

Draft Reply:
...
Evidence:
1. ...
```

Re-running with seed 42 on the bundled generator should match these numbers within ordinary float noise. If you point the pipeline at real `twcs.csv`, **relabel a new golden set** — do not reuse `data/golden_set.csv`.

## Results (this repo)

- **Selected brand:** AppleSupport
- **Pairs used:** 2,780 reconstructed customer→brand pairs (train 1,945 / val 289 / test 546)
- **Golden set:** 200 developer-labeled test conversations (`data/golden_set.csv`)
- **Headline:** proposed intent **macro-F1 = 0.660**
- **Why that is misleading:** `reports/headline_number.md` (class `other` F1 0.065, synthetic templates, always-escalate F1 trap, optimistic judge)
- **False auto-handle rate:** 0.241 (the number I would actually discuss in an interview)
- **LLM/heuristic judge:** 0–4 rubric in `src/evaluation/judge.py`; default is a deterministic heuristic so the demo has no API bill
- **Human agreement (n=40, developer rater vs heuristic):** Pearson 0.36, exact 0.28, within-1 0.80, Cohen's κ 0.03 — `docs/judge_human_agreement.md`

Full write-up: **`REPORT.md`**. Decisions: **`DECISION_LOG.md`**. Failures: **`reports/failure_analysis.md`**.

## Tests

```bash
python -m pytest tests -q
```

Covers reconstruction, conversation-level leakage, classifier/retrieval schema, escalation, metrics, and `POST /predict`.

## Design in one paragraph

Intents come from Apple Support themes, not Banking77. Training labels are **silver** (keywords); the golden set is held out. Retrieval never sees test conversations. Replies are historical Apple text with money/promise stripping — if evidence is weak we clarify or escalate instead of inventing policy. Escalation is a transparent rule list biased against false auto-handles. Complexity we skipped on purpose: fine-tuned transformers, FAISS, multi-agent graphs.

## Limitations

- Bundled tweets are **synthetic**. Taxonomy and brand choice still rest on published analyses of the real Kaggle file.
- Golden labels and the 40 judge scores are **single-annotator developer labels**, not crowd raters.
- 24% false auto-handle is not a ship number.
- 2017 product mix. Apple later left live Twitter support.

## License

MIT. Kaggle data remains under its own terms — this repo does not redistribute it.
