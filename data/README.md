# Data

| Path | Committed? | What it is |
|---|---|---|
| `sample/tweets.csv` | generated, gitignored if huge; small sample is produced by `prepare_data.py` | Schema-compatible Apple Support–style tweets |
| `sample/gold_intents.csv` | generated | Construction-time intent map. **Not used for training.** |
| `raw/twcs.csv` | **never commit** | Official Kaggle dump if you download it |
| `processed/{train,val,test}.jsonl` | generated | Conversation-level splits |
| `golden_set.csv` / `golden_set.jsonl` | **yes** | Held-out developer-labeled evaluation set (~200) |
| `judge_human_validation.csv` | **yes** | 40-example judge vs human sheet |
| `brand_selection.md` | yes | Pointer to the full write-up |

## How to get the official dataset

1. Create a Kaggle API token (`~/.kaggle/kaggle.json`) or set `KAGGLE_USERNAME` / `KAGGLE_KEY`.
2. `pip install kaggle`
3. `python scripts/prepare_data.py --kaggle`

The pipeline filters to `AppleSupport`, reconstructs threads, and rebuilds splits. The bundled golden set is for the **sample** corpus; relabel a fresh golden set with `scripts/annotate_golden.py` if you switch to the real dump.

## Why a sample exists

The Kaggle file is ~3M tweets. The assignment expects a reviewer subsample that runs on a laptop in under 15 minutes. The sample follows the official CSV schema and Apple Support public-reply style; it is **synthetic**, not a scrape of the Kaggle file. Brand choice and taxonomy still come from published analyses of the real corpus — see `docs/brand_selection.md`.
