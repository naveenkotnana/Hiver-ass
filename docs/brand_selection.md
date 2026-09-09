# Brand selection

## Official corpus

[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (thoughtvector, 2017): ~3 million tweets, 20+ companies, CSV columns `tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id`.

This environment did **not** have Kaggle credentials, so we did not recount `author_id` frequencies from `twcs.csv` here. Brand choice is based on **published** measurements of that same file, and the runnable demo uses a schema-compatible Apple Support sample. When `twcs.csv` is present, `src/data/download.py:brand_counts` will print live frequencies.

## Candidate brands (published measurements)

Hardalov, Koychev & Nakov, *Towards Automated Customer Support* ([arXiv:1809.00303](https://arxiv.org/abs/1809.00303)), trained only on Apple because it was the **largest** brand slice they used. They report **49,626 Apple dialog tuples** after filtering (45,582 train / 4,044 test, last five days held out).

Public Kaggle EDA notebooks on `twcs.csv` consistently rank outbound / support authors roughly as:

| Rank (typical) | Brand handle | Why it is / isn't a fit |
|---|---|---|
| 1 | AmazonHelp | Huge volume, but many logistics one-liners and mixed locales. |
| 2 | **AppleSupport** | Large, English-heavy, diverse product issues, distinctive DM-for-accounts style. |
| 3 | Uber_Support | High volume, but trip-specific (pickup pin, driver) — hard to ground without account APIs. |
| 4 | SpotifyCares | Narrower (playback, premium). Fine, but less escalation diversity. |
| 5 | Delta / AmericanAir / British_Airways / SouthwestAir | Strong, but operational (fares, delays) and highly time-sensitive. |
| 6 | Tesco / sainsburys | Grocery; different policy world. |
| 7 | TMobileHelp / VerizonSupport / SprintCares / comcastcares | Telco; account-bound almost always. |
| 8 | XboxSupport / AskPlayStation | Gaming; would also work. |

Exact integer counts should be recomputed from `twcs.csv` (`python -c "from src.data.download import brand_counts; ..."`). We do **not** treat the table above as a live query.

## Selection criterion (decided before modeling)

Score each brand on:

1. **Volume** — enough pairs for train/val/test **and** a 150–250 golden set.
2. **Issue diversity** — not a single intent (e.g. not only "where's my driver").
3. **Publicly grounded replies** — historical tweets that actually contain troubleshooting, not only "please DM".
4. **Escalation texture** — some tickets a bot may draft, some a bot must not.
5. **Interview explainability** — a reviewer who has used an iPhone can judge replies.

**Selected brand: AppleSupport.**

Reasons:

- Largest (or second-largest) slice in the official corpus; Hardalov et al. already treated it as the canonical subset.
- Published topic work (e.g. analyses of iOS updates, battery, iCloud) shows a **small natural taxonomy**, not 77 banking intents.
- Historical replies have a stable style: ask for Settings > General > About, send people to iforgot.apple.com / reportaproblem.apple.com, refuse to take passwords on Twitter.
- Clear **non-goals**: the agent must not unlock Apple IDs or issue refunds — perfect for an escalation policy.

## Limitations

- 2017 product mix (iOS 11, iPhone X). Drift is real; see `reports/headline_number.md`.
- Apple left live Twitter support in 2023 (The Verge). This project studies the **historical** channel the dataset actually contains.
- Many real Apple replies are "please DM". A grounded system should copy that pattern for account issues, not invent a backend.
- Without `twcs.csv` in this workspace, we cannot claim a live row count. The demo sample is labeled as synthetic in `data/README.md`.
