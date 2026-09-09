# Preprocessing decisions

| Decision | Why | Rejected alternative |
|---|---|---|
| Reconstruct threads via `in_response_to_tweet_id` | Official schema; needed for conversation-level splits | Treat each tweet as i.i.d. |
| Pair inbound customer tweet with the **first** brand reply | Matches "issue → resolution sketch" | Concatenate all brand replies (noisy, leakage-prone) |
| Conversation ID = root of the parent chain | Multi-turn threads stay on one side of the split | Split on inbound tweet_id |
| HTML unescape, NFKC, collapse whitespace | `&amp;` should not be a feature | Full Unicode stripping |
| URLs → `URL` token | t.co links are not the issue | Keep raw URLs (sparsity) |
| @handles → `@USER`, keep `@AppleSupport` | Author identity leaks; brand mention is a cue | Drop all handles |
| Keep hashtag words | `#ios11` is intent signal | Drop hashtags |
| No stemming, no stopword list | "not charging" vs "charging" |
| Drop short / non-letter messages | "k" and emoji-only are unusable | Keep everything |
| Train labels = **silver keywords** | Realistic noise; golden set stays clean | Train on construction-time gold (leaks the evaluation label into the model) |
| Retrieval index = **train only** | Prevents copying the test conversation's own reply | Index all pairs |
| Do not commit `twcs.csv` | Size + Kaggle terms | Vendor the 3M-row file |

Silver labeling is intentionally imperfect (`src/data/silver_labels.py`). Ties go to `other`.
