# Decision log

Non-obvious choices. Each one is something an interviewer can push on.

## 1. Brand = AppleSupport

- **Decision:** Evaluate one brand, Apple Support.
- **Reason:** Published work on this corpus (Hardalov et al., 2018) already treats Apple as the largest usable slice; issues are diverse (OS, hardware, identity, billing, how-to); public replies have a stable style.
- **Alternative:** AmazonHelp (more volume) or Uber_Support.
- **Why rejected:** Amazon mix is logistics-heavy and multilingual; Uber tickets are almost all account-bound ("where's my driver") so AUTO_HANDLE is rarely honest.
- **Trade-off:** We inherit 2017 iOS-11 drift and Apple's later exit from Twitter support.

## 2. Synthetic sample instead of committing twcs.csv

- **Decision:** Ship a schema-compatible generator; download Kaggle only when credentials exist.
- **Reason:** 3M-row dump is too big, may be restricted, and was not present in this environment.
- **Alternative:** Check in a 50k-row slice of real tweets.
- **Why rejected:** Licensing/storage; also easy to accidentally leak user content into git.
- **Trade-off:** Template language inflates lexical retrieval and how-to F1. Documented in `reports/headline_number.md`.

## 3. Nine intents, not Banking77

- **Decision:** Data-derived 9-class taxonomy in `configs/intents.yaml`.
- **Reason:** The assignment asks for intents from the selected brand. Banking77 is retail-bank calls.
- **Alternative:** Map Apple tweets onto Banking77 or run k-means with k=20.
- **Why rejected:** Fake mapping; k-means clusters were not stable enough to be ops labels.
- **Trade-off:** `other` is a junk drawer and currently the worst class (F1 0.065).

## 4. Conversation-level random split, not temporal

- **Decision:** 70/10/20 by `conversation_id`, seed 42.
- **Reason:** Leakage control is the hard requirement; the sample is not a real timeline.
- **Alternative:** Hardalov-style last-five-days test split on real `created_at`.
- **Why rejected:** Synthetic timestamps are fake-sequential; a temporal split would be theatre.
- **Trade-off:** On the real dump we *should* switch to temporal. The code path does not assume order.

## 5. Silver labels on train, developer gold only on the 200

- **Decision:** Train on keyword silver labels; never fit on golden `true_intent`.
- **Reason:** Using construction-time gold for training would make golden-set F1 a memorization score.
- **Alternative:** Train on generator gold.
- **Why rejected:** It would look like 90%+ F1 and teach the wrong lesson.
- **Trade-off:** Proposed TF-IDF and the simple TF-IDF baseline are almost tied on intent (0.660 vs 0.656 macro-F1). That is honest.

## 6. No fine-tuned transformer

- **Decision:** Calibrated logistic regression on TF-IDF 1–2 grams + class centroids.
- **Reason:** Laptop, <15 minutes, deterministic, interview-explainable.
- **Alternative:** Fine-tune DistilBERT or call an LLM for every intent.
- **Why rejected:** Cost, non-determinism, and the assignment rewards evidence over architecture.
- **Trade-off:** Weak on paraphrase and on `other`. Optional xAI path exists for generation/judge only.

## 7. Retrieval = train-only TF-IDF cosine, k=5, light rerank

- **Decision:** sklearn TF-IDF, over-retrieve 4k, rerank with token overlap + intent bonus.
- **Reason:** Same vector space as the classifier; no FAISS/Chroma dependency.
- **Alternative:** sentence-transformers + FAISS; BM25-only.
- **Why rejected:** Extra downloads; BM25 alone ignores the intent we just predicted.
- **Trade-off:** k=5 same-intent Recall@5 is 0.645 and almost equal to Recall@1 — neighbors are redundant templates. Wrong-script neighbors (Watch pairing → CarPlay) still sneak through.

## 8. Copy/adapt historical replies instead of free-form generation

- **Decision:** Default generator sanitizes the top brand reply. LLM draft is opt-in.
- **Reason:** "Grounded in how the brand historically resolved similar issues" is literally nearest-neighbor.
- **Alternative:** Always LLM-write a new paragraph.
- **Why rejected:** Hallucinated refunds; unreproducible demo; cost.
- **Trade-off:** If retrieval is the wrong script, the copy is confidently wrong. Safety filter strips money/promises but not wrong troubleshooting.

## 9. Escalation policy is rules, not a classifier

- **Decision:** Multi-signal rules with a bias against false auto-handles.
- **Reason:** A second ML head would need its own golden labels and would hide the reason string.
- **Alternative:** Train a binary escalate model on `expected_action`.
- **Why rejected:** That's training on the golden protocol we also evaluate — circular.
- **Trade-off:** Extra escalations (7 in the golden cluster). False auto-handle rate is 0.241 — still too high for production.

## 10. Always escalate Apple ID and billing

- **Decision:** Those two intents cannot AUTO_HANDLE even with great evidence.
- **Reason:** The agent has no account API. Historical tweets that say "we issued your refund" would be unsafe to mimic.
- **Alternative:** Auto-handle if evidence similarity > 0.5.
- **Why rejected:** Similarity is not authorization.
- **Trade-off:** Escalation F1 can be beaten by "always escalate" on a gold set that is 56% ESCALATE. We therefore headline false auto-handle rate too.

## 11. Confidence is Platt-scaled, not "probability the intent is right"

- **Decision:** Calibrate on val silver labels; still describe scores as ranking-quality.
- **Reason:** Val labels are noisy; ECE on golden is reported, not advertised as calibrated frequencies.
- **Alternative:** Raw softmax, or no score at all.
- **Why rejected:** Raw scores over-claim; no score makes the policy threshold unexplained.
- **Trade-off:** Calibration fitted on silver val can be mis-calibrated on gold.

## 12. Heuristic judge as the default LLM-as-judge

- **Decision:** Deterministic rubric implementation; xAI only if `XAI_API_KEY` is set.
- **Reason:** Reproducible headline path without paid calls.
- **Alternative:** Require GPT-4/Grok for every evaluation.
- **Why rejected:** Reviewer laptop, cost, drift.
- **Trade-off:** Human agreement is weak (Pearson 0.36). The judge loves copied evidence.

## 13. Golden n=200, stratified, one annotator

- **Decision:** 200 test conversations, min 16/intent, developer-labeled via a written protocol encoded in `src/data/labeling.py`.
- **Reason:** Assignment range 150–250; stratification stops `other` from vanishing.
- **Alternative:** 50 hand-labeled real tweets, or crowd dual-annotation.
- **Why rejected:** No Kaggle dump in-environment; no second rater.
- **Trade-off:** Protocol rater = policy author, so escalation accuracy is inflated vs a real ops team.

## 14. Same-intent Recall@k, not exact-conversation hit

- **Decision:** A retrieved train tweet is relevant if its silver intent matches gold intent.
- **Reason:** The exact test conversation is held out on purpose. Exact-ID hit rate would be ~0 or would mean leakage.
- **Alternative:** Use the held-out brand reply as a ROUGE gold.
- **Why rejected:** Many Apple replies are interchangeable templates; ROUGE would reward copying the template, which we already do.
- **Trade-off:** Same-intent recall cannot see wrong-script retrieval inside an intent.

## 15. Do not tune thresholds on the golden set

- **Decision:** Confidence floor 0.38 and similarity floor 0.12 were set from the policy document before the 200-row scores were treated as a leaderboard. We did **not** sweep them to maximize golden F1.
- **Reason:** The assignment forbids tuning on individual golden examples.
- **Alternative:** Grid-search on golden to juice macro-F1.
- **Why rejected:** That is the thing the "headline number" section exists to prevent.
- **Trade-off:** A small sweep on **val** (silver) could still be done in a follow-up week.
