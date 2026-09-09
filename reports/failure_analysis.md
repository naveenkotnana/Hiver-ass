# Failure analysis

Generated from **actual** proposed-agent predictions on the golden set.
Examples are copied from `artifacts/predictions/proposed.jsonl`. They are not invented.

## 1. intent misclassification

- Count in cluster: **24**
- Example ID: `gold_020`
- Customer message: @AppleSupport photos not uploading to iCloud from iPhone 8 Plus for 3 days
- Expected: intent `backup_storage`, action `AUTO_HANDLE`
- Actual: intent `other`, action `ESCALATE`
- Judge overall: 3
- Why the system failed: The classifier assigned the wrong taxonomy label, so retrieval was steered toward the wrong historical cluster.
- Hypothesis: Class overlap plus silver-label noise on train; minority intents have fewer distinctive n-grams.
- Potential fix: Inspect the confusion matrix cells with highest volume; add targeted features or a second-stage binary for that pair.

## 2. ambiguous intent symptom vs cause

- Count in cluster: **20**
- Example ID: `gold_051`
- Customer message: @AppleSupport hotspot from iPhone 7 connects then loses internet immediately
- Expected: intent `connectivity`, action `AUTO_HANDLE`
- Actual: intent `other`, action `ESCALATE`
- Judge overall: 2
- Why the system failed: The message names a symptom (battery, Wi-Fi) that also belongs to another intent, while the cause (iOS update, hardware age) is the gold label.
- Hypothesis: Overlapping keyword features; silver training labels are even noisier on mixed-symptom tweets.
- Potential fix: Add a cause-vs-symptom feature (e.g. 'after updating') or a small set of contrastive examples; do not merge those intents — they need different actions.

## 3. other absorbed into how to

- Count in cluster: **19**
- Example ID: `gold_129`
- Customer message: @AppleSupport love the packaging. that is the tweet
- Expected: intent `other`, action `ESCALATE`
- Actual: intent `how_to_feature`, action `AUTO_HANDLE`
- Judge overall: 3
- Why the system failed: Vague or off-topic tweets were classified as how-to questions, so the agent drafted a product-support reply instead of declining or escalating.
- Hypothesis: The `other` class has weak n-grams; `how_to_feature` is a magnet because questions and @AppleSupport mentions look like support requests. Silver training labels also dump ties into other, so the decision boundary is messy.
- Potential fix: Add a cheap first-pass: if the message has no product symptom keywords, emit `other` and escalate. Evaluate that rule on val, not on golden.

## 4. incorrect escalation false auto handle

- Count in cluster: **8**
- Example ID: `gold_067`
- Customer message: @AppleSupport dropped iPad from waist height and now dead pixels. covered?
- Expected: intent `hardware_device`, action `ESCALATE`
- Actual: intent `how_to_feature`, action `AUTO_HANDLE`
- Judge overall: 3
- Why the system failed: Policy auto-handled a case the golden protocol says needs a human.
- Hypothesis: Intent looked like how-to/troubleshooting and evidence was similar, while the gold label saw account-specific risk.
- Potential fix: Raise the cost of false auto-handles: broaden ACCOUNT_ACTION patterns; never auto-handle backup + 'photos gone'.

## 5. incorrect escalation

- Count in cluster: **7**
- Example ID: `gold_018`
- Customer message: @AppleSupport iCloud backup failed every night this week on iPhone 6s
- Expected: intent `backup_storage`, action `AUTO_HANDLE`
- Actual: intent `backup_storage`, action `ESCALATE`
- Judge overall: 3
- Why the system failed: Escalation decision disagreed with the labeling guide.
- Hypothesis: Policy is conservative (extra escalations) which hurts AUTO_HANDLE recall on purpose.
- Potential fix: If extra escalations dominate, narrow always-escalate intents after measuring false auto-handle, not before.
