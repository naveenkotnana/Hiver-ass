# Intent taxonomy

Nine intents. Derived from Apple Support themes in published analyses of the Kaggle corpus (iOS updates, battery/hardware, Apple ID/iCloud, App Store billing, connectivity, backups, orders, how-to) plus inspection of the bundled sample. **Banking77 is not the production taxonomy** — it is a bank-call dataset.

Machine-readable copy: `configs/intents.yaml`.

## How intents were discovered

1. Read Hardalov et al. 2018 and public AppleSupport tweet EDAs (battery, iOS, iCloud, charging).
2. Generated a schema-compatible sample spanning those themes, then read 80 random customer messages.
3. Clustered *informally* by issue type (not k-means as a source of truth). Candidate labels that could not change the **action** (auto-draft vs escalate vs which public doc) were merged.
4. Stopped at 9. A 77-way taxonomy cannot be labeled reliably at n=200 golden.

No BERT topic model was required to see that "how do I screenshot" and "unlock my Apple ID" are different jobs.

## Why each intent exists

| Intent | Exists because… | Typical action |
|---|---|---|
| `ios_update_bug` | iOS 11-era tweets are full of post-update crashes; troubleshooting is public. | Often AUTO_HANDLE |
| `hardware_device` | Battery, charging, screens need inspection / coverage decisions. | Usually ESCALATE |
| `apple_id_icloud` | Locked IDs, 2FA, iforgot. Agent cannot authenticate. | Always ESCALATE |
| `billing_subscription` | App Store / Music / iCloud charges. Refunds are not a Twitter action. | Always ESCALATE |
| `connectivity` | Wi-Fi / Bluetooth / CarPlay steps are public. | Often AUTO_HANDLE |
| `backup_storage` | Storage math is public; **data loss** is not. | Mixed |
| `order_purchase` | Needs the order record. | Always ESCALATE |
| `how_to_feature` | Pure information. | AUTO_HANDLE |
| `other` | Vague venting, third-party apps, off-topic. | ESCALATE (don't guess) |

## Ambiguous cases

**Primary-cause rule:** "After the iOS 11 update my battery dies" → `ios_update_bug`, not `hardware_device`. The update is the stated cause; the draft should start with software steps, then escalate if hardware remains.

**Billing vs Apple ID:** "Can't download apps because of a billing problem on my Apple ID" → `billing_subscription` (the blocker is a charge).

**Third-party crash:** "Uber crashes" → `other` unless they also report an iOS-wide failure.

**Order + charge:** "Charged for an iPhone that never shipped" → `order_purchase` (needs the order), not generic billing.

If two labels remain equally plausible, use `other` rather than coin a tenth intent.

## "other" handling

`other` is a first-class class, not noise to delete. The classifier must be allowed to say "I don't know." The escalation policy treats `other` as escalate. We do **not** train it away by folding leftovers into `complaint`.

## Why not Banking77

Banking77's labels (`card_arrival`, `beneficiary_not_allowed`, …) do not exist in Apple tweets. Using them would manufacture a mapping and a fake F1. The assignment asked for intents **defined from the selected brand's data**.
