# Escalation policy

Output: `{decision: AUTO_HANDLE|ESCALATE, reason, confidence}`.

`confidence` is **policy certainty**, not a calibrated probability.

## Signals (any one can escalate)

1. Intent in `{apple_id_icloud, billing_subscription}` — always.
2. Intent confidence below `escalation.min_intent_confidence` (default 0.38).
3. No retrieved evidence above `min_evidence_similarity` (default 0.12), or generator `grounded=false`.
4. Legal / safety / fraud language.
5. Requested account action: refund, unlock, replace, cancel order/subscription, lost photos.
6. Severe anger / repeated unresolved wording.
7. High-risk intents (`order_purchase`, plus the always-escalate set).
8. Taxonomy default is ESCALATE and nothing else contradicted it.

## Non-goals

The system **does not** reset Apple IDs, issue refunds, inspect hardware, or see orders. AUTO_HANDLE only means "this draft is safe to send as a public troubleshooting / how-to reply."

## Bias

Prefer extra escalations over false auto-handles. The headline to watch is **false auto-handle rate**, not escalation F1 (an always-escalate baseline can look strong on F1).
