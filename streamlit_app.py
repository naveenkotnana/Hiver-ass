"""Streamlit demo for the Apple Support draft-and-triage agent.

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.agent.support_agent import SupportAgent
from src.paths import ARTIFACTS

EXAMPLES = [
    "I was charged twice for my Apple Music subscription",
    "how do I turn off Wi-Fi Assist on iOS 11?",
    "my iPhone 7 keeps restarting after I updated to iOS 11",
    "my Apple ID is locked and I never got the reset email",
    "iPhone X will not stay connected to Wi-Fi",
    "dropped my iPad and now the screen has dead pixels. is this covered?",
]


@st.cache_resource
def load_agent() -> SupportAgent:
    return SupportAgent()


def models_ready() -> bool:
    return (ARTIFACTS / "models" / "intent_classifier.joblib").exists() and (
        ARTIFACTS / "index" / "tfidf.joblib"
    ).exists()


st.set_page_config(page_title="Apple Support Agent", layout="wide")
st.title("Apple Support — grounded draft + triage")
st.caption(
    "Drafting tool only. It cannot reset Apple IDs, issue refunds, look up orders, "
    "or book a Genius Bar. AUTO_HANDLE means the public reply is safe to send as a draft."
)

st.markdown(
    "Customer message → **intent** → **historical evidence** → **draft reply** → "
    "**AUTO_HANDLE / ESCALATE** → **reason**"
)

if not models_ready():
    st.error(
        "Models are not trained yet. From the repo root run:\n\n"
        "`python scripts/run_demo.py`"
    )
    st.stop()

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Customer message")
    example = st.selectbox("Try an example", ["(type your own)"] + EXAMPLES)
    default = "" if example == "(type your own)" else example
    message = st.text_area("Inbound tweet / message", value=default, height=140)
    context = st.text_input("Previous conversation context (optional)")
    run = st.button("Run agent", type="primary")

with right:
    st.subheader("What this screen shows")
    st.markdown(
        """
1. **Predicted intent** — 9-class taxonomy from Apple Support themes  
2. **Retrieved historical evidence** — train-only similar tickets  
3. **Draft reply** — adapted from how Apple answered similar issues  
4. **AUTO_HANDLE / ESCALATE** — policy decision with a stated reason  
        """
    )

if run:
    if not message.strip():
        st.warning("Enter a customer message first.")
        st.stop()
    try:
        agent = load_agent()
        result = agent.handle(message, context=context)
    except FileNotFoundError:
        st.error("Could not load the trained agent. Run `python scripts/run_demo.py`.")
        st.stop()

    intent_col, decision_col = st.columns(2)
    with intent_col:
        st.subheader("Predicted intent")
        st.markdown(f"**`{result['intent']}`**")
        st.caption(
            f"score {result['intent_confidence']:.3f}"
            + (" · Platt-scaled on val (not a guaranteed probability)" if result.get("intent_calibrated") else "")
        )
        alts = result.get("alternatives") or []
        if alts:
            st.write("Alternatives:")
            for alt in alts:
                st.write(f"- `{alt.get('intent')}` ({float(alt.get('score') or 0):.3f})")

    with decision_col:
        st.subheader("Decision")
        decision = result["decision"]
        if decision == "ESCALATE":
            st.error(decision)
        else:
            st.success(decision)
        st.write(result["reason"])
        st.caption(f"Grounded in retrieved evidence: {bool(result.get('grounded'))}")

    st.subheader("Retrieved historical evidence")
    evidence = result.get("evidence") or []
    if not evidence:
        st.info("No historical neighbors above the similarity floor.")
    for i, hit in enumerate(evidence, start=1):
        label = (
            f"{i}. sim={float(hit.get('similarity') or 0):.3f} · "
            f"{hit.get('intent') or 'unknown'} · id={hit.get('conversation_id')}"
        )
        with st.expander(label, expanded=(i == 1)):
            st.markdown("**Historical customer**")
            st.write(hit.get("customer_message") or "")
            st.markdown("**Historical Apple Support reply**")
            st.write(hit.get("brand_response") or "")

    st.subheader("Draft reply")
    st.write(result.get("draft_reply") or result.get("reply") or "")
    if decision == "ESCALATE":
        st.caption("Specialist holding line is included in the CLI output; this box is the draft for the human.")
