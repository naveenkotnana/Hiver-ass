"""Optional Streamlit UI. Not required for evaluation.

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from src.agent.support_agent import SupportAgent


@st.cache_resource
def load_agent() -> SupportAgent:
    return SupportAgent()


st.set_page_config(page_title="Apple Support Agent", layout="wide")
st.title("Apple Support — grounded draft + triage")
st.caption("Drafting tool. It cannot reset Apple IDs, issue refunds, or look up orders.")

message = st.text_area("Customer message", height=120)
context = st.text_input("Previous context (optional)")
if st.button("Run agent", type="primary") and message.strip():
    agent = load_agent()
    result = agent.handle(message, context=context)
    st.subheader("Predicted intent")
    st.write(f"`{result['intent']}`  (score {result['intent_confidence']:.3f})")
    st.subheader("Retrieved historical evidence")
    for i, hit in enumerate(result.get("evidence") or [], start=1):
        with st.expander(f"{i}. sim={hit.get('similarity'):.3f} · {hit.get('intent')}"):
            st.markdown("**Customer**")
            st.write(hit.get("customer_message"))
            st.markdown("**Brand**")
            st.write(hit.get("brand_response"))
    st.subheader("Draft reply")
    st.write(result["draft_reply"])
    st.subheader(result["decision"])
    st.write(result["reason"])
