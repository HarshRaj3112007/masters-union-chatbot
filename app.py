"""
Streamlit UI for the Masters Union UG Data Science & AI Course Assistant.

Features
--------
- Streaming responses with typing indicator
- Dynamic follow-up question buttons
- Comparison mode for side-by-side queries
- Query history sidebar with clickable past questions
- Confidence badges (high / medium / low)
- Source citation expander
- Topic classification indicator
"""

import streamlit as st
from naya import (
    ask_question_stream,
    ask_comparison_stream,
    classify_query,
    detect_comparison,
    parse_followups,
)

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Masters Union DSAI Assistant",
    page_icon="🎓",
    layout="centered",
)

# ── Header ─────────────────────────────────────────────────────────────────────

st.title("🎓 Masters Union — UG Data Science & AI")
st.caption(
    "AI-powered assistant for the Undergraduate Programme in Data Science & AI. "
    "All answers are grounded in official programme materials."
)
st.divider()

# ── Session state init ─────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""

# ── Suggested questions ────────────────────────────────────────────────────────

SUGGESTED = [
    ("💰 Fee structure",         "What is the complete fee structure including scholarships?"),
    ("📚 Year-wise curriculum",  "What subjects are covered year by year in the programme?"),
    ("✅ Eligibility",           "Am I eligible if I don't have a JEE score?"),
    ("💼 Career outcomes",       "What jobs and companies can I join after this programme?"),
    ("🌍 Global track",          "Tell me about the Global Track and Illinois Tech degree."),
    ("🏫 Campus life",           "What facilities are available on campus?"),
    ("📝 How to apply",          "What is the step-by-step admissions process?"),
    ("🧑‍🏫 Faculty",               "Who are the faculty and industry mentors?"),
]

st.markdown("**Suggested questions — click to ask:**")
cols = st.columns(2)
for idx, (label, question) in enumerate(SUGGESTED):
    with cols[idx % 2]:
        if st.button(label, key=f"sug_{idx}", use_container_width=True):
            st.session_state.pending_question = question

# ── Chat history display ───────────────────────────────────────────────────────

st.divider()

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            # Confidence badge
            conf = msg.get("confidence", "")
            if conf == "low":
                st.warning("⚠ Low confidence — this may be outside programme scope.")
            elif conf == "medium":
                st.info("ℹ Moderate confidence — some details may need verification.")

            # Source citation
            if msg.get("source"):
                with st.expander("📄 View source context"):
                    st.text(msg["source"])

            # Topic badge
            if msg.get("topic"):
                st.caption(f"Query classified as: **{msg['topic'].upper()}**")

            # Follow-up buttons (for past messages, show as disabled-looking text)
            if msg.get("followups"):
                st.markdown("**You might also want to ask:**")
                for fi, fq in enumerate(msg["followups"]):
                    if st.button(
                        fq,
                        key=f"hist_followup_{i}_{fi}",
                        use_container_width=True,
                    ):
                        st.session_state.pending_question = fq
                        st.rerun()

# ── Chat input ─────────────────────────────────────────────────────────────────

question = st.chat_input("Ask anything about the programme ...")

# Handle suggested-question or follow-up button click
if st.session_state.pending_question and not question:
    question = st.session_state.pending_question
    st.session_state.pending_question = ""

# ── Process question ───────────────────────────────────────────────────────────

if question:
    # Display user message
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Detect comparison mode
    is_comparison, aspect_a, aspect_b = detect_comparison(question)

    # Generate answer with streaming
    with st.chat_message("assistant"):
        if is_comparison:
            result = ask_comparison_stream(question, aspect_a, aspect_b)
        else:
            result = ask_question_stream(question)

        # Stream or show static answer
        if result.get("answer_stream"):
            placeholder = st.empty()
            full_answer = ""
            display_text = ""
            buffer = ""
            followup_started = False

            for chunk in result["answer_stream"]:
                full_answer += chunk
                buffer += chunk

                if not followup_started:
                    if "FOLLOWUP:" in buffer:
                        before, _, after = buffer.partition("FOLLOWUP:")
                        display_text += before
                        followup_started = True
                        buffer = "FOLLOWUP:" + after
                    else:
                        display_text += buffer
                        buffer = ""
                    placeholder.markdown(display_text)

            if not followup_started and buffer:
                display_text += buffer
                placeholder.markdown(display_text)
        else:
            full_answer = result.get("answer_static") or result.get("answer", "")
            st.markdown(full_answer)

        # Do not parse or render follow-ups
        main_answer = full_answer
        followups = []

        topic = result.get("topic", classify_query(question))
        confidence = result.get("confidence", "high")
        source = result.get("source", "")

        # Confidence badge
        if confidence == "low":
            st.warning("⚠ Low confidence — this may be outside programme scope.")
        elif confidence == "medium":
            st.info("ℹ Moderate confidence — some details may need verification.")

        # Source citation
        if source:
            with st.expander("📄 View source context"):
                st.text(source)

        # Topic badge
        st.caption(f"Query classified as: **{topic.upper()}**")

        # Follow-ups intentionally removed

    # Store in history
    st.session_state.messages.append({
        "role": "assistant",
        "content": main_answer,
        "source": source,
        "topic": topic,
        "confidence": confidence,
        "followups": [],
    })

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.subheader("About this Assistant")
    st.markdown(
        """
**Data Sources**
- Official programme brochure
- Masters Union programme webpage (5 pages)
- Structured knowledge base

**Architecture**
- **Hybrid Search**: Semantic (ChromaDB) + BM25 keyword
- **RRF Re-ranking**: Reciprocal Rank Fusion
- **HyDE**: Query expansion via LLM
- Sentence-transformer embeddings
- Groq LLaMA 3.1 inference (streaming)

**Grounding**
Answers are sourced exclusively from programme
documents — no hallucination. Out-of-scope
queries are detected and flagged.
        """
    )

    st.divider()
    st.subheader("Direct Contact")
    st.markdown(
        """
📧 ugadmissions@mastersunion.org
📞 +91-7669186660
📍 DLF Cyberpark, Gurugram
        """
    )

    # ── Query History ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Query History")

    user_questions = [
        msg["content"] for msg in st.session_state.messages
        if msg["role"] == "user"
    ]

    if user_questions:
        for i, q in enumerate(reversed(user_questions)):
            display_text = q[:50] + ("..." if len(q) > 50 else "")
            if st.button(
                display_text,
                key=f"history_{i}",
                use_container_width=True,
                help=q,
            ):
                st.session_state.pending_question = q
                st.rerun()
    else:
        st.caption("No questions asked yet.")

    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()
