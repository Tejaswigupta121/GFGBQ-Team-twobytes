import streamlit as st
from claim_extractor import extract_claims
from claim_verifier import verify_claim_pipeline, compute_trust_score
from citation_verifier import verify_citations


# ================= SIDEBAR =================
st.sidebar.title("⚙️ Settings")

top_k = st.sidebar.slider(
    "🔎 Evidence documents (Top-K)",
    min_value=1,
    max_value=5,
    value=3
)

show_confidence = st.sidebar.checkbox(
    "📊 Show confidence scores",
    value=True
)

show_evidence = st.sidebar.checkbox(
    "📄 Show retrieved evidence",
    value=True
)


# ================= PAGE CONFIG =================
st.set_page_config(page_title="AI Hallucination Checker", layout="wide")


# ================= STYLES =================
st.markdown("""
<style>
.claim-card {
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 12px;
    font-size: 16px;
}
.supported {
    background-color: #e6f4ea;
    border-left: 6px solid #2ecc71;
}
.contradicted {
    background-color: #fdecea;
    border-left: 6px solid #e74c3c;
}
.uncertain {
    background-color: #fff8e1;
    border-left: 6px solid #f1c40f;
}
</style>
""", unsafe_allow_html=True)


# ================= MAIN UI =================
st.title("🧠 AI Hallucination & Citation Verifier")

input_text = st.text_area(
    "📄 Paste AI-generated text here:",
    height=250,
    placeholder="Paste text containing claims and citations..."
)

verify_btn = st.button("🔍 Verify")


# ================= CLAIM VERIFICATION =================
if verify_btn:
    if not input_text.strip():
        st.warning("Please paste some text.")
    else:
        claims = extract_claims(input_text)

        st.subheader("📊 Claim Verification Results")

        if not claims:
            st.info("No verifiable claims found.")
        else:
            results = []  # ⭐ REQUIRED for trust score

            for claim in claims:
                result = verify_claim_pipeline(claim)
                results.append(result)

                label = result["label"]
                confidence = result["confidence"]
                evidence = result["evidence"]
                explanation = result.get("explanation", "No explanation available")

                if label == "Supported":
                    css_class = "supported"
                    icon = "🟢"
                elif label == "Contradicted":
                    css_class = "contradicted"
                    icon = "🔴"
                else:
                    css_class = "uncertain"
                    icon = "🟡"

                st.markdown(
                    f"""
                    <div class="claim-card {css_class}">
                        <strong>{icon} {label}</strong><br/>
                        {claim}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.caption(f"🧠 {explanation}")

                if show_evidence:
                    with st.expander("🔍 Evidence"):
                        st.write(evidence)
                        if show_confidence:
                            st.write(f"Confidence: {confidence:.2f}")

            # ================= TRUST SCORE =================
            st.divider()

            trust_score = compute_trust_score(results)

            st.metric(
                label="📊 Trust Score",
                value=f"{trust_score}%",
                help="Percentage of claims verified against trusted sources"
            )


# ================= CITATION VERIFICATION =================
st.divider()
st.subheader("📚 Citation Verification")

citation_results = verify_citations(input_text)

if not citation_results:
    st.info("No citations detected.")
else:
    for item in citation_results:
        citation = item["citation"]
        status = item["status"]

        if "✅" in status:
            st.success(f"{citation} → {status}")
        elif "❌" in status:
            st.error(f"{citation} → {status}")
        else:
            st.warning(f"{citation} → {status}")
