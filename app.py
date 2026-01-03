import streamlit as st
import json

from claim_extractor import extract_claims
from claim_verifier import verify_claim_pipeline, compute_trust_score
from citation_verifier import verify_citations


# ================= SAMPLE DEMO TEXT =================
SAMPLE_TEXT = """
Studies show that large language models hallucinate in over 60% of cases.
According to recent research, citation errors are common in AI-generated text.
Smith et al. (2021) proposed hallucination mitigation techniques.
"""


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

# 🧪 Demo button
if st.button("🧪 Load Sample Demo Text"):
    st.session_state["input_text"] = SAMPLE_TEXT

input_text = st.text_area(
    "📄 Paste AI-generated text here:",
    height=250,
    key="input_text",
    placeholder="Paste text containing claims and citations..."
)


# ================= CLAIM VERIFICATION =================
verify_btn = st.button("🔍 Verify")

if verify_btn:
    if not input_text.strip():
        st.warning("Please paste some text.")
    else:
        claims = extract_claims(input_text)

        st.subheader("📊 Claim Verification Results")

        if not claims:
            st.info("No verifiable claims found.")
        else:
            # ✅ Collect results
            results = []

            for claim in claims:
                result = verify_claim_pipeline(claim)
                result["claim"] = claim
                results.append(result)

            # ================= TRUST SCORE =================
            trust_score = compute_trust_score(results)

            supported = sum(1 for r in results if r["label"] == "Supported")
            contradicted = sum(1 for r in results if r["label"] == "Contradicted")
            uncertain = sum(1 for r in results if r["label"] == "Not enough information")

            if trust_score >= 70:
                color = "🟢"
                message = "High confidence in generated content"
            elif trust_score >= 40:
                color = "🟡"
                message = "Moderate confidence – review uncertain claims"
            else:
                color = "🔴"
                message = "Low confidence – possible hallucinations detected"

            st.markdown(f"""
            ### 📊 Trust Score: {trust_score}%

            {color} **{message}**

            ↳ **{supported} Supported** • **{contradicted} Contradicted** • **{uncertain} Uncertain**
            """)

            st.divider()

            # ================= CLAIM CARDS =================
            for r in results:
                claim = r["claim"]
                label = r["label"]
                confidence = r["confidence"]
                evidence = r["evidence"]

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

                if show_evidence:
                    with st.expander("🔍 Evidence"):
                        st.write(evidence)
                        if show_confidence:
                            st.write(f"Confidence: {confidence:.2f}")

            # ================= DOWNLOAD REPORT =================
            st.divider()

            report_json = json.dumps(results, indent=2)

            st.download_button(
                label="⬇️ Download Verification Report (JSON)",
                data=report_json,
                file_name="verification_report.json",
                mime="application/json"
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
