import streamlit as st
import json
from io import BytesIO

from claim_extractor import extract_claims
from claim_verifier import verify_claim_pipeline, compute_trust_score
from citation_verifier import verify_citations

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


# ================= SAMPLE DEMO TEXT =================
SAMPLE_TEXT = """
Studies show that large language models hallucinate in over 60% of cases.
According to recent research, citation errors are common in AI-generated text.
Smith et al. (2021) proposed hallucination mitigation techniques.
"""


# ================= PAGE CONFIG =================
st.set_page_config(page_title="AI Hallucination Checker", layout="wide")


# ================= SIDEBAR =================
st.sidebar.title("⚙️ Settings")

show_confidence = st.sidebar.checkbox("📊 Show confidence scores", value=True)
show_evidence = st.sidebar.checkbox("📄 Show retrieved evidence", value=True)


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

if st.button("🧪 Load Sample Demo Text"):
    st.session_state["input_text"] = SAMPLE_TEXT

input_text = st.text_area(
    "📄 Paste AI-generated text here:",
    height=250,
    key="input_text"
)

verify_btn = st.button("🔍 Verify")


# ================= PDF GENERATOR =================
def generate_pdf_report(results, trust_score):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>AI Hallucination Verification Report</b>", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Trust Score:</b> {trust_score}%", styles["Normal"]))
    story.append(Spacer(1, 12))

    for i, r in enumerate(results, 1):
        story.append(Paragraph(f"<b>Claim {i}:</b> {r['claim']}", styles["Normal"]))
        story.append(Paragraph(f"Verdict: {r['label']}", styles["Normal"]))
        story.append(Paragraph(f"Confidence: {r['confidence']}", styles["Normal"]))
        story.append(Paragraph(f"Explanation: {r.get('explanation','')}", styles["Normal"]))
        story.append(Spacer(1, 15))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ================= VERIFICATION LOGIC =================
if verify_btn:
    if not input_text.strip():
        st.warning("Please paste some text.")
    else:
        claims = extract_claims(input_text)

        if not claims:
            st.info("No verifiable claims found.")
        else:
            results = []

            for claim in claims:
                r = verify_claim_pipeline(claim)
                confidence = r["confidence"]
                evidence = r["evidence"]
                explanation = r["explanation"]
                r["claim"] = claim
                results.append(r)

            # ================= TRUST SCORE =================
            trust_score = compute_trust_score(results)

            supported = sum(1 for r in results if r["label"] == "Supported")
            contradicted = sum(1 for r in results if r["label"] == "Contradicted")
            uncertain = sum(1 for r in results if r["label"] == "Not enough information")

            if trust_score >= 70:
                icon, msg = "🟢", "High confidence in generated content"
            elif trust_score >= 40:
                icon, msg = "🟡", "Moderate confidence – review uncertain claims"
            else:
                icon, msg = "🔴", "Low confidence – hallucinations likely"

            st.markdown(f"""
            ### 📊 Trust Score: {trust_score}%

            {icon} **{msg}**

            ↳ **{supported} Supported • {contradicted} Contradicted • {uncertain} Uncertain**
            """)

            st.divider()

            # ================= CLAIM CARDS =================
            for r in results:
                label = r["label"]
                css = "supported" if label == "Supported" else "contradicted" if label == "Contradicted" else "uncertain"
                icon = "🟢" if label == "Supported" else "🔴" if label == "Contradicted" else "🟡"

                st.markdown(
                    f"""
                    <div class="claim-card {css}">
                        <strong>{icon} {label}</strong><br/>
                        {r["claim"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if show_evidence:
                    with st.expander("🔍 Evidence"):
                        st.write(r["evidence"])
                        if show_confidence:
                            conf = r.get("confidence", 0.0)
                            if label== "Not enough information":
                                st.write(f"⚠️ Model confidence (neutrality): {confidence: .2f}")
                            else:
                                st.write(f"Confidence: {confidence: .2f}")
                        

            # ================= DOWNLOADS =================
            st.divider()

            st.download_button(
                "⬇️ Download JSON Report",
                json.dumps(results, indent=2),
                "verification_report.json",
                "application/json"
            )

            pdf = generate_pdf_report(results, trust_score)
            st.download_button(
                "📄 Download PDF Report",
                pdf,
                "ai_verification_report.pdf",
                "application/pdf"
            )


# ================= CITATION VERIFICATION =================
st.divider()
st.subheader("📚 Citation Verification")

citations = verify_citations(input_text)

if citations:
    for c in citations:
        if "✅" in c["status"]:
            st.success(f"{c['citation']} → {c['status']}")
        elif "❌" in c["status"]:
            st.error(f"{c['citation']} → {c['status']}")
        else:
            st.warning(f"{c['citation']} → {c['status']}")
else:
    st.info("No citations detected.")
