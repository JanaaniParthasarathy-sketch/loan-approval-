"""
Loan Approval AI — Streamlit App
4 Tabs: Dashboard | Predictor | AI Chatbot | Analytics
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from backend.loan_model import get_dataset_stats, predict, train_model, _load_raw, FEATURE_COLS
from backend.chatbot import LoanAgent

# ─────────────────────────── page config ────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────── global CSS ─────────────────────────────────────
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; }
    h1 { color: #1f2328; font-size: 1.8rem !important; }
    h2 { color: #1f2328; font-size: 1.3rem !important; }
    h3 { color: #3b82d4; font-size: 1.05rem !important; }
    .metric-card {
        background: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 18px 20px;
        text-align: center;
    }
    .metric-card .val { font-size: 2rem; font-weight: 700; color: #3b82d4; }
    .metric-card .lbl { font-size: 0.82rem; color: #57606a; margin-top: 4px; }
    .chat-bubble-user {
        background: #3b82d4; color: white;
        border-radius: 14px 14px 2px 14px;
        padding: 10px 14px; margin: 6px 0 6px 60px;
        font-size: 0.92rem; line-height: 1.5;
    }
    .chat-bubble-bot {
        background: #f7f8fa; color: #1f2328;
        border: 1px solid #e5e7eb;
        border-radius: 14px 14px 14px 2px;
        padding: 10px 14px; margin: 6px 60px 6px 0;
        font-size: 0.92rem; line-height: 1.5;
        white-space: pre-wrap;
    }
    .approve-badge {
        background: #d1fae5; color: #065f46;
        border-radius: 8px; padding: 16px 20px;
        font-weight: 600; font-size: 1.1rem;
        border: 1px solid #6ee7b7;
    }
    .reject-badge {
        background: #fee2e2; color: #991b1b;
        border-radius: 8px; padding: 16px 20px;
        font-weight: 600; font-size: 1.1rem;
        border: 1px solid #fca5a5;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }
    .info-box {
        background: #eff6ff; border-left: 4px solid #3b82d4;
        padding: 12px 16px; border-radius: 0 8px 8px 0;
        font-size: 0.88rem; color: #1e3a5f; margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── session state ───────────────────────────────────
if "agent" not in st.session_state:
    st.session_state.agent = LoanAgent()
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "bot", "content":
         "Hello! 👋 I'm your Loan Approval AI Assistant.\n\n"
         "Ask me about eligibility criteria, credit history, documents required, "
         "EMI calculation, or tips to improve your approval chances!"}
    ]
if "model_trained" not in st.session_state:
    with st.spinner("Initialising ML model…"):
        train_model()
    st.session_state.model_trained = True

# ─────────────────────────── helpers ─────────────────────────────────────────

def metric_card(label: str, value: str):
    st.markdown(
        f'<div class="metric-card"><div class="val">{value}</div>'
        f'<div class="lbl">{label}</div></div>',
        unsafe_allow_html=True,
    )


def bar_chart(data: dict, title: str, color: str = "#3b82d4", suffix: str = "%"):
    fig, ax = plt.subplots(figsize=(5, 3))
    keys = list(data.keys())
    vals = list(data.values())
    bars = ax.barh(keys, vals, color=color, edgecolor="none")
    ax.set_xlabel(f"Approval Rate ({suffix})", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, v in zip(bars, vals):
        ax.text(v + 1, bar.get_y() + bar.get_height() / 2,
                f"{v}%", va="center", fontsize=8, color="#57606a")
    fig.tight_layout()
    return fig


def pie_chart(approved: int, rejected: int):
    fig, ax = plt.subplots(figsize=(4, 4))
    sizes = [approved, rejected]
    colors = ["#3b82d4", "#f87171"]
    labels = [f"Approved\n{approved}", f"Rejected\n{rejected}"]
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
           startangle=140, textprops={"fontsize": 9})
    ax.set_title("Loan Status Distribution", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig


# ─────────────────────────── TABS ────────────────────────────────────────────
st.markdown("## 🏦 Loan Approval AI")
st.markdown('<div class="info-box">AI-powered loan approval analysis and prediction using machine learning on real applicant data.</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard", "🔮 Predictor", "🤖 AI Chatbot", "📈 Analytics"
])

# ════════════════════════ TAB 1 — DASHBOARD ══════════════════════════════════
with tab1:
    st.markdown("### Dataset Overview")
    stats = get_dataset_stats()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Applications", str(stats["total"]))
    with c2:
        metric_card("Loans Approved", str(stats["approved"]))
    with c3:
        metric_card("Loans Rejected", str(stats["rejected"]))
    with c4:
        metric_card("Approval Rate", f"{stats['approval_rate']}%")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Approval by Property Area")
        fig = bar_chart(stats["by_property"], "Approval Rate by Property Area", color="#3b82d4")
        st.pyplot(fig)

    with col_r:
        st.markdown("#### Approval by Education")
        fig = bar_chart(stats["by_education"], "Approval Rate by Education", color="#7c5cd8")
        st.pyplot(fig)

    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        metric_card("Avg. Applicant Income", f"₹{stats['avg_income']:,}")
    with col_b:
        metric_card("Avg. Loan Amount", f"₹{stats['avg_loan']}L")
    with col_c:
        metric_card("Credit-History Approval Rate", f"{stats['credit_approved']}%")

    st.markdown("---")
    st.markdown("#### Loan Status Distribution")
    col_pie, col_txt = st.columns([1, 1])
    with col_pie:
        fig = pie_chart(stats["approved"], stats["rejected"])
        st.pyplot(fig)
    with col_txt:
        st.markdown(f"""
**Key Insights**

- **{stats['approval_rate']}%** of applicants were approved.
- Applicants with a positive credit history enjoy **{stats['credit_approved']}%** approval rate.
- **Semiurban** properties have the highest approval rate.
- **Graduate** applicants are more likely to get approved.
- Average approved loan amount: **₹{stats['avg_loan']}L**
""")

    # Raw data preview
    with st.expander("🗂 View Raw Dataset (first 20 rows)"):
        df_raw = _load_raw()
        st.dataframe(df_raw.head(20), use_container_width=True)

# ════════════════════════ TAB 2 — PREDICTOR ══════════════════════════════════
with tab2:
    st.markdown("### Loan Approval Predictor")
    st.markdown('<div class="info-box">Fill in the applicant details below and click <strong>Predict</strong> to get an AI-powered approval decision.</div>', unsafe_allow_html=True)

    with st.form("predictor_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Marital Status", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])

        with col2:
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
            credit_history = st.selectbox("Credit History", [1.0, 0.0],
                                          format_func=lambda x: "Good (1)" if x == 1.0 else "No History (0)")

        with col3:
            applicant_income = st.number_input("Applicant Income (₹/month)", min_value=0, value=5000, step=500)
            coapplicant_income = st.number_input("Co-applicant Income (₹/month)", min_value=0, value=0, step=500)
            loan_amount = st.number_input("Loan Amount (₹ thousands)", min_value=1, value=150, step=10)
            loan_term = st.selectbox("Loan Term (months)", [360, 180, 120, 84, 60, 36, 12], index=0)

        submitted = st.form_submit_button("🔮 Predict Approval", use_container_width=True)

    if submitted:
        form_data = {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": float(applicant_income),
            "CoapplicantIncome": float(coapplicant_income),
            "LoanAmount": float(loan_amount),
            "Loan_Amount_Term": float(loan_term),
            "Credit_History": float(credit_history),
            "Property_Area": property_area,
        }

        with st.spinner("Analysing application…"):
            label, prob = predict(form_data)

        col_res, col_gauge = st.columns([1, 1])
        with col_res:
            if label == "Y":
                st.markdown(
                    f'<div class="approve-badge">✅ Loan Approved &nbsp;|&nbsp; Confidence: {prob}%</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="reject-badge">❌ Loan Rejected &nbsp;|&nbsp; Confidence: {100 - prob}% rejection confidence</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(f"""
**Application Summary**
| Field | Value |
|---|---|
| Gender | {gender} |
| Married | {married} |
| Education | {education} |
| Self Employed | {self_employed} |
| Property Area | {property_area} |
| Credit History | {"Good" if credit_history == 1.0 else "No History"} |
| Total Income | ₹{applicant_income + coapplicant_income:,}/month |
| Loan Amount | ₹{loan_amount}L |
| Loan Term | {loan_term} months |
""")

        with col_gauge:
            # Approval probability bar
            fig, ax = plt.subplots(figsize=(5, 2.5))
            bar_color = "#3b82d4" if label == "Y" else "#f87171"
            ax.barh(["Approval\nProbability"], [prob], color=bar_color, height=0.4)
            ax.barh(["Approval\nProbability"], [100 - prob], left=[prob],
                    color="#e5e7eb", height=0.4)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probability (%)", fontsize=9)
            ax.set_title("Approval Probability Score", fontsize=10, fontweight="bold")
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.text(prob / 2, 0, f"{prob}%", ha="center", va="center",
                    fontsize=12, fontweight="bold", color="white")
            fig.tight_layout()
            st.pyplot(fig)

            # Agentic reasoning output
            st.markdown("**🧠 Agent Reasoning**")
            reasons = []
            if credit_history == 1.0:
                reasons.append("✅ Good credit history — strong positive signal")
            else:
                reasons.append("⚠️ No credit history — biggest risk factor")
            total_income = applicant_income + coapplicant_income
            if total_income > 0:
                ratio = loan_amount * 1000 / (total_income * 12)
                if ratio < 4:
                    reasons.append(f"✅ Loan-to-income ratio: {ratio:.1f}× (acceptable)")
                else:
                    reasons.append(f"⚠️ Loan-to-income ratio: {ratio:.1f}× (high)")
            if property_area == "Semiurban":
                reasons.append("✅ Semiurban area — historically highest approval rate")
            if education == "Graduate":
                reasons.append("✅ Graduate — slight positive influence")
            for r in reasons:
                st.markdown(r)

# ════════════════════════ TAB 3 — AI CHATBOT ═════════════════════════════════
with tab3:
    st.markdown("### 🤖 Loan AI Assistant")
    st.markdown('<div class="info-box">Ask our AI agent anything about loan eligibility, credit, documents, EMI, or tips to improve your chances.</div>', unsafe_allow_html=True)

    # Chat history display
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-bubble-user">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-bubble-bot">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # Quick question chips
    st.markdown("**Quick Questions:**")
    qcols = st.columns(4)
    quick_qs = [
        "What documents do I need?",
        "How does credit history affect approval?",
        "Tips to improve my chances",
        "How is EMI calculated?",
    ]
    triggered_q = None
    for i, qcol in enumerate(qcols):
        with qcol:
            if st.button(quick_qs[i], key=f"quick_{i}", use_container_width=True):
                triggered_q = quick_qs[i]

    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        col_in, col_btn = st.columns([5, 1])
        with col_in:
            user_input = st.text_input(
                "Your message", placeholder="Ask about loan eligibility, documents, EMI…",
                label_visibility="collapsed"
            )
        with col_btn:
            send = st.form_submit_button("Send ➤", use_container_width=True)

    # Process chat
    message_to_process = None
    if send and user_input.strip():
        message_to_process = user_input.strip()
    elif triggered_q:
        message_to_process = triggered_q

    if message_to_process:
        st.session_state.messages.append({"role": "user", "content": message_to_process})
        with st.spinner("Thinking…"):
            response = st.session_state.agent.chat(message_to_process)
        st.session_state.messages.append({"role": "bot", "content": response})
        st.rerun()

    # Reset button
    col_r1, col_r2 = st.columns([5, 1])
    with col_r2:
        if st.button("🔄 Reset Chat", use_container_width=True):
            st.session_state.agent.reset()
            st.session_state.messages = [
                {"role": "bot", "content":
                 "Chat reset! 👋 How can I help you with your loan application today?"}
            ]
            st.rerun()

# ════════════════════════ TAB 4 — ANALYTICS ══════════════════════════════════
with tab4:
    st.markdown("### 📈 Deep Analytics")
    st.markdown('<div class="info-box">Detailed statistical analysis of the loan dataset — income distributions, approval patterns, and feature correlations.</div>', unsafe_allow_html=True)

    df = _load_raw()

    # ── Income Distribution ──
    st.markdown("#### Income Distribution (Approved vs Rejected)")
    col_inc1, col_inc2 = st.columns(2)

    with col_inc1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        approved_inc = df[df["Loan_Status"] == "Y"]["ApplicantIncome"].dropna()
        rejected_inc = df[df["Loan_Status"] == "N"]["ApplicantIncome"].dropna()
        ax.hist(approved_inc.clip(upper=30000), bins=25, alpha=0.7,
                color="#3b82d4", label="Approved", edgecolor="white")
        ax.hist(rejected_inc.clip(upper=30000), bins=25, alpha=0.7,
                color="#f87171", label="Rejected", edgecolor="white")
        ax.set_xlabel("Applicant Income (₹)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title("Applicant Income Distribution", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)

    with col_inc2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        approved_la = df[df["Loan_Status"] == "Y"]["LoanAmount"].dropna()
        rejected_la = df[df["Loan_Status"] == "N"]["LoanAmount"].dropna()
        ax.hist(approved_la, bins=25, alpha=0.7,
                color="#3b82d4", label="Approved", edgecolor="white")
        ax.hist(rejected_la, bins=25, alpha=0.7,
                color="#f87171", label="Rejected", edgecolor="white")
        ax.set_xlabel("Loan Amount (₹ thousands)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title("Loan Amount Distribution", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)

    st.markdown("---")

    # ── Approval by Gender & Married ──
    st.markdown("#### Approval Rate by Demographics")
    col_g1, col_g2, col_g3 = st.columns(3)

    with col_g1:
        fig, ax = plt.subplots(figsize=(4, 3))
        gender_approval = df.groupby("Gender")["Loan_Status"].apply(
            lambda x: round((x == "Y").sum() / len(x) * 100, 1)).to_dict()
        ax.bar(gender_approval.keys(), gender_approval.values(),
               color=["#3b82d4", "#7c5cd8"], edgecolor="none")
        ax.set_ylabel("Approval Rate (%)", fontsize=9)
        ax.set_title("By Gender", fontsize=10, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 100)
        for x, v in enumerate(gender_approval.values()):
            ax.text(x, v + 1.5, f"{v}%", ha="center", fontsize=8, color="#57606a")
        fig.tight_layout()
        st.pyplot(fig)

    with col_g2:
        fig, ax = plt.subplots(figsize=(4, 3))
        married_approval = df.groupby("Married")["Loan_Status"].apply(
            lambda x: round((x == "Y").sum() / len(x) * 100, 1)).to_dict()
        ax.bar(married_approval.keys(), married_approval.values(),
               color=["#3b82d4", "#34d399"], edgecolor="none")
        ax.set_ylabel("Approval Rate (%)", fontsize=9)
        ax.set_title("By Marital Status", fontsize=10, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 100)
        for x, v in enumerate(married_approval.values()):
            ax.text(x, v + 1.5, f"{v}%", ha="center", fontsize=8, color="#57606a")
        fig.tight_layout()
        st.pyplot(fig)

    with col_g3:
        fig, ax = plt.subplots(figsize=(4, 3))
        dep_approval = df.groupby("Dependents")["Loan_Status"].apply(
            lambda x: round((x == "Y").sum() / len(x) * 100, 1)).to_dict()
        ax.bar(dep_approval.keys(), dep_approval.values(),
               color="#f59e0b", edgecolor="none")
        ax.set_ylabel("Approval Rate (%)", fontsize=9)
        ax.set_title("By Dependents", fontsize=10, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 100)
        for x, v in enumerate(dep_approval.values()):
            ax.text(x, v + 1.5, f"{v}%", ha="center", fontsize=8, color="#57606a")
        fig.tight_layout()
        st.pyplot(fig)

    st.markdown("---")

    # ── Feature Importance ──
    st.markdown("#### ML Model — Feature Importance")
    from backend.loan_model import load_model, CATEGORICAL_COLS
    clf, encoders, le_target, _ = load_model()
    importances = clf.feature_importances_
    feat_names = FEATURE_COLS
    sorted_idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#3b82d4" if i == sorted_idx[-1] else
              "#7c5cd8" if i == sorted_idx[-2] else "#93c5fd"
              for i in sorted_idx]
    ax.barh([feat_names[i] for i in sorted_idx], importances[sorted_idx],
            color=colors, edgecolor="none")
    ax.set_xlabel("Importance Score", fontsize=9)
    ax.set_title("Random Forest — Feature Importance", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    p1 = mpatches.Patch(color="#3b82d4", label="Most important")
    p2 = mpatches.Patch(color="#7c5cd8", label="2nd most important")
    p3 = mpatches.Patch(color="#93c5fd", label="Other features")
    ax.legend(handles=[p1, p2, p3], fontsize=7, loc="lower right")
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # ── Credit History Impact ──
    st.markdown("#### Credit History Impact on Approval")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        credit_grp = df.groupby(["Credit_History", "Loan_Status"]).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        x = np.arange(len(credit_grp.index))
        w = 0.35
        y_vals = credit_grp.get("Y", pd.Series([0] * len(credit_grp))).values
        n_vals = credit_grp.get("N", pd.Series([0] * len(credit_grp))).values
        ax.bar(x - w / 2, y_vals, w, label="Approved", color="#3b82d4", edgecolor="none")
        ax.bar(x + w / 2, n_vals, w, label="Rejected", color="#f87171", edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(["No History (0)", "Good History (1)"], fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title("Approval Count by Credit History", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)

    with col_c2:
        st.markdown("""
**Credit History — Key Statistics**

| Credit History | Approval Rate |
|---|---|
| No History (0) | ~35% |
| Good History (1) | ~80% |

Credit history is the **#1 most predictive feature** in the ML model.  
Applicants with a clean repayment track record are **2.3× more likely** to be approved.

**How to build credit history:**
- Pay all EMIs on time
- Use a credit card responsibly
- Avoid defaulting on any loan
- Maintain a low credit utilisation ratio (<30%)
""")
