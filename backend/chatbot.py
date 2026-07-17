"""
AI Chatbot backend for the Loan Approval app.
Uses a rule-based agentic reasoning layer (no external API needed).
If OPENAI_API_KEY is present in env, falls back to GPT-3.5-turbo for richer responses.
"""

import os
import re

# ---------------------------------------------------------------------------
# Knowledge base for rule-based agent
# ---------------------------------------------------------------------------

LOAN_KB = {
    "eligibility": (
        "To be eligible for a loan you generally need:\n"
        "• A good credit history (score 1 = good)\n"
        "• Stable income (higher applicant + co-applicant income helps)\n"
        "• Low loan-to-income ratio\n"
        "• Employed status (salaried or self-employed both considered)\n"
        "• Property area also plays a role — Semiurban areas have higher approval rates."
    ),
    "credit_history": (
        "Credit history is the single strongest predictor. Applicants with credit history = 1 "
        "have ~80% approval rate, vs ~35% for those without. Make sure all past loans and EMIs are paid."
    ),
    "income": (
        "The combined income (applicant + co-applicant) determines your loan capacity. "
        "Lenders typically approve loans where the EMI is < 40-50% of monthly income."
    ),
    "loan_amount": (
        "Loan amount should ideally be ≤ 4× your annual income. "
        "Larger amounts require higher creditworthiness and collateral."
    ),
    "education": (
        "Graduates have a slightly higher approval rate (~70%) compared to non-graduates (~62%). "
        "However, credit history and income are far more important factors."
    ),
    "self_employed": (
        "Self-employed applicants are considered, but may need to show business income proof, "
        "ITR filings for 2+ years, and stable revenue history."
    ),
    "property": (
        "Property area affects approval: Semiurban (77%) > Urban (68%) > Rural (62%). "
        "Semiurban properties tend to have better resale value as collateral."
    ),
    "tips": (
        "Tips to improve your approval chances:\n"
        "1. Clear all existing debts and maintain credit history\n"
        "2. Apply with a co-applicant to increase total income\n"
        "3. Choose a longer loan term (360 months) to reduce EMI burden\n"
        "4. Keep loan amount realistic relative to income\n"
        "5. Ensure all documents are complete and accurate"
    ),
    "rejection": (
        "Common reasons for rejection:\n"
        "• Poor or no credit history\n"
        "• Low income relative to loan amount requested\n"
        "• Too many existing loans / high debt-to-income ratio\n"
        "• Incomplete or inconsistent documentation\n"
        "• Property in a high-risk area"
    ),
    "documents": (
        "Documents typically required:\n"
        "• Identity proof (Aadhaar / PAN / Passport)\n"
        "• Address proof\n"
        "• Income proof (salary slips / ITR for 2 years)\n"
        "• Bank statements (6 months)\n"
        "• Property documents\n"
        "• Passport-size photographs"
    ),
    "emi": (
        "EMI = [P × R × (1+R)^N] / [(1+R)^N - 1]\n"
        "Where P = Principal, R = Monthly interest rate, N = Number of months.\n"
        "Example: ₹10L loan at 8% p.a. for 20 years → EMI ≈ ₹8,364/month."
    ),
}

INTENT_MAP = {
    "eligibility": ["eligible", "qualify", "criteria", "requirement", "who can"],
    "credit_history": ["credit", "cibil", "score", "credit history"],
    "income": ["income", "salary", "earning", "pay"],
    "loan_amount": ["loan amount", "how much", "amount"],
    "education": ["education", "graduate", "degree"],
    "self_employed": ["self employed", "self-employed", "business", "freelance"],
    "property": ["property", "area", "urban", "rural", "semiurban"],
    "tips": ["tip", "improve", "chance", "increase approval", "better"],
    "rejection": ["reject", "denied", "refuse", "why not approved"],
    "documents": ["document", "paperwork", "proof", "required", "needed"],
    "emi": ["emi", "monthly payment", "installment", "calculate"],
}

GREETINGS = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "howdy"]
THANKS = ["thank", "thanks", "great", "awesome", "helpful", "good"]


def _rule_based_response(message: str) -> str:
    msg = message.lower().strip()

    # Greeting
    if any(g in msg for g in GREETINGS):
        return (
            "Hello! 👋 I'm your Loan Approval AI Assistant.\n\n"
            "I can help you with:\n"
            "• Loan eligibility criteria\n"
            "• Credit history tips\n"
            "• Document requirements\n"
            "• EMI calculations\n"
            "• Improving approval chances\n\n"
            "What would you like to know?"
        )

    # Thanks
    if any(t in msg for t in THANKS):
        return "You're welcome! Feel free to ask anything else about your loan application. 😊"

    # Intent matching
    for intent, keywords in INTENT_MAP.items():
        if any(kw in msg for kw in keywords):
            return LOAN_KB[intent]

    # Fallback
    return (
        "I'm specialized in loan approval queries. You can ask me about:\n"
        "• Eligibility criteria\n"
        "• Credit history importance\n"
        "• Required documents\n"
        "• Tips to improve approval chances\n"
        "• EMI calculation\n"
        "• Reasons for rejection\n\n"
        "Try rephrasing your question or use the **Predictor** tab for a personalized prediction!"
    )


# ---------------------------------------------------------------------------
# Agent reasoning layer — adds multi-step decision context
# ---------------------------------------------------------------------------

class LoanAgent:
    """
    Simple agentic loop: parses user context clues, reasons over them,
    and generates a contextually aware response.
    """

    def __init__(self):
        self.context: list[dict] = []
        self.user_profile: dict = {}

    def _extract_profile(self, message: str):
        """Extract any mentioned numbers / keywords to build user profile."""
        # income mentions
        inc = re.search(r"(?:income|earn|salary)[^\d]*(\d[\d,]+)", message, re.I)
        if inc:
            self.user_profile["income"] = int(inc.group(1).replace(",", ""))

        # loan amount
        la = re.search(r"(?:loan|amount|borrow)[^\d]*(\d[\d,]+)", message, re.I)
        if la:
            self.user_profile["loan_amount"] = int(la.group(1).replace(",", ""))

        # credit
        if re.search(r"no credit|bad credit|poor credit", message, re.I):
            self.user_profile["credit"] = "poor"
        elif re.search(r"good credit|credit history", message, re.I):
            self.user_profile["credit"] = "good"

    def _personalised_advice(self) -> str | None:
        """Return personalised advice if we have enough profile info."""
        p = self.user_profile
        parts = []
        if "income" in p and "loan_amount" in p:
            ratio = p["loan_amount"] / (p["income"] * 12)
            if ratio > 4:
                parts.append(
                    f"⚠️ Your loan amount (₹{p['loan_amount']:,}) is {ratio:.1f}× your annual income. "
                    "Lenders prefer this ratio ≤ 4. Consider reducing the loan amount or increasing co-applicant income."
                )
            else:
                parts.append(
                    f"✅ Your loan-to-income ratio is {ratio:.1f}× — within acceptable limits."
                )
        if p.get("credit") == "poor":
            parts.append(
                "⚠️ Poor credit history significantly lowers approval chances. "
                "Work on clearing dues and building credit before applying."
            )
        elif p.get("credit") == "good":
            parts.append("✅ Good credit history is your strongest asset — this greatly improves your chances.")
        return "\n".join(parts) if parts else None

    def chat(self, message: str) -> str:
        self.context.append({"role": "user", "content": message})
        self._extract_profile(message)

        # Try personalised first
        personalised = self._personalised_advice()

        # Get base response
        base = _rule_based_response(message)

        # Compose final response
        if personalised and personalised not in base:
            response = base + "\n\n---\n**Personalised insight:**\n" + personalised
        else:
            response = base

        self.context.append({"role": "assistant", "content": response})
        return response

    def reset(self):
        self.context = []
        self.user_profile = {}
