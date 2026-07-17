# 🏦 Loan Approval AI — Streamlit App

An AI-powered loan approval prediction and analysis application built with **Streamlit**, **scikit-learn**, and an **Agentic AI chatbot**.

## Features

| Tab | Description |
|---|---|
| 📊 **Dashboard** | Dataset overview — approval rates, income stats, key metrics |
| 🔮 **Predictor** | ML-powered loan approval prediction with agent reasoning |
| 🤖 **AI Chatbot** | Agentic AI assistant for loan eligibility and tips |
| 📈 **Analytics** | Deep analytics — distributions, demographics, feature importance |

## Tech Stack

- **Frontend**: Streamlit 1.x
- **ML Model**: Random Forest Classifier (scikit-learn)
- **AI Agent**: Rule-based agentic reasoning with contextual user profiling
- **Data**: Real loan approval dataset (614 applicants)
- **Visualisation**: Matplotlib

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

### 3. Open in browser

Navigate to `http://localhost:8501`

## Project Structure

```
loan-approval/
├── app.py                  # Main Streamlit frontend
├── requirements.txt
├── data/
│   └── loan_data.csv       # Dataset
├── backend/
│   ├── loan_model.py       # ML model training & prediction
│   └── chatbot.py          # Agentic AI chatbot
└── .streamlit/
    └── config.toml         # Theme configuration
```

## ML Model Details

- **Algorithm**: Random Forest (100 estimators)
- **Features**: Gender, Married, Dependents, Education, Self-Employed, Income, Loan Amount, Term, Credit History, Property Area
- **Target**: Loan Status (Y/N)
- **Key Insight**: Credit History is the strongest predictor (~80% of model weight)

## AI Chatbot Agent

The chatbot uses an agentic design pattern:
1. **Intent Detection** — classifies user query into known topics
2. **Profile Extraction** — extracts income/loan/credit mentions from conversation
3. **Personalised Reasoning** — generates contextually aware responses based on user profile
4. **Knowledge Base** — covers eligibility, documents, EMI, tips, rejection reasons

## Dataset

Source: Loan Prediction dataset with 614 records and 13 features including applicant demographics, income, loan details, and credit history.

## Deploying on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file to `app.py`
5. Deploy!

---
Built with ❤️ using Streamlit & scikit-learn
