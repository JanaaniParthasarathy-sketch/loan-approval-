"""
Loan Approval ML Backend
Trains a RandomForest classifier on the loan dataset and exposes predict().
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "loan_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "loan_model.pkl")
ENCODERS_PATH = os.path.join(os.path.dirname(__file__), "encoders.pkl")

CATEGORICAL_COLS = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
FEATURE_COLS = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Property_Area"
]


def _load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def _preprocess(df: pd.DataFrame, encoders: dict | None = None, fit: bool = False):
    df = df.copy()
    # Fill missing values (pandas ≥2 Copy-on-Write safe)
    df["Gender"] = df["Gender"].fillna("Male")
    df["Married"] = df["Married"].fillna("Yes")
    df["Dependents"] = df["Dependents"].astype(str).fillna("0")
    df["Self_Employed"] = df["Self_Employed"].fillna("No")
    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(360.0)
    df["Credit_History"] = df["Credit_History"].fillna(1.0)

    if fit:
        encoders = {}
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col in CATEGORICAL_COLS:
            le = encoders[col]
            df[col] = df[col].astype(str)
            # Handle unseen labels
            df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)

    return df, encoders


def train_model():
    """Train and persist the model. Returns (model, encoders, accuracy)."""
    df = _load_raw().dropna(subset=["Loan_Status"])
    df, encoders = _preprocess(df, fit=True)

    le_target = LabelEncoder()
    y = le_target.fit_transform(df["Loan_Status"])
    X = df[FEATURE_COLS]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump((clf, le_target), f)
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(encoders, f)

    return clf, encoders, le_target, round(acc * 100, 2)


def load_model():
    """Load persisted model, training if needed."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODERS_PATH):
        return train_model()
    with open(MODEL_PATH, "rb") as f:
        clf, le_target = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)
    return clf, encoders, le_target, None


def predict(form_data: dict) -> tuple[str, float]:
    """
    Predict loan approval.
    form_data keys: same as FEATURE_COLS.
    Returns (label, probability_approved).
    """
    clf, encoders, le_target, _ = load_model()
    row = pd.DataFrame([form_data])
    row, _ = _preprocess(row, encoders=encoders, fit=False)
    X = row[FEATURE_COLS]
    prob = clf.predict_proba(X)[0]
    pred = clf.predict(X)[0]
    label = le_target.inverse_transform([pred])[0]
    # probability of class Y (approved)
    approved_idx = list(le_target.classes_).index("Y") if "Y" in le_target.classes_ else 1
    return label, round(float(prob[approved_idx]) * 100, 1)


def get_dataset_stats() -> dict:
    """Return summary stats for the dashboard."""
    df = _load_raw()
    total = len(df)
    approved = int((df["Loan_Status"] == "Y").sum())
    rejected = int((df["Loan_Status"] == "N").sum())
    avg_income = int(df["ApplicantIncome"].mean())
    avg_loan = round(df["LoanAmount"].dropna().mean(), 1)
    by_property = df.groupby("Property_Area")["Loan_Status"].apply(
        lambda x: round((x == "Y").sum() / len(x) * 100, 1)
    ).to_dict()
    by_education = df.groupby("Education")["Loan_Status"].apply(
        lambda x: round((x == "Y").sum() / len(x) * 100, 1)
    ).to_dict()
    credit_approved = round(
        df[df["Credit_History"] == 1]["Loan_Status"].apply(lambda x: 1 if x == "Y" else 0).mean() * 100, 1
    )
    return {
        "total": total,
        "approved": approved,
        "rejected": rejected,
        "avg_income": avg_income,
        "avg_loan": avg_loan,
        "by_property": by_property,
        "by_education": by_education,
        "credit_approved": credit_approved,
        "approval_rate": round(approved / total * 100, 1),
    }
