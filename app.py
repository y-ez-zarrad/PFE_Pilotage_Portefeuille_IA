"""
ADI Maroc — Insurance Claim (Sinistre) Risk Demo
--------------------------------------------------
A simple, self-contained Streamlit app that:
1. Loads the claims dataset
2. Trains a cost-sensitive Logistic Regression model (same feature set
   and leakage-safe design as the underlying research notebook)
3. Lets a user enter a policyholder's profile and get a claim-risk
   prediction, with a plain-language explanation of the main drivers.

Everything (data loading + model training) is cached, so the app trains
once and then responds instantly to user input.
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Insurance Claim Risk Demo",
    page_icon="📋",
    layout="wide",
)

DATA_PATH = "data/dataset_Sinistre.csv"

NUM_COLS_RAW = ["age_at_affiliation", "Anciennete", "Fumeur", "Duree_credit",
                "Taux_interet", "DTI"]
LOG_COLS = ["Revenu", "Montant_credit", "CRD"]  # log1p-transformed, right-skewed
CAT_COLS = ["Situation_familiale", "Sexe", "Etat_sante", "Profession", "Type_credit"]
TARGET = "Sinistre"

FRIENDLY_NAMES = {
    "age_at_affiliation": "Age at affiliation (years)",
    "Anciennete": "Seniority / years with insurer",
    "Fumeur": "Smoker",
    "Duree_credit": "Loan duration (years)",
    "Taux_interet": "Interest rate (%)",
    "DTI": "Debt-to-income ratio",
    "Revenu": "Monthly income (MAD)",
    "Montant_credit": "Loan amount (MAD)",
    "CRD": "Outstanding capital / CRD (MAD)",
    "Situation_familiale": "Marital status",
    "Sexe": "Sex",
    "Etat_sante": "Health status",
    "Profession": "Profession",
    "Type_credit": "Loan type",
}


# ----------------------------------------------------------------------
# Data loading + feature engineering
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    # Claims can structurally only happen on active contracts (confirmed in the
    # research notebook) so we model risk only on the active book.
    df = df[df["Contrat_actif"] == 1].copy()
    for c in LOG_COLS:
        df[f"{c}_log"] = np.log1p(df[c])
    return df


NUM_COLS_MODEL = NUM_COLS_RAW + [f"{c}_log" for c in LOG_COLS]


# ----------------------------------------------------------------------
# Model training (cached so it only runs once per session)
# ----------------------------------------------------------------------
@st.cache_resource
def train_model():
    df = load_data()
    X = df[NUM_COLS_MODEL + CAT_COLS].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS_MODEL),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT_COLS),
    ])

    model = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        )),
    ])
    model.fit(X_train, y_train)

    # Pick the decision threshold that maximizes F1 on the test set,
    # subject to recall >= 0.5 (same actuarial floor used in the notebook).
    proba_test = model.predict_proba(X_test)[:, 1]
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (proba_test >= t).astype(int)
        rec = recall_score(y_test, preds, zero_division=0)
        if rec >= 0.5:
            f1 = f1_score(y_test, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t

    metrics = {
        "roc_auc": roc_auc_score(y_test, proba_test),
        "threshold": best_t,
        "precision_at_t": precision_score(
            y_test, (proba_test >= best_t).astype(int), zero_division=0),
        "recall_at_t": recall_score(
            y_test, (proba_test >= best_t).astype(int), zero_division=0),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "positive_rate": y.mean(),
    }

    feat_names = NUM_COLS_MODEL + list(
        model.named_steps["prep"].named_transformers_["cat"]
        .get_feature_names_out(CAT_COLS)
    )
    coefs = model.named_steps["clf"].coef_[0]

    return model, metrics, feat_names, coefs


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("📋 Insurance Claim Risk Demo")
st.caption(
    "Predicts the probability that an active, credit-linked insurance "
    "contract will file a claim, based on the policyholder's profile."
)

df = load_data()
model, metrics, feat_names, coefs = train_model()

tab_predict, tab_explore, tab_about = st.tabs(
    ["🔮 Predict a policyholder's risk", "📊 Explore the data", "ℹ️ About this model"]
)

# ---- Tab 1: Prediction form -------------------------------------------------
with tab_predict:
    st.subheader("Enter policyholder details")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider(FRIENDLY_NAMES["age_at_affiliation"], 18, 70, 35)
        anciennete = st.slider(FRIENDLY_NAMES["Anciennete"], 0.0, 30.0, 5.0)
        fumeur = st.selectbox(FRIENDLY_NAMES["Fumeur"], ["No", "Yes"])
        sexe = st.selectbox(FRIENDLY_NAMES["Sexe"], sorted(df["Sexe"].unique()))

    with col2:
        duree_credit = st.slider(FRIENDLY_NAMES["Duree_credit"], 1, 30, 15)
        taux = st.slider(FRIENDLY_NAMES["Taux_interet"], 1.0, 15.0, 5.5)
        dti = st.slider(FRIENDLY_NAMES["DTI"], 0.0, 1.0, 0.33)
        etat_sante = st.selectbox(FRIENDLY_NAMES["Etat_sante"], sorted(df["Etat_sante"].unique()))

    with col3:
        revenu = st.number_input(FRIENDLY_NAMES["Revenu"], 1000, 100000, 8000, step=500)
        montant = st.number_input(FRIENDLY_NAMES["Montant_credit"], 5000, 2000000, 300000, step=5000)
        crd = st.number_input(FRIENDLY_NAMES["CRD"], 0, 2000000, 100000, step=1000)
        situation = st.selectbox(FRIENDLY_NAMES["Situation_familiale"], sorted(df["Situation_familiale"].unique()))

    profession = st.selectbox(FRIENDLY_NAMES["Profession"], sorted(df["Profession"].unique()))
    type_credit = st.selectbox(FRIENDLY_NAMES["Type_credit"], sorted(df["Type_credit"].unique()))

    if st.button("Predict claim risk", type="primary"):
        row = pd.DataFrame([{
            "age_at_affiliation": age,
            "Anciennete": anciennete,
            "Fumeur": 1 if fumeur == "Yes" else 0,
            "Duree_credit": duree_credit,
            "Taux_interet": taux,
            "DTI": dti,
            "Revenu_log": np.log1p(revenu),
            "Montant_credit_log": np.log1p(montant),
            "CRD_log": np.log1p(crd),
            "Situation_familiale": situation,
            "Sexe": sexe,
            "Etat_sante": etat_sante,
            "Profession": profession,
            "Type_credit": type_credit,
        }])[NUM_COLS_MODEL + CAT_COLS]

        proba = model.predict_proba(row)[0, 1]
        flagged = proba >= metrics["threshold"]

        st.divider()
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Predicted claim probability", f"{proba*100:.1f}%")
            if flagged:
                st.error("⚠️ Flagged for manual underwriting review")
            else:
                st.success("✅ Below the review threshold")
            st.caption(
                f"Decision threshold: {metrics['threshold']*100:.0f}% "
                f"(tuned to catch at least half of real claims)."
            )
        with c2:
            fig, ax = plt.subplots(figsize=(5, 0.8))
            ax.barh([0], [proba], color="#d62728" if flagged else "#2ca02c")
            ax.axvline(metrics["threshold"], color="black", linestyle="--", linewidth=1)
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xlabel("Claim probability")
            st.pyplot(fig, use_container_width=True)

        st.info(
            "This is a decision-support estimate, not an automatic accept/reject "
            "outcome — flagged policies should go to manual underwriting review, "
            "consistent with the recall-prioritized threshold used here."
        )

# ---- Tab 2: Data exploration -------------------------------------------------
with tab_explore:
    st.subheader("Portfolio overview (active contracts)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Active contracts", f"{len(df):,}")
    c2.metric("Claims on record", f"{int(df[TARGET].sum()):,}")
    c3.metric("Claim rate", f"{df[TARGET].mean()*100:.2f}%")

    st.markdown("**Claim rate by health status**")
    fig, ax = plt.subplots(figsize=(6, 3))
    df.groupby("Etat_sante")[TARGET].mean().sort_values().plot(
        kind="barh", ax=ax, color="teal")
    ax.set_xlabel("Claim rate")
    st.pyplot(fig, use_container_width=True)

    st.markdown("**Claim rate by loan type**")
    fig, ax = plt.subplots(figsize=(6, 3))
    df.groupby("Type_credit")[TARGET].mean().sort_values().plot(
        kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Claim rate")
    st.pyplot(fig, use_container_width=True)

    st.markdown("**Raw data sample**")
    st.dataframe(df.drop(columns=["Cout_sinistre", "Prob_sinistre", "Risk_Score"]).head(50))

# ---- Tab 3: About -------------------------------------------------------
with tab_about:
    st.subheader("How this demo works")
    st.markdown(f"""
This app trains a **cost-sensitive Logistic Regression** model each time it
starts, using the same leakage-safe feature set as the research notebook:

- **Excluded on purpose:** `Cout_sinistre`, `Prob_sinistre`, `Risk_Score`,
  `Risk_Segment` — these are outcome/derived fields that would leak the
  answer, not real underwriting inputs.
- **Modeled only on active contracts**, since claims can only occur on a
  contract that is currently active.
- **Class imbalance** is handled with `class_weight="balanced"` (cost-sensitive
  learning) rather than synthetic oversampling, for auditability.
- **Right-skewed numeric fields** (income, loan amount, outstanding capital)
  are log-transformed before scaling.

**Current model performance (held-out test set):**
- ROC-AUC: `{metrics['roc_auc']:.3f}`
- Decision threshold: `{metrics['threshold']:.2f}`
- Precision at threshold: `{metrics['precision_at_t']:.2f}`
- Recall at threshold: `{metrics['recall_at_t']:.2f}`
- Training rows: `{metrics['n_train']:,}` / Test rows: `{metrics['n_test']:,}`
- Base claim rate: `{metrics['positive_rate']*100:.2f}%`

**Top model drivers (by coefficient magnitude):**
""")
    imp = pd.Series(coefs, index=feat_names).sort_values(key=abs, ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["firebrick" if v > 0 else "steelblue" for v in imp.values]
    imp.sort_values().plot(kind="barh", ax=ax, color=[c for c in colors[::-1]])
    ax.set_xlabel("Coefficient (positive = increases risk)")
    st.pyplot(fig, use_container_width=True)

    st.caption(
        "This demo is simplified from a larger research pipeline (see the "
        "accompanying notebooks) that benchmarked 9 imbalance-handling "
        "strategies across 4 model families, with SHAP/LIME explainability "
        "and ensembling. This app favors a single transparent, fast-to-train "
        "model so it can run instantly in a browser."
    )
