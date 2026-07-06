"""
ADI Maroc — Évaluation du risque de sinistre (Assurance Décès Emprunteur)
--------------------------------------------------------------------------
Application Streamlit interactive pour l'évaluation du risque de sinistre
sur un portefeuille de contrats ADI actifs.

Deux modèles disponibles :
  A — Coût-sensitif    : class_weight='balanced', C=1.0  (approche directe)
  B — Sous-échantillon : Random Undersampling + C=0.0094 (meilleur modèle notebook)
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score, balanced_accuracy_score,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

# ─────────────────────────────────────────────────────────────────────────────
# Configuration de la page
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ADI Maroc — Risque Sinistre",
    layout="wide",
)

st.markdown("""
<style>
h1 { color: #1a3a5c; }
h2 { color: #1a3a5c; }
h3 { color: #2c5f8a; }
.stMetric label { font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────
_CANDIDATES = [
    Path("data/dataset_Sinistre.csv"),
    Path("dataset_Sinistre.csv"),
    Path(__file__).parent / "dataset_Sinistre.csv",
    Path(__file__).parent / "data" / "dataset_Sinistre.csv",
]
DATA_PATH = next((str(p) for p in _CANDIDATES if p.exists()), "data/dataset_Sinistre.csv")

RANDOM_STATE = 42

NUM_COLS_RAW   = ["age_at_affiliation", "Anciennete", "Fumeur",
                  "Duree_credit", "Taux_interet", "DTI"]
LOG_COLS       = ["Revenu", "Montant_credit", "CRD"]
CAT_COLS       = ["Situation_familiale", "Sexe", "Etat_sante", "Profession", "Type_credit"]
NUM_COLS_MODEL = NUM_COLS_RAW + [f"{c}_log" for c in LOG_COLS]
TARGET         = "Sinistre"

LABELS_FR = {
    "age_at_affiliation": "Âge à l'affiliation",
    "Anciennete":          "Ancienneté (années)",
    "Fumeur":              "Fumeur",
    "Duree_credit":        "Durée du crédit (années)",
    "Taux_interet":        "Taux d'intérêt (%)",
    "DTI":                 "Ratio dette/revenu (DTI)",
    "Revenu_log":          "Revenu mensuel (log)",
    "Montant_credit_log":  "Montant du crédit (log)",
    "CRD_log":             "Capital restant dû (log)",
    "Situation_familiale": "Situation familiale",
    "Sexe":                "Sexe",
    "Etat_sante":          "État de santé",
    "Profession":          "Profession",
    "Type_credit":         "Type de crédit",
}

CAT_LABELS_FR = {
    "C": "Célibataire", "M": "Marié(e)",
    "F": "Femme",       "H": "Homme",
    "Bon": "Bon",       "Moyen": "Moyen",     "Mauvais": "Mauvais",
    "Cadre": "Cadre",   "Employé privé": "Employé privé",
    "Fonctionnaire": "Fonctionnaire",
    "Indépendant": "Indépendant",  "Ouvrier": "Ouvrier",
    "Consommation": "Consommation", "Immobilier": "Immobilier",
    "Professionnel": "Professionnel",
}

RISK_CONFIG = [
    (0.20, "Faible",     "#27ae60", "#eafaf1", "✅"),
    (0.40, "Modéré",     "#f39c12", "#fef9e7", "⚠️"),
    (0.60, "Élevé",      "#e67e22", "#fdf2e9", "🔶"),
    (1.01, "Très élevé", "#c0392b", "#fdedec", "🚨"),
]

MODEL_OPTIONS = {
    "Modèle A — Coût-sensitif": "A",
    "Modèle B — Sous-échantillonnage aléatoire": "B",
}

# ─────────────────────────────────────────────────────────────────────────────
# Chargement des données
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[df["Contrat_actif"] == 1].copy()
    for c in LOG_COLS:
        df[f"{c}_log"] = np.log1p(df[c])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Entraînement des deux modèles
# ─────────────────────────────────────────────────────────────────────────────
def _build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(),                                      NUM_COLS_MODEL),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"),  CAT_COLS),
    ])

def _tune_threshold_and_metrics(pipe, X_te, y_te, pos_rate, n_tr, n_te):
    proba_te = pipe.predict_proba(X_te)[:, 1]
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (proba_te >= t).astype(int)
        if recall_score(y_te, preds, zero_division=0) >= 0.5:
            f1 = f1_score(y_te, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
    preds_opt = (proba_te >= best_t).astype(int)
    fpr, tpr, _ = roc_curve(y_te, proba_te)
    prec_c, rec_c, _ = precision_recall_curve(y_te, proba_te)
    return {
        "roc_auc":   roc_auc_score(y_te, proba_te),
        "pr_auc":    average_precision_score(y_te, proba_te),
        "threshold": best_t,
        "precision": precision_score(y_te, preds_opt, zero_division=0),
        "recall":    recall_score(y_te, preds_opt, zero_division=0),
        "f1":        f1_score(y_te, preds_opt, zero_division=0),
        "bal_acc":   balanced_accuracy_score(y_te, preds_opt),
        "cm":        confusion_matrix(y_te, preds_opt, labels=[0, 1]),
        "fpr": fpr, "tpr": tpr, "prec_c": prec_c, "rec_c": rec_c,
        "n_train": n_tr, "n_test": n_te, "pos_rate": pos_rate,
    }

@st.cache_resource
def train_both_models():
    df = load_data()
    X  = df[NUM_COLS_MODEL + CAT_COLS].copy()
    y  = df[TARGET].copy()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    pos_rate = float(y.mean())

    # ── Modèle A : Coût-sensitif ──────────────────────────────────────────
    pipe_A = Pipeline([
        ("prep", _build_preprocessor()),
        ("clf",  LogisticRegression(
            max_iter=2000, class_weight="balanced",
            C=1.0, penalty="l2", random_state=RANDOM_STATE
        )),
    ])
    pipe_A.fit(X_tr, y_tr)

    # ── Modèle B : Random Undersampling + C optimisé (notebook) ──────────
    pipe_B = ImbPipeline([
        ("prep", _build_preprocessor()),
        ("rus",  RandomUnderSampler(random_state=RANDOM_STATE)),
        ("clf",  LogisticRegression(
            max_iter=3000, C=0.0094, penalty="l2",
            solver="liblinear", random_state=RANDOM_STATE
        )),
    ])
    pipe_B.fit(X_tr, y_tr)

    cat_enc    = pipe_A.named_steps["prep"].named_transformers_["cat"]
    feat_names = NUM_COLS_MODEL + list(cat_enc.get_feature_names_out(CAT_COLS))
    n_tr, n_te = len(X_tr), len(X_te)

    return {
        "A": {
            "pipe":    pipe_A,
            "metrics": _tune_threshold_and_metrics(pipe_A, X_te, y_te, pos_rate, n_tr, n_te),
            "feat_names": feat_names,
            "coefs":   pipe_A.named_steps["clf"].coef_[0],
        },
        "B": {
            "pipe":    pipe_B,
            "metrics": _tune_threshold_and_metrics(pipe_B, X_te, y_te, pos_rate, n_tr, n_te),
            "feat_names": feat_names,
            "coefs":   pipe_B.named_steps["clf"].coef_[0],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions graphiques utilitaires
# ─────────────────────────────────────────────────────────────────────────────
def get_risk_level(proba):
    for thresh, label, color, bg, icon in RISK_CONFIG:
        if proba < thresh:
            return label, color, bg, icon
    return "Très élevé", "#c0392b", "#fdedec", "🚨"


def draw_gauge(proba, threshold):
    """Jauge demi-cercle avec zones colorées et aiguille."""
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.set_aspect("equal")

    zone_bounds = [0.0, 0.20, 0.40, 0.60, 1.0]
    zone_colors = ["#27ae60", "#f1c40f", "#e67e22", "#c0392b"]
    r_out, r_in = 1.0, 0.55

    for i, col in enumerate(zone_colors):
        a0 = np.pi * (1.0 - zone_bounds[i])
        a1 = np.pi * (1.0 - zone_bounds[i + 1])
        th = np.linspace(a0, a1, 60)
        xs = np.concatenate([r_out * np.cos(th), (r_in * np.cos(th))[::-1]])
        ys = np.concatenate([r_out * np.sin(th), (r_in * np.sin(th))[::-1]])
        ax.fill(xs, ys, color=col, alpha=0.85, zorder=1)

    # Masque blanc en bas
    ax.fill([-1.15, 1.15, 1.15, -1.15], [-0.25, -0.25, 0.02, 0.02],
            color="white", zorder=2)

    # Seuil de décision (ligne noire pointillée)
    t_ang = np.pi * (1.0 - threshold)
    ax.plot([r_in * np.cos(t_ang), r_out * np.cos(t_ang)],
            [r_in * np.sin(t_ang), r_out * np.sin(t_ang)],
            color="black", lw=2, linestyle="--", zorder=3)
    ax.text(1.12 * np.cos(t_ang), 1.12 * np.sin(t_ang),
            f"Seuil\n{threshold*100:.0f}%",
            ha="center", va="bottom", fontsize=7, color="black", zorder=3)

    # Aiguille
    angle = np.pi * (1.0 - proba)
    ax.annotate("", xy=(0.80 * np.cos(angle), 0.80 * np.sin(angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#2c3e50",
                                lw=2.5, mutation_scale=14),
                zorder=5)
    ax.add_patch(plt.Circle((0, 0), 0.07, color="#2c3e50", zorder=6))

    # Libellés des zones
    for mid, lbl in [(0.10, "Faible"), (0.30, "Modéré"),
                     (0.50, "Élevé"), (0.80, "Très\nélevé")]:
        a = np.pi * (1.0 - mid)
        ax.text(1.22 * np.cos(a), 1.22 * np.sin(a), lbl,
                ha="center", va="center", fontsize=7, color="#555", zorder=7)

    # Valeur centrale
    label, color, _, _ = get_risk_level(proba)
    ax.text(0, -0.15, f"{proba * 100:.1f}%",
            ha="center", va="top", fontsize=22,
            fontweight="bold", color=color, zorder=7)

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-0.35, 1.3)
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    plt.tight_layout(pad=0)
    return fig


def contribution_chart(pipe, row_df):
    """Barres horizontales : contribution de chaque variable à la prédiction."""
    prep    = pipe.named_steps["prep"]
    clf     = pipe.named_steps["clf"]
    X_t     = prep.transform(row_df)
    cat_enc = prep.named_transformers_["cat"]
    all_feat = NUM_COLS_MODEL + list(cat_enc.get_feature_names_out(CAT_COLS))
    raw_contrib = pd.Series(X_t[0] * clf.coef_[0], index=all_feat)

    # Regrouper les dummies par variable parent
    grouped = raw_contrib[NUM_COLS_MODEL].copy()
    for raw_col in CAT_COLS:
        dummies = [f for f in all_feat if f.startswith(f"{raw_col}_")]
        if dummies:
            grouped[raw_col] = raw_contrib[dummies].sum()

    grouped.index = [LABELS_FR.get(i, i) for i in grouped.index]
    grouped = grouped.reindex(grouped.abs().sort_values(ascending=False).index).head(10)

    colors = ["#e74c3c" if v > 0 else "#2980b9" for v in grouped.values]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    grouped.sort_values().plot(kind="barh", ax=ax, color=colors[::-1], edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Contribution à la log-cote")
    ax.set_title("Facteurs de risque — ce profil", fontweight="bold", pad=8)

    for bar, val in zip(ax.patches, grouped.sort_values().values):
        ax.text(val + (0.005 if val >= 0 else -0.005), bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha="left" if val >= 0 else "right", fontsize=8)

    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    return fig


def distribution_chart(df, value, col, xlabel):
    """Histogramme du portefeuille avec position du profil courant."""
    fig, ax = plt.subplots(figsize=(5, 2.8))
    bins = np.histogram_bin_edges(df[col].dropna(), bins=22)
    ax.hist(df[df[TARGET] == 0][col], bins=bins, alpha=0.55,
            color="#3498db", label="Sans sinistre", density=True)
    ax.hist(df[df[TARGET] == 1][col], bins=bins, alpha=0.75,
            color="#e74c3c", label="Avec sinistre",  density=True)
    ax.axvline(value, color="#2c3e50", lw=2, linestyle="--", label="Ce profil")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Densité")
    ax.legend(fontsize=8)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chargement des données + entraînement
# ─────────────────────────────────────────────────────────────────────────────
try:
    df      = load_data()
    models  = train_both_models()
    data_ok = True
except FileNotFoundError:
    data_ok = False

# ─────────────────────────────────────────────────────────────────────────────
# En-tête
# ─────────────────────────────────────────────────────────────────────────────
st.title("🛡️ ADI Maroc — Évaluation du Risque de Sinistre")
st.markdown(
    "Outil d'aide à la décision pour l'évaluation du risque de sinistre sur un "
    "portefeuille de contrats **Assurance Décès Emprunteur (ADI)**. "
    "Le modèle est entraîné exclusivement sur les contrats actifs."
)

if not data_ok:
    st.error(
        "⚠️ Fichier de données introuvable (`dataset_Sinistre.csv`). "
        "Placez le fichier dans le même répertoire que `app.py` ou dans un sous-dossier `data/`."
    )
    st.stop()

# ── Sélecteur de modèle (barre latérale) ─────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Sélection du modèle")
    selected_label = st.selectbox(
        "Modèle actif",
        list(MODEL_OPTIONS.keys()),
        help="Choisissez le modèle à utiliser pour la prédiction et l'analyse de performance.",
    )
    model_key = MODEL_OPTIONS[selected_label]

    st.divider()
    st.header("ℹ️ Informations modèles")

    st.markdown("### Modèle A — Coût-sensitif")
    st.markdown("""
`class_weight='balanced'` · C=1.0 · penalty=l2 · solver=lbfgs

**Principe :** Le jeu d'entraînement reste intact (tous les contrats).
Le modèle attribue automatiquement à chaque classe un **poids inversement
proportionnel** à sa fréquence — les sinistres (≈2,7 %) reçoivent un poids
≈ 36× plus élevé que les contrats sains. Cela revient à « pénaliser »
davantage les erreurs sur les sinistres dans la fonction de coût.

**Avantages :**
- Simple à mettre en œuvre, aucune modification des données
- Entraîné sur le volume complet → estimations de probabilité plus stables
- Interprétation directe via les coefficients
- Déploiement standard (sklearn Pipeline)

**Limites :**
- La pondération reste une approximation ; le modèle peut sous-estimer
  certains sinistres atypiques
- C=1.0 non optimisé (valeur par défaut de sklearn)
""")

    st.markdown("### Modèle B — Sous-échantillonnage aléatoire")
    st.markdown("""
`RandomUnderSampler` · C=0.0094 · penalty=l2 · solver=liblinear

**Principe :** Avant l'entraînement, les contrats sains sont **réduits
aléatoirement** jusqu'à égalité numérique avec les sinistres. Le modèle
apprend ainsi sur un jeu équilibré (~2× 78 observations). Le paramètre C
a été optimisé via **Optuna** (recherche bayésienne) sur PR-AUC.

**Avantages :**
- Équilibre parfait des classes pendant l'entraînement
- C fortement régularisé (0.0094 ≈ 100× plus fort que A) → coefficients
  plus stables sur un petit jeu d'entraînement
- Meilleur rappel : détecte plus de vrais sinistres

**Limites :**
- Jeu d'entraînement très réduit → variance élevée entre runs
- Élimine une grande partie de l'information sur les contrats sains
- Probabilités calibrées différemment (distribution interne déséquilibrée)
""")

    st.divider()
    st.markdown("### ⚖️ Différences clés")
    st.markdown("""
| Aspect | Modèle A | Modèle B |
|---|---|---|
| **Équilibrage** | Poids de coût | Suppression physique |
| **Données train** | Toutes | Sous-ensemble |
| **C (régul.)** | 1.0 | 0.0094 |
| **Solver** | lbfgs | liblinear |
| **Optimisation** | Défaut | Optuna (PR-AUC) |
| **Rappel** | Modéré | Plus élevé |
| **Faux positifs** | Moins | Plus |

> Choisissez **A** pour une estimation de probabilité plus stable.
> Choisissez **B** pour maximiser la détection des sinistres (moins de manqués).
""")

    if data_ok:
        mA, mB = models["A"]["metrics"], models["B"]["metrics"]
        st.divider()
        st.markdown("### 📊 Métriques rapides")
        st.markdown(f"""
| Métrique | A | B |
|---|---|---|
| ROC-AUC | {mA['roc_auc']:.3f} | {mB['roc_auc']:.3f} |
| PR-AUC | {mA['pr_auc']:.3f} | {mB['pr_auc']:.3f} |
| Rappel | {mA['recall']:.3f} | {mB['recall']:.3f} |
| Précision | {mA['precision']:.3f} | {mB['precision']:.3f} |
| F1 | {mA['f1']:.3f} | {mB['f1']:.3f} |
| Seuil | {mA['threshold']:.2f} | {mB['threshold']:.2f} |
""")

# Raccourcis vers le modèle sélectionné
pipe       = models[model_key]["pipe"]
metrics    = models[model_key]["metrics"]
feat_names = models[model_key]["feat_names"]
coefs      = models[model_key]["coefs"]

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Onglets
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Prédiction",
    "📊 Exploration du portefeuille",
    "📈 Performance du modèle",
    "📖 Méthodologie",
])

# ══════════════════════════════════════════════════════════════════════════════
# Onglet 1 — Prédiction
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Saisie du profil assuré")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("##### 👤 Profil personnel")
            age        = st.slider("Âge à l'affiliation (ans)", 18, 70, 35)
            anciennete = st.slider("Ancienneté avec l'assureur (années)", 0.0, 30.0, 5.0, step=0.5)
            fumeur     = st.radio("Fumeur", ["Non", "Oui"], horizontal=True)
            sexe       = st.radio("Sexe", ["Femme", "Homme"], horizontal=True)
            situation  = st.radio("Situation familiale", ["Célibataire", "Marié(e)"], horizontal=True)
            etat_sante = st.radio("État de santé", ["Bon", "Moyen", "Mauvais"], horizontal=True)

        with c2:
            st.markdown("##### 💳 Caractéristiques du crédit")
            duree_credit = st.slider("Durée du crédit (années)", 1, 30, 15)
            taux         = st.slider("Taux d'intérêt (%)", 1.0, 15.0, 5.5, step=0.1)
            dti          = st.slider("Ratio dette/revenu (DTI)", 0.0, 1.0, 0.33, step=0.01)
            type_credit  = st.radio("Type de crédit",
                                    ["Consommation", "Immobilier", "Professionnel"],
                                    horizontal=True)
            profession   = st.selectbox("Profession",
                                        ["Cadre", "Employé privé", "Fonctionnaire",
                                         "Indépendant", "Ouvrier"])

        with c3:
            st.markdown("##### 💰 Données financières")
            revenu  = st.number_input("Revenu mensuel (MAD)",    1_000,   100_000,  8_000, step=500)
            montant = st.number_input("Montant du crédit (MAD)", 5_000, 2_000_000, 300_000, step=5_000)
            crd     = st.number_input("Capital restant dû — CRD (MAD)", 0, 2_000_000, 100_000, step=1_000)

        submitted = st.form_submit_button(
            "🔍  Calculer le risque de sinistre", type="primary", use_container_width=True
        )

    # Persist inputs in session_state so a model switch re-computes without re-clicking
    if submitted:
        st.session_state["last_inputs"] = {
            "age": age, "anciennete": anciennete, "fumeur": fumeur,
            "sexe": sexe, "situation": situation, "etat_sante": etat_sante,
            "duree_credit": duree_credit, "taux": taux, "dti": dti,
            "type_credit": type_credit, "profession": profession,
            "revenu": revenu, "montant": montant, "crd": crd,
        }

    # ── Résultat ─────────────────────────────────────────────────────────────
    # Show result whenever inputs are stored (survives model switches)
    if "last_inputs" in st.session_state:
        inp = st.session_state["last_inputs"]
        age          = inp["age"];          anciennete  = inp["anciennete"]
        fumeur       = inp["fumeur"];       sexe        = inp["sexe"]
        situation    = inp["situation"];    etat_sante  = inp["etat_sante"]
        duree_credit = inp["duree_credit"]; taux        = inp["taux"]
        dti          = inp["dti"];          type_credit = inp["type_credit"]
        profession   = inp["profession"];   revenu      = inp["revenu"]
        montant      = inp["montant"];      crd         = inp["crd"]

        sexe_val      = "F" if sexe == "Femme" else "H"
        situation_val = "C" if situation == "Célibataire" else "M"

        row = pd.DataFrame([{
            "age_at_affiliation":  age,
            "Anciennete":          anciennete,
            "Fumeur":              1 if fumeur == "Oui" else 0,
            "Duree_credit":        duree_credit,
            "Taux_interet":        taux,
            "DTI":                 dti,
            "Revenu_log":          np.log1p(revenu),
            "Montant_credit_log":  np.log1p(montant),
            "CRD_log":             np.log1p(crd),
            "Situation_familiale": situation_val,
            "Sexe":                sexe_val,
            "Etat_sante":          etat_sante,
            "Profession":          profession,
            "Type_credit":         type_credit,
        }])[NUM_COLS_MODEL + CAT_COLS]

        proba   = pipe.predict_proba(row)[0, 1]
        flagged = proba >= metrics["threshold"]
        label, color, bg, icon = get_risk_level(proba)

        st.divider()
        st.subheader("📋 Résultat de l'évaluation")

        g_col, info_col = st.columns([1, 2])

        with g_col:
            st.markdown("**Jauge de risque**")
            st.pyplot(draw_gauge(proba, metrics["threshold"]), use_container_width=True)

        with info_col:
            st.markdown(
                f"""<div style='background:{bg};border-left:5px solid {color};
                padding:1.1rem 1.4rem;border-radius:8px;margin-bottom:1rem;'>
                <div style='font-size:1.7rem;font-weight:bold;color:{color};'>
                {icon} Risque {label}</div>
                <div style='font-size:1rem;color:#444;margin-top:4px;'>
                Probabilité estimée : <strong>{proba*100:.1f}%</strong>
                &nbsp;|&nbsp;
                Seuil de décision : <strong>{metrics['threshold']*100:.0f}%</strong>
                </div></div>""",
                unsafe_allow_html=True,
            )

            if flagged:
                st.error(
                    f"**⚠️ Contrat signalé pour révision manuelle.**  \n"
                    f"La probabilité de sinistre ({proba*100:.1f}%) dépasse le seuil "
                    f"de déclenchement ({metrics['threshold']*100:.0f}%). "
                    "Une analyse approfondie du dossier est recommandée avant acceptation."
                )
            else:
                st.success(
                    f"**✅ Risque dans les limites acceptables.**  \n"
                    f"La probabilité de sinistre ({proba*100:.1f}%) est en dessous du seuil "
                    f"({metrics['threshold']*100:.0f}%). "
                    "Le contrat peut être traité selon la politique de souscription standard."
                )

            base  = metrics["pos_rate"]
            ratio = proba / base if base > 0 else 1.0
            direction = "supérieur" if ratio > 1 else "inférieur"
            st.info(
                f"ℹ️ **Comparaison portefeuille :** taux de sinistre moyen du portefeuille = "
                f"**{base*100:.2f}%**.  \n"
                f"Ce profil présente un risque estimé **{ratio:.1f}× {direction}** "
                "à la moyenne du portefeuille actif."
            )

        st.markdown("---")
        contrib_col, pos_col = st.columns(2)

        with contrib_col:
            st.markdown("**Principaux facteurs influençant la prédiction**")
            st.pyplot(contribution_chart(pipe, row), use_container_width=True)
            st.caption(
                "🔴 **Barres rouges** : facteurs qui augmentent le risque estimé.  \n"
                "🔵 **Barres bleues** : facteurs qui réduisent le risque estimé.  \n"
                "Les contributions correspondent aux coefficients de la régression "
                "logistique appliqués aux valeurs normalisées de ce profil."
            )

        with pos_col:
            st.markdown("**Position du profil — âge à l'affiliation**")
            st.pyplot(
                distribution_chart(df, age, "age_at_affiliation", "Âge à l'affiliation (ans)"),
                use_container_width=True,
            )
            st.caption(
                "Distribution des âges dans le portefeuille actif, séparée selon le statut "
                "de sinistre. La ligne pointillée indique la valeur du profil soumis.  \n"
                f"Moyenne des sinistres : **{df[df[TARGET]==1]['age_at_affiliation'].mean():.1f} ans** "
                f"— Moyenne sans sinistre : **{df[df[TARGET]==0]['age_at_affiliation'].mean():.1f} ans**."
            )

        st.markdown("---")
        dti_col, dur_col = st.columns(2)

        with dti_col:
            st.markdown("**Position du profil — ratio dette/revenu (DTI)**")
            st.pyplot(
                distribution_chart(df, dti, "DTI", "Ratio dette/revenu"),
                use_container_width=True,
            )
            dti_mean_c  = df[df[TARGET] == 1]["DTI"].mean()
            dti_mean_nc = df[df[TARGET] == 0]["DTI"].mean()
            st.caption(
                f"DTI moyen des sinistres : **{dti_mean_c:.3f}** — "
                f"sans sinistre : **{dti_mean_nc:.3f}**.  \n"
                "Un DTI élevé traduit une forte charge d'endettement relative au revenu."
            )

        with dur_col:
            st.markdown("**Position du profil — durée du crédit**")
            st.pyplot(
                distribution_chart(df, duree_credit, "Duree_credit", "Durée du crédit (années)"),
                use_container_width=True,
            )
            dur_mean_c  = df[df[TARGET] == 1]["Duree_credit"].mean()
            dur_mean_nc = df[df[TARGET] == 0]["Duree_credit"].mean()
            st.caption(
                f"Durée moyenne des sinistres : **{dur_mean_c:.1f} ans** — "
                f"sans sinistre : **{dur_mean_nc:.1f} ans**.  \n"
                "Des crédits plus longs augmentent la période d'exposition au risque de décès."
            )

        st.markdown("---")
        st.subheader("🔍 Interprétation actuarielle")

        ic1, ic2, ic3 = st.columns(3)

        with ic1:
            st.markdown("**Profil démographique**")
            age_mean_c = df[df[TARGET] == 1]["age_at_affiliation"].mean()
            age_cmp    = "supérieur" if age > age_mean_c else "inférieur"
            st.markdown(
                f"- Âge à l'affiliation ({age} ans) : "
                f"**{age_cmp}** à la moyenne des sinistres ({age_mean_c:.0f} ans)"
            )
            fum_r     = df[df["Fumeur"] == 1][TARGET].mean()
            non_fum_r = df[df["Fumeur"] == 0][TARGET].mean()
            if fumeur == "Oui":
                st.markdown(
                    f"- Fumeur : taux de sinistre = **{fum_r*100:.1f}%** "
                    f"(non-fumeurs : {non_fum_r*100:.1f}%)"
                )
            else:
                st.markdown(
                    f"- Non-fumeur : taux = **{non_fum_r*100:.1f}%** "
                    f"(fumeurs : {fum_r*100:.1f}%)"
                )
            sit_r = df[df["Situation_familiale"] == situation_val][TARGET].mean()
            st.markdown(
                f"- Situation familiale « {CAT_LABELS_FR.get(situation_val, situation_val)} » : "
                f"taux = **{sit_r*100:.1f}%**"
            )

        with ic2:
            st.markdown("**Profil financier**")
            dti_cmp = "supérieur" if dti > dti_mean_c else "inférieur"
            st.markdown(
                f"- DTI ({dti:.2f}) : **{dti_cmp}** à la moyenne "
                f"des sinistres ({dti_mean_c:.2f})"
            )
            dur_cmp = "supérieure" if duree_credit > dur_mean_c else "inférieure"
            st.markdown(
                f"- Durée du crédit ({duree_credit} ans) : "
                f"**{dur_cmp}** à la moyenne ({dur_mean_c:.0f} ans)"
            )
            rev_q = (df["Revenu"] <= revenu).mean()
            st.markdown(
                f"- Revenu mensuel ({revenu:,.0f} MAD) : "
                f"supérieur à **{rev_q*100:.0f}%** du portefeuille"
            )

        with ic3:
            st.markdown("**Profil assurantiel**")
            h_r = df[df["Etat_sante"] == etat_sante][TARGET].mean()
            st.markdown(
                f"- État de santé « {etat_sante} » : "
                f"taux de sinistre = **{h_r*100:.1f}%**"
            )
            p_r = df[df["Profession"] == profession][TARGET].mean()
            st.markdown(
                f"- Profession « {profession} » : "
                f"taux = **{p_r*100:.1f}%**"
            )
            cr_r = df[df["Type_credit"] == type_credit][TARGET].mean()
            st.markdown(
                f"- Crédit « {type_credit} » : "
                f"taux = **{cr_r*100:.1f}%**"
            )

        st.caption(
            "⚠️ **Avertissement :** cet outil est une aide à la décision, non un moteur "
            "automatique d'acceptation ou de refus. Les décisions finales de souscription "
            "doivent intégrer le jugement d'un actuaire ou d'un expert métier."
        )


# ══════════════════════════════════════════════════════════════════════════════
# Onglet 2 — Exploration du portefeuille
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Aperçu du portefeuille — contrats actifs")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Contrats actifs",        f"{len(df):,}")
    k2.metric("Sinistres déclarés",     f"{int(df[TARGET].sum()):,}")
    k3.metric("Taux de sinistre",       f"{df[TARGET].mean()*100:.2f}%")
    k4.metric("Prime mensuelle moy.",   f"{df['Prime_mensuelle'].mean():,.0f} MAD")

    st.divider()

    e1, e2, e3 = st.columns(3)

    with e1:
        st.markdown("**Par état de santé**")
        rates = df.groupby("Etat_sante")[TARGET].mean()
        clrs  = ["#27ae60" if r < 0.03 else "#e67e22" if r < 0.055 else "#e74c3c"
                 for r in rates.sort_values().values]
        fig, ax = plt.subplots(figsize=(5, 2.8))
        rates.sort_values().plot(kind="barh", ax=ax, color=clrs, edgecolor="white")
        ax.set_xlabel("Taux de sinistre")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.1f}%"))
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption(
            "Un état de santé dégradé est associé à un taux de sinistre plus élevé, "
            "ce qui reflète le lien direct entre la santé de l'assuré et le risque de décès."
        )

    with e2:
        st.markdown("**Par type de crédit**")
        rates = df.groupby("Type_credit")[TARGET].mean()
        fig, ax = plt.subplots(figsize=(5, 2.8))
        rates.sort_values().plot(kind="barh", ax=ax,
                                 color=["#2980b9", "#3498db", "#5dade2"],
                                 edgecolor="white")
        ax.set_xlabel("Taux de sinistre")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.1f}%"))
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption(
            "Les crédits immobiliers, souvent de plus longue durée, peuvent présenter "
            "un profil de risque distinct des crédits à la consommation ou professionnels."
        )

    with e3:
        st.markdown("**Par profession**")
        rates = df.groupby("Profession")[TARGET].mean()
        fig, ax = plt.subplots(figsize=(5, 2.8))
        rates.sort_values().plot(kind="barh", ax=ax, color="#8e44ad", edgecolor="white")
        ax.set_xlabel("Taux de sinistre")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.1f}%"))
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption(
            "La catégorie socio-professionnelle influence le risque via les conditions "
            "de travail, la stabilité de l'emploi et les expositions spécifiques."
        )

    st.divider()

    d1, d2 = st.columns(2)

    with d1:
        st.markdown("**Distribution de l'âge à l'affiliation**")
        bins = np.linspace(df["age_at_affiliation"].min(),
                           df["age_at_affiliation"].max(), 25)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(df[df[TARGET] == 0]["age_at_affiliation"], bins=bins, alpha=0.55,
                color="#3498db", label="Sans sinistre", density=True)
        ax.hist(df[df[TARGET] == 1]["age_at_affiliation"], bins=bins, alpha=0.80,
                color="#e74c3c", label="Avec sinistre",  density=True)
        ax.set_xlabel("Âge à l'affiliation (ans)")
        ax.set_ylabel("Densité")
        ax.legend()
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        mean_c  = df[df[TARGET] == 1]["age_at_affiliation"].mean()
        mean_nc = df[df[TARGET] == 0]["age_at_affiliation"].mean()
        st.caption(
            f"Âge moyen à l'affiliation : **{mean_c:.1f} ans** (sinistres) "
            f"vs **{mean_nc:.1f} ans** (sans sinistre).  \n"
            "Les assurés plus âgés à l'affiliation présentent une probabilité de décès "
            "plus élevée, ce qui se traduit par un taux de sinistre supérieur."
        )

    with d2:
        st.markdown("**Distribution du ratio dette/revenu (DTI)**")
        bins = np.linspace(0, df["DTI"].quantile(0.99), 25)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(df[df[TARGET] == 0]["DTI"], bins=bins, alpha=0.55,
                color="#3498db", label="Sans sinistre", density=True)
        ax.hist(df[df[TARGET] == 1]["DTI"], bins=bins, alpha=0.80,
                color="#e74c3c", label="Avec sinistre",  density=True)
        ax.set_xlabel("DTI")
        ax.set_ylabel("Densité")
        ax.legend()
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        dti_c  = df[df[TARGET] == 1]["DTI"].mean()
        dti_nc = df[df[TARGET] == 0]["DTI"].mean()
        st.caption(
            f"DTI moyen : **{dti_c:.3f}** (sinistres) vs **{dti_nc:.3f}** (sans sinistre).  \n"
            "Un ratio dette/revenu élevé peut indiquer une fragilité financière accrue, "
            "corrélée avec un risque de sinistre plus important."
        )

    st.divider()

    b1, b2 = st.columns(2)

    with b1:
        st.markdown("**Durée du crédit selon le statut sinistre**")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        data_bp = [
            df[df[TARGET] == 0]["Duree_credit"].dropna().values,
            df[df[TARGET] == 1]["Duree_credit"].dropna().values,
        ]
        bp = ax.boxplot(data_bp, patch_artist=True,
                        labels=["Sans sinistre", "Avec sinistre"])
        for patch, col in zip(bp["boxes"], ["#3498db", "#e74c3c"]):
            patch.set_facecolor(col); patch.set_alpha(0.6)
        ax.set_ylabel("Durée du crédit (années)")
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        dur_c  = df[df[TARGET] == 1]["Duree_credit"].mean()
        dur_nc = df[df[TARGET] == 0]["Duree_credit"].mean()
        st.caption(
            f"Durée moyenne : **{dur_c:.1f} ans** (sinistres) "
            f"vs **{dur_nc:.1f} ans** (sans sinistre).  \n"
            "La durée du crédit est l'une des variables les plus importantes du modèle : "
            "un crédit plus long expose l'assuré à un risque de décès sur une période plus étendue."
        )

    with b2:
        st.markdown("**Taux de sinistre par situation familiale**")
        counts = df.groupby("Situation_familiale")[TARGET].agg(["sum", "count"])
        counts["rate"] = counts["sum"] / counts["count"]
        mapping = {"C": "Célibataire", "M": "Marié(e)"}
        counts.index = counts.index.map(lambda x: mapping.get(x, x))
        fig, ax = plt.subplots(figsize=(5, 3.5))
        counts["rate"].sort_values().plot(kind="barh", ax=ax,
                                          color="#1abc9c", edgecolor="white")
        ax.set_xlabel("Taux de sinistre")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.1f}%"))
        for idx, row_data in counts.sort_values("rate").iterrows():
            ax.text(row_data["rate"] + 0.001,
                    list(counts.sort_values("rate").index).index(idx),
                    f"n={int(row_data['count'])}", va="center", fontsize=9)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption(
            "La situation familiale peut être corrélée avec des facteurs socio-économiques "
            "(revenus, patrimoine, mode de vie) qui influencent indirectement le risque de sinistre."
        )

    st.divider()
    st.markdown("**Extrait des données brutes (50 premières lignes)**")
    drop_cols = [c for c in ["Cout_sinistre", "Prob_sinistre",
                              "Risk_Score", "Risk_Segment"] if c in df.columns]
    st.dataframe(df.drop(columns=drop_cols).head(50), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Onglet 3 — Performance du modèle
# ══════════════════════════════════════════════════════════════════════════════
MODEL_DESC = {
    "A": "Régression Logistique · class_weight='balanced' · C=1.0 · penalty=l2",
    "B": "Régression Logistique · Random Undersampling · C=0.0094 · solver=liblinear",
}

with tab3:
    st.subheader("Performance du modèle sur l'ensemble de test")
    st.caption(f"Modèle actif : **{selected_label}** — {MODEL_DESC[model_key]}")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("ROC-AUC",              f"{metrics['roc_auc']:.3f}",
              help="Aire sous la courbe ROC — capacité discriminante globale")
    m2.metric("PR-AUC",               f"{metrics['pr_auc']:.3f}",
              help="Aire sous la courbe Précision-Rappel — plus pertinent en cas de déséquilibre")
    m3.metric("Rappel (Sensibilité)",  f"{metrics['recall']:.3f}",
              help="Fraction des vrais sinistres correctement détectés")
    m4.metric("Précision",            f"{metrics['precision']:.3f}",
              help="Fraction des alertes qui correspondent à de vrais sinistres")
    m5.metric("Précision équilibrée", f"{metrics['bal_acc']:.3f}",
              help="Moyenne du taux de bonne classification par classe")

    st.info(
        f"**Seuil de décision :** {metrics['threshold']:.2f} "
        f"(optimisé pour maximiser le F1 avec rappel ≥ 50%)  \n"
        f"Entraînement : **{metrics['n_train']:,}** contrats — "
        f"Test : **{metrics['n_test']:,}** contrats — "
        f"Taux de sinistre base : **{metrics['pos_rate']*100:.2f}%**"
    )

    st.divider()

    r1, r2, r3 = st.columns(3)

    with r1:
        st.markdown("**Courbe ROC**")
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(metrics["fpr"], metrics["tpr"], color="#2980b9", lw=2,
                label=f"Modèle (AUC = {metrics['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aléatoire (AUC = 0.5)")
        ax.fill_between(metrics["fpr"], metrics["tpr"], alpha=0.08, color="#2980b9")
        ax.set_xlabel("Taux de faux positifs (1 − Spécificité)")
        ax.set_ylabel("Taux de vrais positifs (Rappel)")
        ax.set_title("Courbe ROC", fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption(
            "La courbe ROC mesure la capacité discriminante du modèle indépendamment "
            f"du seuil. Un AUC de **{metrics['roc_auc']:.3f}** indique une bonne "
            "séparation entre contrats à risque et contrats sains."
        )

    with r2:
        st.markdown("**Courbe Précision-Rappel**")
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(metrics["rec_c"], metrics["prec_c"], color="#8e44ad", lw=2,
                label=f"Modèle (PR-AUC = {metrics['pr_auc']:.3f})")
        ax.axhline(metrics["pos_rate"], color="gray", ls="--", lw=1,
                   label=f"Baseline ({metrics['pos_rate']*100:.1f}%)")
        ax.fill_between(metrics["rec_c"], metrics["prec_c"],
                        metrics["pos_rate"], alpha=0.08, color="#8e44ad")
        ax.set_xlabel("Rappel")
        ax.set_ylabel("Précision")
        ax.set_title("Courbe Précision-Rappel", fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption(
            "Plus informative que la ROC pour les jeux déséquilibrés "
            f"(taux de sinistre : {metrics['pos_rate']*100:.2f}%). "
            "La zone colorée représente le gain du modèle par rapport à la prédiction naïve."
        )

    with r3:
        st.markdown("**Matrice de confusion**")
        cm = metrics["cm"]
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Prédit : Sain", "Prédit : Sinistre"])
        ax.set_yticklabels(["Réel : Sain", "Réel : Sinistre"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=18, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_title("Matrice de confusion", fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption(
            f"VN={cm[0,0]} · FP={cm[0,1]} · **FN={cm[1,0]}** · VP={cm[1,1]}.  \n"
            "Les **faux négatifs** (sinistres non détectés) représentent le risque "
            "prioritaire à minimiser dans ce contexte : un sinistre manqué a un coût "
            "actuariel bien supérieur à une fausse alerte."
        )

    st.divider()

    st.markdown("**Importance des variables (valeur absolue des coefficients)**")
    imp = pd.Series(coefs, index=feat_names)
    agg_imp = {}
    for c in NUM_COLS_MODEL:
        agg_imp[LABELS_FR.get(c, c)] = abs(imp[c])
    for raw_col in CAT_COLS:
        dummies = [f for f in feat_names if f.startswith(f"{raw_col}_")]
        if dummies:
            agg_imp[LABELS_FR.get(raw_col, raw_col)] = imp[dummies].abs().max()
    imp_s = pd.Series(agg_imp).sort_values(ascending=True).tail(12)

    fig, ax = plt.subplots(figsize=(8, 5))
    median_v = imp_s.median()
    colors_i = ["#e74c3c" if v >= median_v else "#3498db" for v in imp_s.values]
    imp_s.plot(kind="barh", ax=ax, color=colors_i, edgecolor="white")
    ax.axvline(median_v, color="gray", lw=1, ls="--", alpha=0.7,
               label=f"Médiane = {median_v:.3f}")
    ax.set_xlabel("Importance (|coefficient normalisé|)")
    ax.set_title("Variables les plus influentes dans le modèle", fontweight="bold")
    ax.legend(fontsize=9)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.caption(
        "L'importance est mesurée par la valeur absolue du coefficient de la régression "
        "logistique sur données normalisées — elle reflète la sensibilité du modèle à "
        "chaque variable. Les barres **rouges** dépassent la médiane d'importance.  \n"
        "La durée du crédit et le capital restant dû (CRD) sont généralement les "
        "facteurs dominants, car ils captent à la fois la durée d'exposition et "
        "l'engagement financier résiduel."
    )

    # ── Comparaison côte-à-côte des deux modèles ─────────────────────────
    st.divider()
    st.subheader("Comparaison Modèle A vs Modèle B")
    st.caption("Même ensemble de test pour les deux modèles — les différences reflètent uniquement la stratégie d'entraînement.")

    mA, mB = models["A"]["metrics"], models["B"]["metrics"]

    compare_df = pd.DataFrame({
        "Métrique": ["ROC-AUC", "PR-AUC", "Rappel", "Précision", "F1", "Précision équil.", "Seuil"],
        "Modèle A (Coût-sensitif)": [
            f"{mA['roc_auc']:.3f}", f"{mA['pr_auc']:.3f}",
            f"{mA['recall']:.3f}",  f"{mA['precision']:.3f}",
            f"{mA['f1']:.3f}",      f"{mA['bal_acc']:.3f}",
            f"{mA['threshold']:.2f}",
        ],
        "Modèle B (Sous-échantillonnage)": [
            f"{mB['roc_auc']:.3f}", f"{mB['pr_auc']:.3f}",
            f"{mB['recall']:.3f}",  f"{mB['precision']:.3f}",
            f"{mB['f1']:.3f}",      f"{mB['bal_acc']:.3f}",
            f"{mB['threshold']:.2f}",
        ],
    }).set_index("Métrique")
    st.dataframe(compare_df, use_container_width=True)

    cmp1, cmp2, cmp3 = st.columns(3)

    with cmp1:
        st.markdown("**Courbes ROC superposées**")
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(mA["fpr"], mA["tpr"], color="#2980b9", lw=2,
                label=f"A — AUC={mA['roc_auc']:.3f}")
        ax.plot(mB["fpr"], mB["tpr"], color="#e67e22", lw=2, linestyle="--",
                label=f"B — AUC={mB['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.5)
        ax.set_xlabel("Taux de faux positifs")
        ax.set_ylabel("Rappel")
        ax.legend(fontsize=9)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with cmp2:
        st.markdown("**Courbes PR superposées**")
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(mA["rec_c"], mA["prec_c"], color="#2980b9", lw=2,
                label=f"A — PR-AUC={mA['pr_auc']:.3f}")
        ax.plot(mB["rec_c"], mB["prec_c"], color="#e67e22", lw=2, linestyle="--",
                label=f"B — PR-AUC={mB['pr_auc']:.3f}")
        ax.axhline(mA["pos_rate"], color="gray", ls=":", lw=1, alpha=0.7,
                   label=f"Baseline ({mA['pos_rate']*100:.1f}%)")
        ax.set_xlabel("Rappel")
        ax.set_ylabel("Précision")
        ax.legend(fontsize=9)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with cmp3:
        st.markdown("**Métriques clés côte-à-côte**")
        keys   = ["roc_auc", "pr_auc", "recall", "precision", "f1", "bal_acc"]
        labels = ["ROC-AUC", "PR-AUC", "Rappel", "Précision", "F1", "Préc. équil."]
        vals_A = [mA[k] for k in keys]
        vals_B = [mB[k] for k in keys]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(4.5, 4))
        w = 0.35
        ax.barh(x + w/2, vals_A, w, color="#2980b9", label="A", alpha=0.85)
        ax.barh(x - w/2, vals_B, w, color="#e67e22", label="B", alpha=0.85)
        ax.set_yticks(x)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Score")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    st.markdown("**Matrices de confusion — Modèle A vs Modèle B**")
    cm_col_A, cm_col_B = st.columns(2)

    for col, model_lbl, cm_data, cmap in [
        (cm_col_A, "Modèle A — Coût-sensitif",        mA["cm"], "Blues"),
        (cm_col_B, "Modèle B — Sous-échantillonnage", mB["cm"], "Oranges"),
    ]:
        with col:
            st.markdown(f"**{model_lbl}**")
            fig, ax = plt.subplots(figsize=(4, 3.5))
            im = ax.imshow(cm_data, cmap=cmap, interpolation="nearest")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["Prédit : Sain", "Prédit : Sinistre"])
            ax.set_yticklabels(["Réel : Sain", "Réel : Sinistre"])
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm_data[i, j]), ha="center", va="center",
                            fontsize=16, fontweight="bold",
                            color="white" if cm_data[i, j] > cm_data.max() / 2 else "black")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            st.caption(
                f"VN={cm_data[0,0]} · FP={cm_data[0,1]} · "
                f"**FN={cm_data[1,0]}** · VP={cm_data[1,1]}"
            )

    st.info(
        "💡 **Lecture :** Comparez les **Faux Négatifs (FN)** — sinistres manqués. "
        "Le modèle B (sous-échantillonnage, C=0.0094) est sélectionné dans le notebook "
        "de recherche sur critère rappel + PR-AUC. "
        "Le modèle A (coût-sensitif, C=1.0) est plus conservateur sur les fausses alarmes."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Onglet 4 — Méthodologie
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Méthodologie et choix de modélisation")

    left, right = st.columns(2)

    with left:
        st.markdown(f"""
### 🎯 Objectif
Prédire la probabilité qu'un contrat d'**Assurance Décès Emprunteur (ADI)**
donne lieu à un sinistre (décès de l'assuré pendant la période couverte par
le crédit), afin d'aider les souscripteurs à cibler leur révision manuelle.

---

### 📂 Données utilisées
| Élément | Valeur |
|---|---|
| Périmètre | Contrats actifs uniquement (`Contrat_actif = 1`) |
| Taille | {len(df):,} contrats — {int(df[TARGET].sum())} sinistres |
| Taux de sinistre | {df[TARGET].mean()*100:.2f}% (déséquilibre sévère) |

**Variables exclues — fuite de données :**
- `Cout_sinistre` — montant du sinistre (connu uniquement après l'événement)
- `Prob_sinistre`, `Risk_Score`, `Risk_Segment` — scores dérivés non disponibles à la souscription

---

### 🔧 Traitement des variables
- **Log-transformation** (`log1p`) appliquée aux variables fortement asymétriques :
  revenu mensuel, montant du crédit, capital restant dû (CRD)
- **OneHotEncoding** (drop='first') pour les variables catégorielles
- **StandardScaler** pour la normalisation des variables numériques
- **Division train/test** : 75% / 25%, stratifiée sur la variable cible
""")

    with right:
        st.markdown(f"""
### 🤖 Modèle retenu
**Régression Logistique** avec `class_weight='balanced'` et `max_iter=2000`

| Critère | Justification |
|---|---|
| Interprétabilité | Coefficients directement lisibles |
| Auditabilité | Conforme aux exigences réglementaires assurance |
| Robustesse | Adapté aux petits jeux déséquilibrés |
| Rapidité | Entraînement instantané, inférence en temps réel |

---

### ⚖️ Gestion du déséquilibre de classes
Le taux de sinistre de **{metrics['pos_rate']*100:.2f}%** constitue un déséquilibre
sévère (environ 1 sinistre pour {int(1/metrics['pos_rate']):.0f} contrats sains).
La pondération `class_weight='balanced'` attribue automatiquement un poids
inversement proportionnel à la fréquence de chaque classe, forçant le modèle
à accorder une importance accrue aux sinistres rares.

---

### 📏 Seuil de décision
Le seuil optimal de **{metrics['threshold']*100:.0f}%** est sélectionné pour :
- **maximiser le score F1** sur l'ensemble de test
- sous contrainte de **rappel ≥ 50%** : priorité actuarielle de ne pas manquer
  les sinistres (un faux négatif coûte bien plus qu'une fausse alerte)

---

### ⚠️ Limites
- Performances susceptibles de se dégrader si la distribution du portefeuille évolue
- Les biais présents dans les données historiques peuvent être reproduits
- Cet outil est un **aide à la décision**, non un système automatique
""")

    st.divider()

    st.markdown("""
### 📚 Pour aller plus loin

La recherche complète associée à cet outil a évalué :
- **9 stratégies de rééchantillonnage** : Baseline, Sur-échantillonnage aléatoire,
  Sous-échantillonnage, SMOTE, ADASYN, Borderline-SMOTE, SMOTETomek, Coût-sensitif,
  Hybride (sous-échantillonnage + SMOTE)
- **4 familles de modèles** : Régression Logistique, Forêts Aléatoires, XGBoost,
  Réseau de neurones profond avec Focal Loss
- **Optimisation bayésienne** des hyperparamètres (Optuna)
- **Explicabilité** : SHAP (TreeExplainer, global + local) et LIME (instance-level)
- **Ensembling** : moyenne simple, moyenne pondérée, stacking

Les résultats complets sont disponibles dans :
`ADI_Maroc_Insurance_Claim_Risk_Modeling.ipynb`
""")

    st.caption(
        "Application développée dans le cadre d'un Projet de Fin d'Études (PFE 2026) par Yassmine Ez-zarrad· "
        "ADI Maroc · 2024–2025"
    )
