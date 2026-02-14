# ============================================================
# DASHBOARD STREAMLIT — MÉTÉO TUNIS (style BIAT)
# Sidebar boutons -> navigation pages
# + CSV viewer + EDA + Alertes + Prédiction demain + Saisie manuelle
# ============================================================

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(
    page_title="Météo Tunis — Dashboard",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# CSS (look premium)
# ----------------------------
def inject_css():
    st.markdown(
        """
        <style>
          .stApp {
            background: radial-gradient(1100px 600px at 10% 10%, rgba(70,110,255,0.12), transparent 60%),
                        radial-gradient(900px 500px at 90% 15%, rgba(255,140,0,0.10), transparent 55%),
                        linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.00));
          }
          section[data-testid="stSidebar"]{
            background: linear-gradient(180deg, rgba(20,22,28,0.95), rgba(15,16,20,0.95));
            border-right: 1px solid rgba(255,255,255,0.08);
          }
          section[data-testid="stSidebar"] * { color: rgba(255,255,255,0.92) !important; }

          .stButton>button{
            width:100%;
            border-radius:14px;
            padding:10px 12px;
            border:1px solid rgba(255,255,255,0.14);
            background: rgba(255,255,255,0.06);
            transition: .2s ease;
          }
          .stButton>button:hover{
            transform: translateY(-1px);
            border:1px solid rgba(255,255,255,0.28);
            background: rgba(255,255,255,0.10);
          }

          .card{
            padding: 16px;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.10);
            background: rgba(255,255,255,0.05);
            box-shadow: 0 10px 28px rgba(0,0,0,0.18);
          }
          .card-title{ font-size: 13px; opacity:.85; margin-bottom:6px; }
          .card-value{ font-size: 26px; font-weight: 800; line-height:1.1; }
          .card-sub{ font-size: 12px; opacity:.75; margin-top:6px; }

          div[data-testid="stDataFrame"]{
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.10);
            overflow: hidden;
          }
        </style>
        """,
        unsafe_allow_html=True
    )

inject_css()

# ----------------------------
# PATHS (modifiables dans la sidebar)
# ----------------------------
DEFAULT_DATA_PATH = "tunis_2010_2024_clean_features.csv"
DEFAULT_MODEL_PATH = "best_model_t2mmax_nextday_2010_2024.pkl"

# ----------------------------
# UTILS
# ----------------------------
def file_exists(path: str) -> bool:
    return Path(path).exists()

def safe_read_csv(path: str) -> pd.DataFrame:
    if not file_exists(path):
        st.error(f"CSV introuvable: {path}")
        st.stop()

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    else:
        st.error("Le CSV doit contenir une colonne 'date'.")
        st.stop()

    return df

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return safe_read_csv(path)

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

def kpi_card(title: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="card">
          <div class="card-title">{title}</div>
          <div class="card-value">{value}</div>
          <div class="card-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------------------
# ALERTES
# ----------------------------
def detect_extreme_events(df: pd.DataFrame, th_heat=40.0, th_rain=30.0, th_wind=12.0, th_freeze=0.0) -> pd.DataFrame:
    alerts = []

    if "T2M_MAX" in df.columns:
        m = df["T2M_MAX"] >= th_heat
        for _, r in df.loc[m, ["date", "T2M_MAX"]].iterrows():
            alerts.append({"date": r["date"], "type": "Canicule", "valeur": float(r["T2M_MAX"]), "seuil": th_heat})

    if "PRECTOTCORR" in df.columns:
        m = df["PRECTOTCORR"] >= th_rain
        for _, r in df.loc[m, ["date", "PRECTOTCORR"]].iterrows():
            alerts.append({"date": r["date"], "type": "Pluie extrême", "valeur": float(r["PRECTOTCORR"]), "seuil": th_rain})

    if "WS2M" in df.columns:
        m = df["WS2M"] >= th_wind
        for _, r in df.loc[m, ["date", "WS2M"]].iterrows():
            alerts.append({"date": r["date"], "type": "Vent fort", "valeur": float(r["WS2M"]), "seuil": th_wind})

    if "T2M_MIN" in df.columns:
        m = df["T2M_MIN"] < th_freeze
        for _, r in df.loc[m, ["date", "T2M_MIN"]].iterrows():
            alerts.append({"date": r["date"], "type": "Gel", "valeur": float(r["T2M_MIN"]), "seuil": th_freeze})

    a = pd.DataFrame(alerts)
    if not a.empty:
        a = a.sort_values("date").reset_index(drop=True)
    return a

# ----------------------------
# ML HELPERS (cible lendemain)
# ----------------------------
def build_nextday_target(df: pd.DataFrame) -> pd.DataFrame:
    if "T2M_MAX" not in df.columns:
        st.error("T2M_MAX introuvable dans le CSV.")
        st.stop()
    d = df.copy()
    d["y_next"] = d["T2M_MAX"].shift(-1)
    d = d.dropna(subset=["y_next"]).reset_index(drop=True)
    return d

def build_ml_input_from_row(row_df: pd.DataFrame) -> pd.DataFrame:
    # On enlève ce qu'on ne veut pas dans X (comme Phase 3)
    X = row_df.drop(columns=["date", "T2M_MAX", "y_next"], errors="ignore").copy()
    # Optionnel: retirer colonnes "raw date" redondantes si tu veux
    # (ne supprime rien si ton modèle a été entraîné avec ces colonnes)
    return X

def split_time(df2: pd.DataFrame, train_ratio=0.8):
    split_idx = int(len(df2) * train_ratio)
    train = df2.iloc[:split_idx].copy()
    test = df2.iloc[split_idx:].copy()
    return train, test

def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

# ============================================================
# SIDEBAR — Navigation boutons (style BIAT)
# ============================================================
PAGES = [
    ("🏠 Accueil", "home"),
    ("📄 Données", "data"),
    ("📈 EDA", "eda"),
    ("🚨 Alertes", "alerts"),
    ("🔮 Prédiction", "predict"),
    ("🧪 Évaluation", "eval"),
    ("ℹ️ À propos", "about"),
]

if "page" not in st.session_state:
    st.session_state.page = "home"

with st.sidebar:
    st.markdown("## 🌦️ Menu")

    for label, key in PAGES:
        if st.button(label):
            st.session_state.page = key

    st.markdown("---")
    st.markdown("## 📦 Fichiers")
    data_path = st.text_input("CSV", value=DEFAULT_DATA_PATH)
    model_path = st.text_input("Modèle .pkl", value=DEFAULT_MODEL_PATH)
    st.caption("Conseil : mets les fichiers dans le même dossier que app.py")

# Charger données
df = load_data(data_path)

# ============================================================
# PAGES
# ============================================================

def page_home():
    st.title("🌦️ Dashboard Météo — Tunis")
    st.caption("le temps qu’il fait, là où tu es.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Période", f"{df['date'].min().date()} → {df['date'].max().date()}")
    c2.metric("Nb jours", f"{len(df)}")
    if "T2M_MAX" in df.columns:
        c3.metric("T2M_MAX moyenne", f"{df['T2M_MAX'].mean():.2f} °C")
        c4.metric("T2M_MAX max", f"{df['T2M_MAX'].max():.2f} °C")

    st.markdown("---")
    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("✅ Fonctionnalités")
        st.write(
            """
            - **Données** : visualiser CSV, infos, stats, télécharger  
            - **EDA** : tendances, saisonnalité, distributions  
            - **Alertes** : canicule / pluie extrême / vent fort / gel  
            - **Prédiction** : prédire **T2M_MAX du lendemain**  
            - **Évaluation** : MAE, RMSE, R² + courbes + résidus
            """
        )

    with right:
        st.subheader("Aperçu rapide")
        st.dataframe(df.head(10), use_container_width=True)

def page_data():
    st.title("📄 Données (CSV)")

    st.subheader("Dataset complet")
    st.dataframe(df, use_container_width=True)

    st.markdown("### Infos")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Shape** :", df.shape)
        st.write("**Colonnes** :", list(df.columns))
    with c2:
        na = df.isna().sum().sort_values(ascending=False)
        st.write("**Missing values (Top 20)**")
        st.dataframe(na.head(20), use_container_width=True)

    st.markdown("### Statistiques numériques")
    num = df.select_dtypes(include=["number"])
    if num.shape[1] > 0:
        st.dataframe(num.describe().T, use_container_width=True)
    else:
        st.info("Aucune colonne numérique détectée.")

    st.markdown("### Télécharger le CSV (export)")
    st.download_button(
        "⬇️ Télécharger",
        data=df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
        file_name="export_dashboard.csv",
        mime="text/csv"
    )

def page_eda():
    st.title("📈 EDA")

    # Filtres dates
    st.subheader("Filtres")
    start, end = st.date_input("Période", (df["date"].min().date(), df["date"].max().date()))
    d = df[(df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))].copy()

    num_cols = d.select_dtypes(include=["number"]).columns.tolist()
    if not num_cols:
        st.info("Pas de colonnes numériques à tracer.")
        return

    col = st.selectbox("Variable", num_cols)
    agg = st.selectbox("Agrégation", ["Jour", "Mois", "Année"], index=1)

    s = d.set_index("date")[col].dropna()
    if agg == "Jour":
        series = s
    elif agg == "Mois":
        series = s.resample("M").mean()
    else:
        series = s.resample("Y").mean()

    fig = plt.figure()
    plt.plot(series.index, series.values)
    plt.title(f"{col} — {agg}")
    plt.xlabel("Date")
    plt.ylabel(col)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Boxplot par mois")
    tmp = d.copy()
    tmp["mois_num"] = tmp["date"].dt.month
    data = [tmp[tmp["mois_num"] == m][col].dropna().values for m in range(1, 13)]
    fig2 = plt.figure()
    plt.boxplot(data, labels=list(range(1, 13)))
    plt.title(f"Boxplot mensuel — {col}")
    plt.xlabel("Mois")
    plt.ylabel(col)
    plt.tight_layout()
    st.pyplot(fig2)

def page_alerts():
    st.title("🚨 Alertes météo automatiques")

    # Dates
    start, end = st.date_input("Période", (df["date"].min().date(), df["date"].max().date()))
    d = df[(df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))].copy()

    st.subheader("Seuils")
    c1, c2, c3, c4 = st.columns(4)
    th_heat = c1.number_input("Canicule (T2M_MAX ≥ °C)", value=40.0)
    th_rain = c2.number_input("Pluie extrême (mm/j ≥)", value=30.0)
    th_wind = c3.number_input("Vent fort (m/s ≥)", value=12.0)
    th_freeze = c4.number_input("Gel (T2M_MIN < °C)", value=0.0)

    alerts = detect_extreme_events(d, th_heat, th_rain, th_wind, th_freeze)

    st.markdown("---")
    cA, cB, cC = st.columns([1, 1, 2])
    with cA:
        kpi_card("Alertes totales", f"{len(alerts)}", "sur la période")
    with cB:
        kpi_card("Types", f"{alerts['type'].nunique() if not alerts.empty else 0}", "différents")
    with cC:
        kpi_card("Statut", "✅ OK" if alerts.empty else "⚠️ Attention", "résumé")

    st.subheader("Table des alertes")
    if alerts.empty:
        st.success("Aucune alerte détectée ✅")
    else:
        st.dataframe(alerts, use_container_width=True)

def page_predict():
    st.title("🔮 Prédire T2M_MAX du lendemain")
    st.caption("2 modes : (1) baser sur une date existante (recommandé) (2) saisir manuellement")

    if not file_exists(model_path):
        st.warning("Modèle .pkl introuvable. Mets-le dans le dossier et vérifie le nom.")
        st.info("Tu peux quand même utiliser les pages Données/EDA/Alertes.")
        return

    model = load_model(model_path)

    df2 = build_nextday_target(df)

    mode = st.radio("Mode", ["Basé sur une date du dataset", "Saisie manuelle (avancé)"], horizontal=True)

    if mode == "Basé sur une date du dataset":
        # Choisir une date existante
        dates = df2["date"].dt.date.unique()
        pick = st.selectbox("Date (features)", dates, index=len(dates)-2 if len(dates) > 2 else 0)

        row = df2[df2["date"].dt.date == pick].iloc[-1:].copy()

        # Option : modifier quelques champs avant prédiction
        st.markdown("### Option : modifier des valeurs avant de prédire (facultatif)")
        editable_cols = [c for c in ["T2M_MIN", "RH2M", "WS2M", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"] if c in row.columns]

        cols_ui = st.columns(len(editable_cols)) if editable_cols else []
        for i, c in enumerate(editable_cols):
            row[c] = cols_ui[i].number_input(f"{c}", value=float(row[c].iloc[0]))

        X_row = build_ml_input_from_row(row)
        pred = float(model.predict(X_row)[0])

        c1, c2, c3 = st.columns([1, 1, 2])
        c1.metric("Date sélectionnée", str(pick))
        c2.metric("Prédiction demain", f"{pred:.2f} °C")
        th = c3.number_input("Seuil canicule (°C)", value=40.0)
        st.info("🔥 CANICULE (prédite)" if pred >= th else "✅ Pas de canicule (prédite)")

        st.markdown("### Features utilisées")
        st.dataframe(X_row.T, use_container_width=True)

    else:
        # Saisie manuelle (avancé) : on exige que tu entres TOUTES les colonnes que le modèle attend.
        st.warning("Mode avancé : à utiliser si tu connais les colonnes attendues par ton modèle pipeline.")
        st.write("Astuce : entraîne ton modèle sur des colonnes simples pour faciliter la saisie manuelle.")

        # On prend un “template” basé sur une ligne du dataset
        template = df2.iloc[-2:-1].copy()
        X_template = build_ml_input_from_row(template)

        st.markdown("### Formulaire de saisie")
        form_vals = {}
        with st.form("manual_form"):
            st.write("Remplis les valeurs (les colonnes non numériques restent texte).")
            for c in X_template.columns:
                if pd.api.types.is_numeric_dtype(X_template[c]):
                    form_vals[c] = st.number_input(c, value=float(X_template[c].iloc[0]))
                else:
                    form_vals[c] = st.text_input(c, value=str(X_template[c].iloc[0]))
            submitted = st.form_submit_button("Prédire")

        if submitted:
            X_manual = pd.DataFrame([form_vals])
            pred = float(model.predict(X_manual)[0])
            st.success(f"✅ Prédiction T2M_MAX demain = {pred:.2f} °C")
            st.dataframe(X_manual, use_container_width=True)

def page_eval():
    st.title("🧪 Évaluation du modèle")

    if not file_exists(model_path):
        st.warning("Modèle .pkl introuvable.")
        return

    model = load_model(model_path)

    df2 = build_nextday_target(df)
    train, test = split_time(df2, train_ratio=0.8)

    X_test = build_ml_input_from_row(test)
    y_test = test["y_next"].astype(float)

    y_pred = model.predict(X_test)

    mae, rmse, r2 = metrics(y_test, y_pred)

    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("MAE", f"{mae:.3f}", "Erreur absolue moyenne")
    with c2: kpi_card("RMSE", f"{rmse:.3f}", "Plus sensible aux grosses erreurs")
    with c3: kpi_card("R²", f"{r2:.3f}", "Variance expliquée")

    st.markdown("---")
    st.subheader("Réel vs Prédit")
    fig = plt.figure()
    plt.plot(test["date"], y_test.values, label="Réel")
    plt.plot(test["date"], y_pred, label="Prédit")
    plt.title("T2M_MAX (lendemain) — Test")
    plt.xlabel("Date")
    plt.ylabel("°C")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Résidus")
    resid = y_test.values - y_pred
    fig2 = plt.figure()
    plt.plot(test["date"], resid)
    plt.title("Résidus (Réel - Prédit)")
    plt.xlabel("Date")
    plt.ylabel("Erreur (°C)")
    plt.tight_layout()
    st.pyplot(fig2)

def page_about():
    st.title("ℹ️ À propos")
    st.write(
        """
        Application Streamlit pour :
        - Visualiser les données NASA POWER (Tunis)
        - Faire de l’EDA
        - Générer des alertes météo
        - Prédire la température maximale du lendemain (si modèle dispo)
        """
    )

# ROUTER
routes = {
    "home": page_home,
    "data": page_data,
    "eda": page_eda,
    "alerts": page_alerts,
    "predict": page_predict,
    "eval": page_eval,
    "about": page_about,
}

routes.get(st.session_state.page, page_home)()
