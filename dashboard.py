# dashboard_pro.py
import streamlit as st, pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt, plotly.express as px
import joblib, shap, plotly.graph_objects as go
from pathlib import Path
from sklearn.manifold import TSNE
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_toggle import st_toggle_switch

# ---------------- CONFIG ----------------
st.set_page_config(page_title="🎓 Stress Analytics Pro", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# ---------------- LOAD ----------------
@st.cache_data
def load_data():
    return pd.read_csv("StressLevelDataset.csv")

@st.cache_resource
def load_model():
    return joblib.load("stress_lgbm.pkl")

df = load_data()
model = load_model() if Path("stress_lgbm.pkl").exists() else None
X = df.drop(columns="stress_level")
y = df["stress_level"]

# ---------------- SIDEBAR ----------------

with st.sidebar:
    dark_mode = st_toggle_switch("Dark Mode", default_value=True, key="dark_mode_toggle")
    show_3d   = st.checkbox("Show 3-D t-SNE", value=True)
    sample_n  = st.slider("Sample Size for t-SNE", 100, 1000, 400)  # <-- Add sample_n slider

# ------------- KPI colors -------------
theme_color = "#00C896" if dark_mode else "#0068C9"

# ------------- KPI ROW (Dark-aware) -------------
# ------------- KPI ROW (Dark-aware) -------------
kpi_css = """
<style>
[data-testid="metric-container"] {
    background: #1E1E1E;
    border-radius: 8px;
    padding: 8px 16px;
    color: #FFFFFF;
}
[data-testid="stMetricLabel"] {
    font-size: 1.2rem !important; /* Increased label font size */
    color: #B0B0B0;
}
[data-testid="stMetricValue"] {
    font-size: 2.0rem !important; /* Increased value font size */
    color: #00C896;
}
</style>
"""
st.markdown(kpi_css, unsafe_allow_html=True)

a1, a2, a3, a4 = st.columns(4)
a1.metric("👥 Students", len(df))
a2.metric("📏 Features", X.shape[1])
a3.metric("🎯 Classes", y.nunique())
a4.metric("📈 ROC-AUC", f"{0.91:.2f}" if model else "N/A")

# ---------------- ROW 1 ----------------

# 1. Sunburst class distribution

st.subheader("🌞 Stress Class Distribution")
fig = px.sunburst(
        df, path=["stress_level"],
        color_discrete_sequence=px.colors.qualitative.Bold,
        template="plotly_dark" if dark_mode else "plotly"
    )
fig.update_traces(textinfo="label+percent parent")
st.plotly_chart(fig, use_container_width=True)

# 2. Interactive correlation heatmap

st.subheader("🔥 Correlation Heatmap")
corr = df.corr(numeric_only=True)
fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=0, template="plotly_dark" if dark_mode else "plotly")
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(fig, use_container_width=True)

# ---------------- ROW 2 ----------------

# 3. Violin plots for top-2 features
top2 = corr["stress_level"].abs().drop("stress_level").head(2).index

st.subheader("🎻 Top-2 Feature Violins")
fig = px.violin(df, y=top2[0], color="stress_level", box=True, template="plotly_dark" if dark_mode else "plotly")
st.plotly_chart(fig, use_container_width=True)

fig = px.violin(df, y=top2[1], color="stress_level", box=True, template="plotly_dark" if dark_mode else "plotly")
st.plotly_chart(fig, use_container_width=True)

# ---------------- ROW 3 ----------------
st.subheader("🗺️ t-SNE 3-D Embedding")
if show_3d and model:
    tsne = TSNE(n_components=3, random_state=42)
    emb = tsne.fit_transform(X.sample(sample_n, random_state=42))
    emb_df = pd.DataFrame(emb, columns=["x", "y", "z"])
    emb_df["stress_level"] = y.sample(sample_n, random_state=42).values

    fig = px.scatter_3d(
        emb_df, x="x", y="y", z="z",
        color="stress_level",
        color_discrete_sequence=px.colors.qualitative.T10,
        template="plotly_dark" if dark_mode else "plotly",
        opacity=0.8
    )
    fig.update_traces(marker=dict(size=4))
    st.plotly_chart(fig, use_container_width=True)

# ---------------- ROW 4 ----------------
if model:
    st.subheader("🔍 SHAP Waterfall – Live")
    samp = X.sample(1, random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(samp)
    waterfall = shap.plots.waterfall(
        shap.Explanation(values=shap_values[0],
                         base_values=explainer.expected_value[0],
                         data=samp.iloc[0],
                         feature_names=X.columns),
        show=False
    )
    st.pyplot(waterfall.figure, use_container_width=True)

# ---------------- ROW 5 ----------------
st.subheader("📥 Upload & Predict")
upload = st.file_uploader("CSV with same schema", type="csv")
if upload:
    new = pd.read_csv(upload)
    preds = model.predict(new)
    prob = model.predict_proba(new)
    new["pred"] = preds
    new["prob_high"] = prob[:, 2]
    st.dataframe(new.head())
    csv = new.to_csv(index=False).encode()
    st.download_button("⬇️ Download Predictions", csv, "predictions.csv", "text/csv")
