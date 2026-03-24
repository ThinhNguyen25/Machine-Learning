import streamlit as st
import joblib
import os
import pandas as pd
import re
from underthesea import word_tokenize
import plotly.express as px
from sentence_transformers import SentenceTransformer

# ======================
# CONFIG PAGE
# ======================
st.set_page_config(
    page_title="Sentiment Dashboard",
    page_icon="📊",
    layout="wide"
)

# ======================
# CUSTOM CSS
# ======================
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.big-title {
    font-size: 40px;
    font-weight: bold;
    color: #00ffcc;
}
.card {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ======================
# LOAD EMBEDDING MODEL
# ======================
@st.cache_resource
def load_embed_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

embed_model = load_embed_model()

# ======================
# LOAD MODELS
# ======================
@st.cache_resource
def load_all_models():
    metadata = joblib.load("model_metadata.pkl")
    
    models = []
    
    for item in metadata:
        filename = os.path.basename(item["path"])
        local_path = os.path.join("saved_models", filename)
        
        model = joblib.load(local_path)
        
        models.append({
            "name": item["name"],
            "model": model,
            "accuracy": item["accuracy"],
            "f1": item["f1"],
            "feature": item["feature"]
        })
    
    return models

models = load_all_models()

# ======================
# PREPROCESS
# ======================
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = word_tokenize(text, format="text")
    return text

label_map = {
    0: "Negative 😡",
    1: "Neutral 😐",
    2: "Positive 😍"
}

# ======================
# HEADER
# ======================
st.markdown('<div class="big-title">🚀 Sentiment Model Dashboard</div>', unsafe_allow_html=True)
st.markdown("### So sánh nhiều mô hình học máy trên cùng một câu")

# ======================
# SIDEBAR
# ======================
st.sidebar.title("⚙️ Thông tin")
st.sidebar.write(f"📦 Tổng model: {len(models)}")
st.sidebar.info("BoW / TF-IDF / N-gram / Char n-gram / Embedding")

# ======================
# INPUT
# ======================
col1, col2 = st.columns([2,1])

with col1:
    user_input = st.text_area(
        "✍️ Nhập câu:",
        "môn này chán vl",
        height=150
    )

with col2:
    st.markdown("### ⚡ Test nhanh")
    examples = [
        "môn này chán vl",
        "giảng viên dạy rất hay",
        "bình thường",
        "lecture này khá chill",
        "thầy giảng cuốn vãi"
    ]
    for ex in examples:
        if st.button(ex):
            user_input = ex

# ======================
# PREDICT
# ======================
if st.button("🚀 Predict All Models"):
    
    processed = preprocess(user_input)
    
    results = []
    
    for m in models:
        try:
            if m["feature"] == "Embedding":
                emb = embed_model.encode([user_input])
                pred = m["model"].predict(emb)[0]
            else:
                pred = m["model"].predict([processed])[0]
            
            results.append({
                "Model": m["name"],
                "Prediction": label_map[pred],
                "Accuracy": round(m["accuracy"], 3),
                "F1": round(m["f1"], 3)
            })
        
        except Exception as e:
            results.append({
                "Model": m["name"],
                "Prediction": "Error",
                "Accuracy": 0,
                "F1": 0
            })

    df = pd.DataFrame(results)
    df = df.sort_values(by="F1", ascending=False)

    # BEST MODEL
    best = df.iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("🏆 Best Model", best["Model"])
    col2.metric("F1 Score", best["F1"])
    col3.metric("Prediction", best["Prediction"])

    # DISAGREEMENT CHECK
    if len(df["Prediction"].unique()) > 1:
        st.warning("⚠️ Models disagree → input khó hoặc từ mới")
    else:
        st.success("✅ All models agree")

    # TABLE
    st.markdown("## 📋 Model Comparison")
    st.dataframe(df, use_container_width=True)

    # TOP 3
    st.markdown("## 🥇 Top 3 Models")
    st.table(df.head(3)[["Model", "F1", "Prediction"]])

    # BAR CHART
    fig = px.bar(
        df,
        x="Model",
        y="F1",
        color="Prediction",
        text_auto=".3f",
        title="F1 Score Comparison"
    )
    st.plotly_chart(fig, use_container_width=True)

# ======================
# FOOTER
# ======================
st.markdown("---")
st.markdown("💡 AIoT Student Project | Machine Learning Dashboard")
# streamlit run app.py