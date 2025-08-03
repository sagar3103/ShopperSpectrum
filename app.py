import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import difflib

# ---------------------------
# Load Data and Models
# ---------------------------
@st.cache_resource
def load_data():
    df = pd.read_csv("df.csv")
    product_list = joblib.load("product_list.pkl")
    similarity_matrix = joblib.load("product_similarity_matrix.pkl")
    kmeans = joblib.load("kmeans_rfm_model.pkl")
    scaler = joblib.load("rfm_scaler.pkl")
    return df, product_list, similarity_matrix, kmeans, scaler

df, product_list, similarity_matrix, kmeans, scaler = load_data()

df, product_list, similarity_matrix, kmeans, scaler = load_data()

# ------------------------------
# 🔍 Get Similar Products (Improved Search)
# ------------------------------
def get_similar_products(product_name, product_list, similarity_matrix, top_n=5):
    product_name = product_name.lower().strip()

    matches = difflib.get_close_matches(product_name, [p.lower() for p in product_list], n=3, cutoff=0.5)

    if matches:
        original_name = next((p for p in product_list if p.lower() == matches[0]), None)
    else:
        # Try pandas contains() search
        product_series = pd.Series(product_list)
        partial_matches = product_series[product_series.str.lower().str.contains(product_name, na=False)]
        if not partial_matches.empty:
            original_name = partial_matches.iloc[0]
        else:
            return None, None

    idx = product_list.index(original_name)
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommended_indices = [i for i, _ in similarity_scores[1:top_n+1]]
    return original_name, [product_list[i] for i in recommended_indices]

# ------------------------------
# 📊 Predict RFM Cluster
# ------------------------------
def predict_cluster(recency, frequency, monetary):
    scaled = scaler.transform([[recency, frequency, monetary]])
    cluster = kmeans.predict(scaled)[0]
    segment_map = {
        0: "High-Value",
        1: "Regular",
        2: "Occasional",
        3: "At-Risk"
    }
    return segment_map.get(cluster, "Unknown")

# ------------------------------
# 🌟 Streamlit UI
# ------------------------------
st.set_page_config(page_title="🛍️ Shopper Spectrum", layout="wide")
st.title("🛍️ Shopper Spectrum: Product Recommendation & Customer Segmentation")

tab1, tab2 = st.tabs(["🔎 Product Recommendation", "📈 Customer Segmentation"])

# ------------------------------
# 🔎 Tab 1: Product Recommendation
# ------------------------------
with tab1:
    st.subheader("🔍 Recommend Similar Products")

    # Dropdown for product selection
    product_selected = st.selectbox(
        "🔽 Choose or Search a Product:",
        options=sorted(product_list),
        help="Type product name to search and select"
    )

    if st.button("🎯 Get Recommendations"):
        product, recommendations = get_similar_products(product_selected, product_list, similarity_matrix)

        if not recommendations:
            st.warning("❌ No similar products found. Try another.")
        else:
            st.success(f"✅ Recommendations for: **{product}**")
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"**{i}.** {rec}")

# ------------------------------
# 📈 Tab 2: Customer Segmentation
# ------------------------------
with tab2:
    st.subheader("📊 Predict Customer Cluster")

    recency = st.number_input("📅 Recency (days since last purchase):", min_value=0, value=30)
    frequency = st.number_input("🔁 Frequency (number of purchases):", min_value=1, value=5)
    monetary = st.number_input("💰 Monetary Value (total spent in £):", min_value=1.0, value=100.0)

    if st.button("📍 Predict Segment"):
        segment = predict_cluster(recency, frequency, monetary)
        st.success(f"🧠 Predicted Segment: **{segment}**")