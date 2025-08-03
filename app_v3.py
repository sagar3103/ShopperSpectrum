import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved files
df = joblib.load("df.pkl")
rfm_df = joblib.load("rfm_df.pkl")         # This contains RFM and Segment column
scaler = joblib.load("scaler.pkl")         # Trained StandardScaler
kmeans = joblib.load("kmeans_model.pkl")   # Trained KMeans model

# Define segment label mapping
segment_labels = {
    0: "High-Value",
    1: "Regular",
    2: "Occasional",
    3: "At-Risk"
}

# --- STREAMLIT TAB 2: CUSTOMER SEGMENTATION ---
st.header("🎯 Customer Segmentation")

# Sidebar or inputs for user-defined RFM
st.markdown("#### 🔢 Enter RFM Values to Predict Segment")

recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=90)
frequency = st.number_input("Frequency (number of purchases)", min_value=1, max_value=100, value=5)
monetary = st.number_input("Monetary (total spend)", min_value=1.0, max_value=10000.0, value=100.0)

if st.button("🔍 Predict Segment"):
    user_rfm = pd.DataFrame({
        'Recency': [recency],
        'Frequency': [frequency],
        'Monetary': [monetary]
    })

    # Scale input
    user_rfm_scaled = scaler.transform(user_rfm)

    # Predict cluster
    cluster = kmeans.predict(user_rfm_scaled)[0]
    segment_label = segment_labels.get(cluster, "Unknown")

    # Show result
    st.success(f"🎯 Predicted Segment: **{segment_label}** (Cluster {cluster})")

# Table 2 - Cluster-wise Min & Max for R, F, M
st.markdown("### 📊 Table 2: RFM Min-Max Summary for Each Segment")

# Make sure Segment is labeled for groupby
if 'Segment' not in rfm_df.columns:
    rfm_df['Segment'] = rfm_df['Cluster'].map(segment_labels)

summary_table = rfm_df.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].agg(['min', 'max']).round(2)
summary_table.columns = ['_'.join(col) for col in summary_table.columns]
summary_table.reset_index(inplace=True)

st.dataframe(summary_table, use_container_width=True)
