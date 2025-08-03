# 🛍️ Shopper Spectrum

**Shopper Spectrum** is a data science project that combines **customer segmentation** and **product recommendation** using the popular **Online Retail dataset**. The project uses **RFM Analysis**, **KMeans Clustering**, and **Collaborative Filtering** to deliver a powerful dashboard for business insights and customer targeting, built using **Streamlit**.

---

## 📌 Features

### 🔍 Product Recommendation
- Enter any product name to receive **5 similar products**
- Based on **Cosine Similarity** and **Collaborative Filtering**

### 📊 Customer Segmentation
- Enter **Recency**, **Frequency**, and **Monetary** values
- Predicts the customer’s **cluster**:
  - 💎 High-Value
  - 🔁 Regular
  - 🎯 Occasional
  - ⚠️ At-Risk

### 📈 Data Visualization & Insights
- Transaction volume by top countries
- Top 10 selling products
- Monthly purchase trends
- Distribution of Recency, Frequency, Monetary
- RFM-based clustering with 3D PCA visualization

---

## 🎯 Business Objectives

- Understand customer purchase patterns
- Segment customers for targeted marketing
- Recommend products to boost sales
- Build a **Streamlit dashboard** for real-time analysis

---

## 🛠️ Tech Stack

- **Python**, **Pandas**, **NumPy**
- **Scikit-learn** (KMeans, PCA, StandardScaler)
- **Matplotlib**, **Seaborn**
- **Streamlit** for web app
- **Joblib** for model persistence
- **Git & GitHub** for version control
- **Git LFS** for large file support

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
``` bash
git clone https://github.com/sagar3103/ShopperSpectrum.git
cd ShopperSpectrum ```

### 2️⃣ Install Requirements
``` bash
Copy
Edit
pip install -r requirements.txt ```
### 3️⃣ Run the Streamlit App
``` bash
Copy
Edit
streamlit run app.py ```
