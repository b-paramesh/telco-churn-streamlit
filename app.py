import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Telco Churn Dashboard",
    page_icon="📉",
    layout="wide"
)

# ------------------------------------------------
# CUSTOM CSS (UI DESIGN)
# ------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
h1, h2, h3 {
    color: #f1f1f1;
}
.metric-card {
    background-color: #1f2933;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.4);
}
.metric-title {
    font-size: 18px;
    color: #9ca3af;
}
.metric-value {
    font-size: 32px;
    color: #38bdf8;
    font-weight: bold;
}
.sidebar .sidebar-content {
    background-color: #020617;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# TITLE SECTION
# ------------------------------------------------
st.markdown("<h1>📊 Telco Customer Churn Analytics</h1>", unsafe_allow_html=True)
st.markdown("### Predict • Analyze • Retain Customers")

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------
try:
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
except:
    st.error("⚠️ Dataset not found. Place CSV in same folder as app.py")
    st.stop()

# ------------------------------------------------
# DATA CLEANING
# ------------------------------------------------
df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------
st.sidebar.markdown("## ⚙ Dashboard Controls")
test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)
show_data = st.sidebar.checkbox("Show Raw Data")

if show_data:
    st.sidebar.dataframe(df.head())

# ------------------------------------------------
# MODEL PIPELINE
# ------------------------------------------------
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# ------------------------------------------------
# KPI METRICS
# ------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Model Accuracy</div>
        <div class="metric-value">{accuracy:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    churn_rate = df["Churn"].mean() * 100
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Churn Rate</div>
        <div class="metric-value">{churn_rate:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Customers</div>
        <div class="metric-value">{df.shape[0]}</div>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------
# TABS
# ------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📌 Confusion Matrix", "📊 Tenure Analysis", "💰 Charges Analysis", "📄 Report"]
)

# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Stay", "Churn"],
                yticklabels=["Stay", "Churn"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("Tenure vs Churn")
    fig, ax = plt.subplots()
    sns.boxplot(x="Churn", y="tenure", data=df, ax=ax)
    ax.set_xticklabels(["Stay", "Churn"])
    st.pyplot(fig)

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("Total Charges vs Churn")
    fig, ax = plt.subplots()
    sns.boxplot(x="Churn", y="TotalCharges", data=df, ax=ax)
    ax.set_xticklabels(["Stay", "Churn"])
    st.pyplot(fig)

# ---------------- TAB 4 ----------------
with tab4:
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.markdown("### 💡 Business Insight")
    st.success(
        "Customers with low tenure and low total charges have a higher churn risk. "
        "Retention strategies should focus on early-stage customers."
    )

st.markdown("---")
st.markdown("✅ **Built with Streamlit | Logistic Regression | Business Analytics**")
