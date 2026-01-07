
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# GLOBAL PLOT SETTINGS (CLARITY)
# -----------------------------
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Telco Churn Dashboard",
    page_icon="📉",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS (UNCHANGED)
# -----------------------------
st.markdown("""
<style>
.main { background-color: #0e1117; }
h1, h2, h3 { color: #f1f1f1; }
.metric-card {
    background-color: #1f2933;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.4);
}
.metric-title { font-size: 18px; color: #9ca3af; }
.metric-value { font-size: 32px; color: #38bdf8; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------
st.markdown("<h1>📊 Telco Customer Churn Analytics</h1>", unsafe_allow_html=True)
st.markdown("### Predict • Analyze • Retain Customers")

# -----------------------------
# LOAD DATA (ERROR-FREE)
# -----------------------------
@st.cache_data
def load_data():
    files = [
        "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "WA_Fn-UseC_-Telco-Customer-Churn (1).csv"
    ]
    for f in files:
        try:
            return pd.read_csv(f)
        except:
            continue
    st.error("Dataset not found. Keep CSV in same folder as app.py")
    st.stop()

df = load_data()

# -----------------------------
# DATA PREVIEW
# -----------------------------
st.subheader("📄 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# -----------------------------
# DATA CLEANING
# -----------------------------
df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("⚙ Dashboard Controls")
test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)
show_data = st.sidebar.checkbox("Show Raw Data")

if show_data:
    st.sidebar.dataframe(df.head())

# -----------------------------
# MODEL PIPELINE
# -----------------------------
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

TN, FP, FN, TP = cm.ravel()

# -----------------------------
# KPI METRICS
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Accuracy</div>
        <div class="metric-value">{accuracy:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">True Positives</div>
        <div class="metric-value">{TP}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">False Negatives</div>
        <div class="metric-value">{FN}</div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📌 Confusion Matrix", "📊 Tenure Analysis", "💰 Charges Analysis", "📄 Report", "🔮 Predict Churn"]
)

# ---------------- TAB 1 ----------------
with tab1:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        annot_kws={"size": 13},
        xticklabels=["Stay", "Churn"],
        yticklabels=["Stay", "Churn"]
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.markdown("""
    **TP** – Correctly identified churn  
    **TN** – Correctly identified non-churn  
    **FP** – Loyal customers flagged as churn  
    **FN** – Missed churn customers (most costly)
    """)

# ---------------- TAB 2 ----------------
with tab2:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(x="Churn", y="tenure", data=df, ax=ax)
    ax.set_xticklabels(["Stay", "Churn"])
    ax.set_xlabel("Customer Status")
    ax.set_ylabel("Tenure (Months)")
    ax.set_title("Tenure vs Churn")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ---------------- TAB 3 ----------------
with tab3:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(x="Churn", y="TotalCharges", data=df, ax=ax)
    ax.set_xticklabels(["Stay", "Churn"])
    ax.set_xlabel("Customer Status")
    ax.set_ylabel("Total Charges")
    ax.set_title("Total Charges vs Churn")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ---------------- TAB 4 ----------------
with tab4:
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    st.success(
        "Customers with lower tenure and lower total charges "
        "have higher churn risk. Early retention is critical."
    )

# ---------------- TAB 5 ----------------
with tab5:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    total = st.slider(
        "Total Charges",
        float(df["TotalCharges"].min()),
        float(df["TotalCharges"].max()),
        float(df["TotalCharges"].median())
    )

    input_df = pd.DataFrame({"tenure": [tenure], "TotalCharges": [total]})

    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[X.columns]
    input_scaled = scaler.transform(input_df)

    prob = model.predict_proba(input_scaled)[0][1]
    pred = model.predict(input_scaled)[0]

    st.metric("Churn Probability", f"{prob:.2f}")

    if pred == 1:
        st.error("⚠ Customer Likely to Churn")
    else:
        st.success("✅ Customer Likely to Stay")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("✅ **Built with Streamlit | Logistic Regression | Business Analytics**")
