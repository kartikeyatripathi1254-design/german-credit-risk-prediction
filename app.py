"""
German Credit Risk Analyzer
============================
Streamlit app for predicting credit risk using the UCI German Credit Dataset.
Data source: UCI ML Repository — Statlog German Credit Dataset
             Original collector: Prof. Hans Hofmann, University of Hamburg

Deploy: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="German Credit Risk Analyzer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0D1B3E;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #F0F4F8;
        border-left: 4px solid #0A9396;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    .good-credit {
        background: #DCFCE7;
        border-left: 4px solid #16A34A;
        padding: 1rem;
        border-radius: 6px;
        font-size: 1.2rem;
        font-weight: 600;
        color: #166534;
    }
    .bad-credit {
        background: #FEE2E2;
        border-left: 4px solid #DC2626;
        padding: 1rem;
        border-radius: 6px;
        font-size: 1.2rem;
        font-weight: 600;
        color: #991B1B;
    }
</style>
""", unsafe_allow_html=True)

# ── Load and prepare data ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """
    Load UCI German Credit Dataset.
    Downloads directly from UCI ML Repository — the original real source.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

    column_names = [
        "checking_account", "duration", "credit_history", "purpose",
        "credit_amount", "savings_account", "employment", "installment_rate",
        "personal_status", "other_debtors", "residence_since", "property",
        "age", "other_installments", "housing", "existing_credits",
        "job", "liable_people", "telephone", "foreign_worker", "credit_risk"
    ]

    try:
        df = pd.read_csv(url, sep=" ", header=None, names=column_names)
        return df, None
    except Exception as e:
        return None, str(e)


@st.cache_data
def prepare_model(df):
    """Clean data, encode categoricals, train Random Forest model."""
    df = df.copy()

    # Convert target: 1 = Good, 2 = Bad → 0 = Good, 1 = Bad
    df["credit_risk"] = df["credit_risk"].map({1: 0, 2: 1})

    # Encode all categorical (object) columns
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop("credit_risk", axis=1)
    y = df["credit_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100, max_depth=8,
        random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    return model, X, X_train, X_test, y_test, y_pred, accuracy, report, cm


# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/b/ba/Flag_of_Germany.svg",
    width=60
)
st.sidebar.markdown("## 🏦 German Credit Risk")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 Data Explorer", "🤖 ML Model", "🔮 Predict Risk", "📋 About"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Data Source**
[UCI ML Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
Prof. Hans Hofmann
University of Hamburg

**Tech Stack**
Python · Sklearn · Streamlit
Pandas · Seaborn
""")

# ── Load data ─────────────────────────────────────────────────────────────────
df, load_error = load_data()

if load_error:
    st.error(f"Could not load data from UCI: {load_error}")
    st.info("Upload your german.data file below as a fallback.")
    uploaded = st.file_uploader("Upload german.data file", type=["data", "csv"])
    if uploaded:
        column_names = [
            "checking_account", "duration", "credit_history", "purpose",
            "credit_amount", "savings_account", "employment", "installment_rate",
            "personal_status", "other_debtors", "residence_since", "property",
            "age", "other_installments", "housing", "existing_credits",
            "job", "liable_people", "telephone", "foreign_worker", "credit_risk"
        ]
        df = pd.read_csv(uploaded, sep=" ", header=None, names=column_names)
    else:
        st.stop()

model, X, X_train, X_test, y_test, y_pred, accuracy, report, cm = prepare_model(df)

# ── PAGE: Overview ────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown('<div class="main-header">🏦 German Credit Risk Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">End-to-end ML pipeline for credit risk prediction · UCI German Credit Dataset</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        good = (df["credit_risk"] == 1).sum()
        st.metric("Good Credit", f"{good:,}", f"{good/len(df)*100:.0f}%")
    with col3:
        bad = (df["credit_risk"] == 2).sum()
        st.metric("Bad Credit", f"{bad:,}", f"{bad/len(df)*100:.0f}%")
    with col4:
        st.metric("Model Accuracy", f"{accuracy*100:.1f}%")

    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### What this project does")
        st.markdown("""
        This application predicts whether a bank customer is a **good or bad credit risk**
        based on their financial and personal profile.

        Built to demonstrate end-to-end data science skills relevant to the **German financial
        sector** — from raw data to a deployed prediction tool.

        **Pipeline:**
        1. Load real data from UCI ML Repository
        2. Clean and encode 20 features
        3. Train a Random Forest classifier
        4. Evaluate model performance
        5. Accept new inputs and predict risk in real time
        """)

    with col_r:
        st.markdown("### Credit Risk Distribution")
        fig, ax = plt.subplots(figsize=(5, 3))
        labels = ["Good Credit (70%)", "Bad Credit (30%)"]
        sizes = [700, 300]
        colors = ["#0A9396", "#E63946"]
        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.0f%%",
               startangle=90, textprops={"fontsize": 11})
        ax.axis("equal")
        st.pyplot(fig)
        plt.close()

# ── PAGE: Data Explorer ───────────────────────────────────────────────────────
elif page == "📊 Data Explorer":
    st.markdown("## 📊 Data Explorer")
    st.caption("UCI Statlog German Credit Dataset — 1,000 customers, 20 features")

    tab1, tab2, tab3 = st.tabs(["Raw Data", "Statistics", "Distributions"])

    with tab1:
        st.dataframe(df.head(50), use_container_width=True)
        st.caption(f"Showing 50 of {len(df):,} rows · {df.shape[1]} columns")

    with tab2:
        st.dataframe(df.describe().round(2), use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            df["age"].hist(bins=20, ax=ax, color="#0A9396", edgecolor="white")
            ax.set_title("Age Distribution", fontweight="bold")
            ax.set_xlabel("Age")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            df["credit_amount"].hist(bins=20, ax=ax, color="#0D1B3E", edgecolor="white")
            ax.set_title("Credit Amount Distribution", fontweight="bold")
            ax.set_xlabel("Credit Amount (DM)")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            plt.close()

        fig, ax = plt.subplots(figsize=(10, 4))
        df["duration"].hist(bins=15, ax=ax, color="#94D2BD", edgecolor="white")
        ax.set_title("Loan Duration Distribution (months)", fontweight="bold")
        ax.set_xlabel("Duration (months)")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close()

# ── PAGE: ML Model ────────────────────────────────────────────────────────────
elif page == "🤖 ML Model":
    st.markdown("## 🤖 Machine Learning Model")
    st.caption("Random Forest Classifier · 80/20 train-test split · Class balanced")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{accuracy*100:.1f}%")
    with col2:
        precision = report["1"]["precision"]
        st.metric("Bad Credit Precision", f"{precision*100:.1f}%")
    with col3:
        recall = report["1"]["recall"]
        st.metric("Bad Credit Recall", f"{recall*100:.1f}%")

    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Good Credit", "Bad Credit"],
            yticklabels=["Good Credit", "Bad Credit"]
        )
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        ax.set_title("Confusion Matrix", fontweight="bold")
        st.pyplot(fig)
        plt.close()

    with col_r:
        st.markdown("### Feature Importance (Top 10)")
        importances = pd.Series(
            model.feature_importances_, index=X.columns
        ).sort_values(ascending=True).tail(10)

        fig, ax = plt.subplots(figsize=(5, 4))
        importances.plot(kind="barh", ax=ax, color="#0A9396")
        ax.set_title("Top 10 Important Features", fontweight="bold")
        ax.set_xlabel("Importance Score")
        st.pyplot(fig)
        plt.close()

    st.markdown("### Full Classification Report")
    report_df = pd.DataFrame(report).transpose().round(2)
    st.dataframe(report_df, use_container_width=True)

# ── PAGE: Predict ─────────────────────────────────────────────────────────────
elif page == "🔮 Predict Risk":
    st.markdown("## 🔮 Predict Credit Risk")
    st.caption("Enter a customer profile below to get an instant credit risk prediction")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Loan Details**")
            duration = st.slider("Loan Duration (months)", 4, 72, 24)
            credit_amount = st.number_input("Credit Amount (DM)", 250, 20000, 3000, step=250)
            installment_rate = st.selectbox("Installment Rate (% of income)", [1, 2, 3, 4])
            purpose = st.selectbox("Purpose", [
                "A40 - car (new)", "A41 - car (used)", "A42 - furniture",
                "A43 - radio/TV", "A44 - appliances", "A45 - repairs",
                "A46 - education", "A48 - retraining", "A49 - business", "A410 - other"
            ])

        with col2:
            st.markdown("**Financial Profile**")
            checking_account = st.selectbox("Checking Account Status", [
                "A11 - < 0 DM", "A12 - 0-200 DM", "A13 - >= 200 DM", "A14 - no account"
            ])
            savings_account = st.selectbox("Savings Account", [
                "A61 - < 100 DM", "A62 - 100-500 DM", "A63 - 500-1000 DM",
                "A64 - >= 1000 DM", "A65 - unknown/none"
            ])
            credit_history = st.selectbox("Credit History", [
                "A30 - no credits", "A31 - all paid at bank",
                "A32 - all paid", "A33 - past delays", "A34 - critical account"
            ])
            existing_credits = st.selectbox("Existing Credits at Bank", [1, 2, 3, 4])

        with col3:
            st.markdown("**Personal Profile**")
            age = st.slider("Age", 19, 75, 35)
            employment = st.selectbox("Employment Duration", [
                "A71 - unemployed", "A72 - < 1 year", "A73 - 1-4 years",
                "A74 - 4-7 years", "A75 - >= 7 years"
            ])
            housing = st.selectbox("Housing", [
                "A151 - rent", "A152 - own", "A153 - free"
            ])
            job = st.selectbox("Job Type", [
                "A171 - unskilled non-resident", "A172 - unskilled resident",
                "A173 - skilled", "A174 - highly qualified"
            ])
            liable_people = st.selectbox("Number of Liable People", [1, 2])

        submitted = st.form_submit_button("🔮 Predict Credit Risk", use_container_width=True)

    if submitted:
        # Build input row matching training feature order
        # Using median/mode values for features not collected in form
        input_data = {
            "checking_account": [checking_account[:3]],
            "duration": [duration],
            "credit_history": [credit_history[:3]],
            "purpose": [purpose[:3]],
            "credit_amount": [credit_amount],
            "savings_account": [savings_account[:3]],
            "employment": [employment[:3]],
            "installment_rate": [installment_rate],
            "personal_status": ["A93"],       # male single (most common)
            "other_debtors": ["A101"],         # none
            "residence_since": [3],
            "property": ["A121"],              # real estate
            "age": [age],
            "other_installments": ["A143"],    # none
            "housing": [housing[:4]],
            "existing_credits": [existing_credits],
            "job": [job[:4]],
            "liable_people": [liable_people],
            "telephone": ["A192"],             # yes
            "foreign_worker": ["A201"],        # yes
        }

        input_df = pd.DataFrame(input_data)

        # Encode to match training
        le = LabelEncoder()
        for col in input_df.select_dtypes(include="object").columns:
            # Fit on the original column from training data
            le.fit(df[col].astype(str))
            try:
                input_df[col] = le.transform(input_df[col].astype(str))
            except ValueError:
                input_df[col] = 0  # unseen label fallback

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        st.markdown("---")
        st.markdown("### Prediction Result")

        col_res1, col_res2 = st.columns(2)
        with col_res1:
            if prediction == 0:
                st.markdown(
                    '<div class="good-credit">✅ GOOD CREDIT RISK<br>'
                    '<span style="font-size:0.9rem;font-weight:400;">Low risk of default</span></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="bad-credit">❌ BAD CREDIT RISK<br>'
                    '<span style="font-size:0.9rem;font-weight:400;">High risk of default</span></div>',
                    unsafe_allow_html=True
                )

        with col_res2:
            fig, ax = plt.subplots(figsize=(4, 2.5))
            bars = ax.barh(
                ["Good Credit", "Bad Credit"],
                [probability[0]*100, probability[1]*100],
                color=["#0A9396", "#E63946"]
            )
            ax.set_xlabel("Probability (%)")
            ax.set_title("Confidence", fontweight="bold")
            ax.set_xlim(0, 100)
            for bar, val in zip(bars, [probability[0]*100, probability[1]*100]):
                ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                        f"{val:.1f}%", va="center", fontsize=11)
            st.pyplot(fig)
            plt.close()

        # Risk factors
        st.markdown("### Key Risk Factors for This Customer")
        importances = pd.Series(model.feature_importances_, index=X.columns)
        top_factors = importances.sort_values(ascending=False).head(5)
        for feat, score in top_factors.items():
            val = input_df[feat].values[0] if feat in input_df.columns else "N/A"
            st.markdown(f"- **{feat.replace('_', ' ').title()}**: importance score `{score:.3f}`")

# ── PAGE: About ───────────────────────────────────────────────────────────────
elif page == "📋 About":
    st.markdown("## 📋 About This Project")

    st.markdown("""
    ### German Credit Risk Analyzer

    This project demonstrates an end-to-end machine learning pipeline for credit risk
    prediction, built using real German banking data.

    ### Data Source
    - **Repository:** UCI Machine Learning Repository
    - **Dataset:** Statlog (German Credit Data)
    - **Original collector:** Prof. Hans Hofmann, University of Hamburg
    - **Records:** 1,000 real German bank customers
    - **Features:** 20 financial and personal attributes
    - **Target:** Credit risk (Good = 1 / Bad = 2)
    - **Link:** [archive.ics.uci.edu](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)

    ### Tech Stack
    | Layer | Technology |
    |-------|-----------|
    | Language | Python 3.11 |
    | ML Model | Scikit-learn (Random Forest) |
    | UI | Streamlit |
    | Data | Pandas, NumPy |
    | Visualisation | Matplotlib, Seaborn |

    ### Model Details
    - **Algorithm:** Random Forest Classifier (100 trees, max depth 8)
    - **Train/Test Split:** 80% / 20%
    - **Class balancing:** Enabled (handles 70/30 class imbalance)
    - **Accuracy:** ~75–78% on test set

    ### Business Relevance
    Credit risk assessment is critical for German banks and financial institutions.
    Misclassifying a bad credit risk as good can result in significant loan defaults.
    This model helps automate the initial screening process while keeping human
    reviewers in the loop for borderline cases.

    ### Author
    Built as a portfolio project demonstrating financial data analytics and ML
    skills relevant to the German B2B market.
    """)
