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
import matplotlib.pyplot as plt
import seaborn as sns
import credit_model as cm

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

# ── Load data and train models (logic lives in credit_model.py) ─────────────
@st.cache_data
def load_data():
    try:
        return cm.load_data()
    except RuntimeError as e:
        return None, str(e)


@st.cache_resource
def train_models(df):
    return cm.train(df)


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
df, data_source = load_data()

if df is None:
    st.error(f"Could not load the dataset: {data_source}")
    uploaded = st.file_uploader("Upload the UCI german.data file", type=["data", "csv", "txt"])
    if uploaded:
        df = pd.read_csv(uploaded, sep=" ", header=None, names=cm.COLUMNS)
        cm.validate_raw(df)
    else:
        st.stop()

results = train_models(df)
rf, baseline = results["rf"], results["baseline"]
m = rf.metrics
X = df.drop(columns=cm.TARGET)

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
        st.metric("ROC-AUC (test set)", f"{m['roc_auc']:.2f}")

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
        3. Train a Random Forest and a logistic-regression baseline
        4. Evaluate with ROC-AUC, bad-loan recall and 5-fold cross-validation
        5. Choose the decision threshold that minimises the cost of wrong decisions
        6. Accept new inputs and predict risk in real time
        """)

    with col_r:
        st.markdown("### Credit Risk Distribution")
        fig, ax = plt.subplots(figsize=(5, 3))
        counts = df[cm.TARGET].map({1: "Good Credit", 2: "Bad Credit"}).value_counts()
        labels = [f"{k} ({v / len(df):.0%})" for k, v in counts.items()]
        sizes = counts.values
        colors = ["#0A9396", "#E63946"]
        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.0f%%",
               startangle=90, textprops={"fontsize": 11})
        ax.axis("equal")
        st.pyplot(fig)
        plt.close()

# ── PAGE: Data Explorer ───────────────────────────────────────────────────────
elif page == "📊 Data Explorer":
    st.markdown("## 📊 Data Explorer")
    st.caption(f"UCI Statlog German Credit Dataset — {len(df):,} customers, {X.shape[1]} features")

    tab1, tab2, tab3 = st.tabs(["Raw Data", "Statistics", "Distributions"])

    with tab1:
        st.dataframe(df.head(50), width="stretch")
        st.caption(f"Showing 50 of {len(df):,} rows · {df.shape[1]} columns")

    with tab2:
        st.dataframe(df.describe().round(2), width="stretch")

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
    st.caption(
        "Random Forest vs logistic-regression baseline · stratified 80/20 split · "
        "5-fold cross-validation · cost-based decision threshold"
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC (test)", f"{m['roc_auc']:.2f}", help="How well the model ranks bad loans above good ones")
    col2.metric("Bad-loan recall", f"{m['bad_recall']:.0%}", help="Share of actual bad loans the model catches")
    col3.metric("Bad-loan precision", f"{m['bad_precision']:.0%}")
    col4.metric("CV ROC-AUC", f"{m['cv_auc_mean']:.2f} ± {m['cv_auc_std']:.2f}")

    st.info(
        f"Missing a bad loan costs 5× more than rejecting a good customer (UCI cost matrix), "
        f"so the model flags a customer as bad when P(bad) ≥ **{rf.threshold:.2f}** instead of 0.50. "
        f"This trades some accuracy ({m['accuracy']:.0%}) for catching more defaults."
    )

    st.markdown("### Random Forest vs baseline")
    comp = pd.DataFrame({
        name: {
            "ROC-AUC (test)": f"{mod.metrics['roc_auc']:.3f}",
            "CV ROC-AUC": f"{mod.metrics['cv_auc_mean']:.3f} ± {mod.metrics['cv_auc_std']:.3f}",
            "Bad-loan recall": f"{mod.metrics['bad_recall']:.0%}",
            "Bad-loan precision": f"{mod.metrics['bad_precision']:.0%}",
            "Accuracy": f"{mod.metrics['accuracy']:.0%}",
            "Expected cost / customer": f"{mod.metrics['expected_cost']:.3f}",
            "Threshold": f"{mod.threshold:.2f}",
        }
        for name, mod in (("Random Forest", rf), ("Logistic regression (baseline)", baseline))
    })
    st.dataframe(comp, width="stretch")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### Confusion Matrix (test set)")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            m["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Good Credit", "Bad Credit"],
            yticklabels=["Good Credit", "Bad Credit"]
        )
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        st.pyplot(fig)
        plt.close()

    with col_r:
        st.markdown("### Feature Importance (Top 10)")
        importances = rf.feature_importances.head(10).sort_values()
        fig, ax = plt.subplots(figsize=(5, 4))
        importances.plot(kind="barh", ax=ax, color="#0A9396")
        ax.set_xlabel("Importance Score")
        st.pyplot(fig)
        plt.close()

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

        submitted = st.form_submit_button("🔮 Predict Credit Risk", width="stretch")

    if submitted:
        # Build input row matching training feature order
        # Using median/mode values for features not collected in form
        input_data = {
            "checking_account": [checking_account.split(" ")[0]],
            "duration": [duration],
            "credit_history": [credit_history.split(" ")[0]],
            "purpose": [purpose.split(" ")[0]],
            "credit_amount": [credit_amount],
            "savings_account": [savings_account.split(" ")[0]],
            "employment": [employment.split(" ")[0]],
            "installment_rate": [installment_rate],
            "personal_status": ["A93"],       # male single (most common)
            "other_debtors": ["A101"],         # none
            "residence_since": [3],
            "property": ["A121"],              # real estate
            "age": [age],
            "other_installments": ["A143"],    # none
            "housing": [housing.split(" ")[0]],
            "existing_credits": [existing_credits],
            "job": [job.split(" ")[0]],
            "liable_people": [liable_people],
            "telephone": ["A192"],             # yes
            "foreign_worker": ["A201"],        # yes
        }

        input_df = pd.DataFrame(input_data)

        p_bad = float(rf.predict_proba_bad(input_df)[0])
        prediction = int(p_bad >= rf.threshold)
        probability = [1 - p_bad, p_bad]

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
        st.caption(f"Flagged as bad when P(bad) ≥ {rf.threshold:.2f} (cost-based threshold).")
        st.markdown("### Most influential features overall")
        st.caption("These are global model importances, not an explanation of this single prediction.")
        for feat, score in rf.feature_importances.head(5).items():
            st.markdown(f"- **{feat.replace('_', ' ').title()}**: importance `{score:.3f}`")

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
    - **Algorithm:** Random Forest (300 trees, max depth 8), compared with a logistic-regression baseline
    - **Validation:** stratified 80/20 split plus 5-fold cross-validation on the training set
    - **Class balancing:** enabled (the data is 70% good / 30% bad)
    - **Decision threshold:** tuned on out-of-fold training predictions to minimise the UCI cost
      matrix (a missed bad loan costs 5, a rejected good customer costs 1)
    - **Results:** shown live on the ML Model page

    ### Business Relevance
    Credit risk assessment is critical for German banks and financial institutions.
    Misclassifying a bad credit risk as good can result in significant loan defaults.
    This model helps automate the initial screening process while keeping human
    reviewers in the loop for borderline cases.

    ### Author
    Built by Kartikeya Tripathi as a portfolio project in financial data analytics and ML.
    """)
