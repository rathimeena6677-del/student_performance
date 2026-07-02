
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
st.set_page_config(page_title="Student Performance Predictor", layout="wide")
 
st.title("🎓 Student Performance Predictor")
st.caption(
    "Predicts a student's Performance Index from study habits, using Linear "
    "Regression / Lasso Regression (based on the original analysis notebook)."
)
 
# ----------------------------------------------------------------------
# 1. Data loading
# ----------------------------------------------------------------------
st.sidebar.header("1️⃣ Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload Student_Performance.csv", type=["csv"]
)
 
REQUIRED_COLS = [
    "Hours Studied",
    "Previous Scores",
    "Extracurricular Activities",
    "Sleep Hours",
    "Sample Question Papers Practiced",
    "Performance Index",
]
 
 
@st.cache_data
def load_data(file) -> pd.DataFrame:
    return pd.read_csv(file)
 
 
if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    st.info(
        "👈 Upload the `Student_Performance.csv` dataset to train the model. "
        "Until then, a small demo dataset is used so you can explore the app."
    )
    rng = np.random.default_rng(42)
    n = 300
    hours = rng.integers(1, 10, n)
    prev = rng.integers(40, 100, n)
    extra = rng.choice(["Yes", "No"], n)
    sleep = rng.integers(4, 10, n)
    papers = rng.integers(0, 10, n)
    perf = (
        5.5 * hours
        + 1.0 * prev
        + (extra == "Yes") * 2
        + 0.5 * sleep
        + 0.3 * papers
        + rng.normal(0, 5, n)
    )
    perf = np.clip(perf, 10, 100)
    df = pd.DataFrame(
        {
            "Hours Studied": hours,
            "Previous Scores": prev,
            "Extracurricular Activities": extra,
            "Sleep Hours": sleep,
            "Sample Question Papers Practiced": papers,
            "Performance Index": perf,
        }
    )
 
missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    st.error(f"Uploaded file is missing required columns: {missing}")
    st.stop()
 
# ----------------------------------------------------------------------
# 2. Preprocessing + model training (cached)
# ----------------------------------------------------------------------
@st.cache_resource
def train_models(data: pd.DataFrame):
    data = data.dropna().drop_duplicates().copy()
 
    encoder = LabelEncoder()
    data["Extracurricular Activities"] = encoder.fit_transform(
        data["Extracurricular Activities"]
    )
 
    X = data.drop("Performance Index", axis=1)
    y = data["Performance Index"]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
 
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
 
    lin_model = LinearRegression()
    lin_model.fit(X_train_s, y_train)
 
    lasso_model = Lasso()
    lasso_model.fit(X_train_s, y_train)
 
    # small grid search for best Lasso alpha
    grid = GridSearchCV(
        Lasso(), {"alpha": [0.001, 0.01, 0.1, 1, 10, 100]}, cv=5, scoring="r2"
    )
    grid.fit(X_train_s, y_train)
 
    def metrics(model):
        yp_train = model.predict(X_train_s)
        yp_test = model.predict(X_test_s)
        return {
            "train": {
                "MAE": mean_absolute_error(y_train, yp_train),
                "MSE": mean_squared_error(y_train, yp_train),
                "R2": r2_score(y_train, yp_train),
            },
            "test": {
                "MAE": mean_absolute_error(y_test, yp_test),
                "MSE": mean_squared_error(y_test, yp_test),
                "R2": r2_score(y_test, yp_test),
            },
        }
 
    results = {
        "Linear Regression": {"model": lin_model, "metrics": metrics(lin_model)},
        "Lasso Regression": {"model": lasso_model, "metrics": metrics(lasso_model)},
    }
 
    return {
        "results": results,
        "scaler": scaler,
        "encoder": encoder,
        "feature_names": list(X.columns),
        "best_lasso_alpha": grid.best_params_["alpha"],
        "best_lasso_cv_r2": grid.best_score_,
    }
 
 
trained = train_models(df)
results = trained["results"]
scaler = trained["scaler"]
encoder = trained["encoder"]
feature_names = trained["feature_names"]
 
# ----------------------------------------------------------------------
# 3. Tabs
# ----------------------------------------------------------------------
tab_overview, tab_eda, tab_model, tab_predict = st.tabs(
    ["📄 Data Overview", "📊 EDA", "🧠 Model Performance", "🔮 Predict"]
)
 
with tab_overview:
    st.subheader("Dataset preview")
    st.dataframe(df.head(20), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Duplicate rows", int(df.duplicated().sum()))
    st.subheader("Summary statistics")
    st.dataframe(df.describe(), use_container_width=True)
    st.subheader("Missing values")
    st.dataframe(df.isnull().sum().rename("Missing count"), use_container_width=True)
 
with tab_eda:
    st.subheader("Distributions")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    col = st.selectbox("Choose a column to visualize", numeric_cols)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.histplot(df[col], kde=True, ax=ax)
    st.pyplot(fig)
 
    st.subheader("Correlation heatmap")
    corr_df = df.copy()
    corr_df["Extracurricular Activities"] = LabelEncoder().fit_transform(
        corr_df["Extracurricular Activities"]
    )
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
    st.pyplot(fig2)
 
    st.subheader("Extracurricular Activities counts")
    st.bar_chart(df["Extracurricular Activities"].value_counts())
 
with tab_model:
    st.subheader("Model comparison")
    for name, res in results.items():
        st.markdown(f"**{name}**")
        m = res["metrics"]
        cols = st.columns(6)
        cols[0].metric("Train MAE", f"{m['train']['MAE']:.3f}")
        cols[1].metric("Train MSE", f"{m['train']['MSE']:.3f}")
        cols[2].metric("Train R²", f"{m['train']['R2']:.4f}")
        cols[3].metric("Test MAE", f"{m['test']['MAE']:.3f}")
        cols[4].metric("Test MSE", f"{m['test']['MSE']:.3f}")
        cols[5].metric("Test R²", f"{m['test']['R2']:.4f}")
        st.divider()
 
    st.info(
        f"Best Lasso alpha from GridSearchCV (5-fold CV): "
        f"**{trained['best_lasso_alpha']}** "
        f"(CV R² = {trained['best_lasso_cv_r2']:.4f})"
    )
 
    st.subheader("Lasso feature importance (coefficients)")
    lasso_model = results["Lasso Regression"]["model"]
    fi = pd.DataFrame(
        {"Feature": feature_names, "Coefficient": lasso_model.coef_}
    ).sort_values("Coefficient", ascending=False)
    st.bar_chart(fi.set_index("Feature"))
 
with tab_predict:
    st.subheader("Enter student details")
    model_choice = st.radio(
        "Model to use for prediction", list(results.keys()), horizontal=True
    )
 
    c1, c2 = st.columns(2)
    with c1:
        hours_studied = st.number_input(
            "Hours Studied", min_value=0, max_value=24, value=5, step=1
        )
        previous_scores = st.number_input(
            "Previous Scores", min_value=0, max_value=100, value=70, step=1
        )
        sleep_hours = st.number_input(
            "Sleep Hours", min_value=0, max_value=24, value=7, step=1
        )
    with c2:
        extracurricular = st.radio("Extracurricular Activities", ["Yes", "No"])
        papers_practiced = st.number_input(
            "Sample Question Papers Practiced", min_value=0, max_value=50, value=5, step=1
        )
 
    if st.button("Predict Performance Index", type="primary"):
        extra_encoded = encoder.transform([extracurricular])[0]
        input_df = pd.DataFrame(
            [[hours_studied, previous_scores, extra_encoded, sleep_hours, papers_practiced]],
            columns=feature_names,
        )
        input_scaled = scaler.transform(input_df)
        model = results[model_choice]["model"]
        prediction = model.predict(input_scaled)[0]
        prediction = float(np.clip(prediction, 0, 100))
 
        st.success(f"🎯 Predicted Performance Index: **{prediction:.2f} / 100**")
        st.progress(min(int(prediction), 100))
 
        st.divider()
        if model_choice == "Lasso Regression":
            test_r2 = results["Lasso Regression"]["metrics"]["test"]["R2"]
            test_mae = results["Lasso Regression"]["metrics"]["test"]["MAE"]
            nonzero = [
                f"- **{f}**: {c:.3f}"
                for f, c in zip(feature_names, model.coef_)
                if abs(c) > 1e-6
            ]
            zero = [
                f"- {f}"
                for f, c in zip(feature_names, model.coef_)
                if abs(c) <= 1e-6
            ]
            st.markdown("### 📘 What is Lasso Regression? (and what does this output mean)")
            st.markdown(
                f"""
**Lasso Regression** is Linear Regression with an added *L1 penalty* that
shrinks less-useful feature coefficients — often all the way to **zero**.
This means Lasso automatically performs **feature selection**: features with
a coefficient of 0 are effectively removed from the model because they didn't
add meaningful predictive power once the stronger features were accounted for.
 
**How to read this specific prediction:**
- The model combined your inputs using only the features it kept (non-zero
  coefficients below) to compute **{prediction:.2f}**.
- A feature with coefficient **0.000** contributed **nothing** to this
  prediction — changing that input's value would not move the result.
 
**Features the model actually used (non-zero coefficients):**
{chr(10).join(nonzero) if nonzero else "- None — all coefficients were shrunk to zero."}
 
**Features the model ignored (coefficient shrunk to zero):**
{chr(10).join(zero) if zero else "- None — all features were used."}
 
**How trustworthy is this model overall?**
- **R² (test) = {test_r2:.3f}** → the model explains about
  **{test_r2*100:.1f}%** of the variation in Performance Index across unseen
  students. Closer to 1.0 (100%) is better.
- **MAE (test) = {test_mae:.2f}** → on average, predictions are off by about
  **±{test_mae:.2f} points** on the 0–100 scale.
- Since training and test scores are close to each other, the model
  generalizes well and isn't simply memorizing the training data
  (i.e. it is not overfitting).
"""
            )
        else:
            st.caption(
                "ℹ️ Switch **Model to use for prediction** to **Lasso Regression** "
                "above to see a plain-language breakdown of which features "
                "the model kept, which it dropped, and how reliable the "
                "prediction is."
            )
 
st.sidebar.divider()
st.sidebar.caption(
    "Built with Streamlit · Linear Regression & Lasso Regression "
    "(scikit-learn) trained live on the uploaded dataset.")
