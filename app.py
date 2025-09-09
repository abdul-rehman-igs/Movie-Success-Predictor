# app/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import io
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import base64
import os
from ast import literal_eval
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Movie Success Predictor 🎬",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for professional theme
st.markdown("""
    <style>
        body {
            color: #0d0d0d;
            background-color: #f9f9f9;
        }
        .stApp {
            background-color: #f9f9f9;
        }
        h1, h2, h3, h4 {
            color: #1a1a1a;
        }
        .css-18e3th9 {
            padding: 2rem;
        }
        .stSidebar {
            background-color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

# Optional shap (won't break if not installed)
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# Directories
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("models").mkdir(parents=True, exist_ok=True)
Path("reports/figures").mkdir(parents=True, exist_ok=True)

# ---- Page config & styling ----
st.set_page_config(
    page_title="Movie Success Predictor — Business Dashboard",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Utilities ----
def read_csv_bytes(u):
    try:
        return pd.read_csv(u, low_memory=False)
    except Exception:
        u.seek(0)
        return pd.read_csv(io.BytesIO(u.read()), low_memory=False)

def save_pickle(obj, path: str):
    joblib.dump(obj, path)

def load_pickle(path: str):
    return joblib.load(path)

def safe_literal_eval(x):
    try:
        return literal_eval(x)
    except Exception:
        return x

def parse_genres_field(x):
    # TMDB genres can be a JSON-like list-of-dicts string or simple string
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return [ (g.get('name') if isinstance(g, dict) else str(g)) for g in x ]
    if isinstance(x, str):
        # try parse json-like
        try:
            parsed = literal_eval(x)
            if isinstance(parsed, list):
                names = []
                for entry in parsed:
                    if isinstance(entry, dict) and 'name' in entry:
                        names.append(entry['name'])
                    else:
                        names.append(str(entry))
                return names
        except Exception:
            # fallback: split by comma
            return [s.strip() for s in x.split(',') if s.strip()]
    return []

# ---- Data processing & feature engineering ----
def load_tmdb(movies_csv_path: str = None):
    """Load either processed parquet (priority) or raw CSV uploaded path (if provided)."""
    processed_path = Path("data/processed/movies_processed.parquet")
    if processed_path.exists() and movies_csv_path is None:
        df = pd.read_parquet(processed_path)
        return df
    if movies_csv_path:
        # uploaded file-like object
        if isinstance(movies_csv_path, (str, Path)):
            df = pd.read_csv(movies_csv_path, low_memory=False)
        else:
            df = read_csv_bytes(movies_csv_path)
        return df
    # nothing to load
    return None

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize key columns."""
    df = df.copy()
    # normalize id/title
    if 'id' in df.columns:
        df['id'] = pd.to_numeric(df['id'], errors='coerce')
    if 'title' not in df.columns and 'original_title' in df.columns:
        df['title'] = df['original_title']

    # numeric fields
    for col in ['budget','revenue','runtime','popularity','vote_average','vote_count']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0

    # release_date
    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

    # genres parsing
    if 'genres' in df.columns:
        df['genres_parsed'] = df['genres'].apply(parse_genres_field)
    else:
        df['genres_parsed'] = [[] for _ in range(len(df))]

    # overview text
    if 'overview' not in df.columns:
        df['overview'] = ""

    # cast/crew columns might be present as json strings (if using merged credits)
    for col in ['cast','crew']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: safe_literal_eval(x) if isinstance(x, str) else x)

    # basic filters (keep some rows)
    # do not drop zeros here — we might want to predict for low-budget films
    return df

def extract_top_genres(df: pd.DataFrame, top_n=12):
    counts = {}
    for lst in df['genres_parsed']:
        for g in lst:
            counts[g] = counts.get(g, 0) + 1
    sorted_g = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    top = [g for g,_ in sorted_g[:top_n]]
    return top

def compute_cast_power_from_cast(col_cast):
    # crude score: position weighting and popularity if available
    def score(cast_list):
        if not isinstance(cast_list, list):
            return 0.0
        s = 0.0
        for i, m in enumerate(cast_list[:6]):
            if isinstance(m, dict):
                pop = m.get('popularity') or m.get('known_for_department') or 0
                s += (6 - i) * (float(pop) if isinstance(pop, (int,float)) else 1.0)
            else:
                s += (6 - i)
        return s
    return col_cast.apply(score)

def build_feature_matrix(df: pd.DataFrame, top_genres=None, use_text=True, tfidf_max_features=1500):
    df = df.copy()
    if top_genres is None:
        top_genres = extract_top_genres(df, top_n=10)
    # numeric base
    X_num = df[['budget','runtime','popularity','vote_count','vote_average']].fillna(0).copy()
    # cast power
    if 'cast' in df.columns:
        X_num['cast_power'] = compute_cast_power_from_cast(df['cast'])
    else:
        X_num['cast_power'] = 0

    # genre one-hot
    for g in top_genres:
        X_num[f'genre_{g}'] = df['genres_parsed'].apply(lambda lst: int(g in lst))

    # text features (tfidf)
    tfidf = None
    X_text = None
    if use_text:
        tfidf = TfidfVectorizer(max_features=tfidf_max_features, stop_words='english', ngram_range=(1,2))
        X_text = tfidf.fit_transform(df['overview'].fillna(''))
    return X_num, X_text, tfidf, top_genres

def combine_features(X_num, X_text):
    # Return a DataFrame-friendly X; if X_text present, keep separately for advanced pipelines
    if X_text is None:
        return X_num
    # convert sparse to dense for small datasets (caveat: can be big)
    try:
        X_text_dense = pd.DataFrame(X_text.toarray(), columns=[f"tf_{i}" for i in range(X_text.shape[1])])
        X_comb = pd.concat([X_num.reset_index(drop=True), X_text_dense.reset_index(drop=True)], axis=1)
        return X_comb
    except Exception:
        # fallback: only numeric
        return X_num

# ---- Model helpers ----
def train_quick_models(X, y, save_prefix="models/movie_model"):
    results = {}
    saved_models = {}
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42)

    # Linear Regression (baseline)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)
    results['linear'] = {'r2': r2_score(y_test, pred), 'rmse': np.sqrt(mean_squared_error(y_test, pred))}
    saved_models['linear'] = lr
    joblib.dump(lr, f"{save_prefix}_linear.joblib")

    # RandomForest
    rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    results['random_forest'] = {'r2': r2_score(y_test, pred), 'rmse': np.sqrt(mean_squared_error(y_test, pred))}
    saved_models['random_forest'] = rf
    joblib.dump(rf, f"{save_prefix}_rf.joblib")

    # XGBoost (if available)
    try:
        xgb = XGBRegressor(n_estimators=300, learning_rate=0.06, random_state=42, n_jobs=-1, verbosity=0)
        xgb.fit(X_train, y_train)
        pred = xgb.predict(X_test)
        results['xgboost'] = {'r2': r2_score(y_test, pred), 'rmse': np.sqrt(mean_squared_error(y_test, pred))}
        saved_models['xgboost'] = xgb
        joblib.dump(xgb, f"{save_prefix}_xgb.joblib")
    except Exception as e:
        results['xgboost'] = {'error': str(e)}

    # persist metrics & feature list
    Path("reports").mkdir(exist_ok=True)
    with open("reports/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    return results, saved_models

def load_best_model():
    # Prefer xgboost, then rf, then linear
    if Path("models/movie_model_xgb.joblib").exists():
        return joblib.load("models/movie_model_xgb.joblib")
    if Path("models/movie_model_xgb.joblib").exists():
        return joblib.load("models/movie_model_xgb.joblib")
    # check our training naming
    for name in ["models/movie_model_xgb.joblib", "models/movie_model_xgb.joblib",
                 "models/movie_model_xgboost.joblib", "models/movie_model_rf.joblib",
                 "models/movie_model_random_forest.joblib", "models/movie_model_linear.joblib",
                 "models/movie_model_linear.joblib"]:
        p = Path(name)
        if p.exists():
            return joblib.load(str(p))
    # fallback: find any joblib in models folder
    for f in Path("models").glob("*.joblib"):
        try:
            return joblib.load(f)
        except Exception:
            continue
    return None

# ---- Streamlit UI ----
st.sidebar.header("Movie Success Predictor")
mode = st.sidebar.selectbox("Navigation", ["Dashboard", "Upload Data", "EDA", "Train Models", "Predict", "Model Interpretability", "Export / Deploy"])

st.title("🎬 Movie Success Predictor — Business Dashboard")
st.markdown("A professional dashboard to predict box-office revenue using TMDB data. Use the side menu to navigate.")

# ---- Dashboard (KPIs) ----
if mode == "Dashboard":
    col1, col2, col3 = st.columns([1.5,1,1])
    df = None
    processed_path = Path("data/processed/movies_processed.parquet")
    if processed_path.exists():
        df = pd.read_parquet(processed_path)

    if df is None:
        st.info("No processed data found. Upload raw TMDB CSV in 'Upload Data' or place processed parquet in data/processed.")
    else:
        total = len(df)
        median_budget = int(df['budget'].median())
        median_revenue = int(df['revenue'].median())
        avg_runtime = round(df['runtime'].median(),1)
        col1.metric("Movies in dataset", total)
        col2.metric("Median Budget (USD)", f"${median_budget:,}")
        col3.metric("Median Revenue (USD)", f"${median_revenue:,}")
        st.markdown("### Quick visual — Budget vs Revenue (log scale)")
        sample = df[(df['budget']>0) & (df['revenue']>0)].sample(min(1500, max(1, len(df))))
        fig = px.scatter(sample, x="budget", y="revenue", hover_data=["title"], log_x=True, log_y=True,
                         labels={"budget":"Budget (USD)", "revenue":"Revenue (USD)"},
                         title="Budget vs Revenue (log scale)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Top genres")
        top_genres = extract_top_genres(df, top_n=12)
        genre_counts = {g: sum(df['genres_parsed'].apply(lambda lst: int(g in lst))) for g in top_genres}
        gdf = pd.DataFrame({"genre": list(genre_counts.keys()), "count": list(genre_counts.values())})
        fig2 = px.bar(gdf, x="genre", y="count", title="Top Genres by Count")
        st.plotly_chart(fig2, use_container_width=True)

# ---- Upload Data ----
elif mode == "Upload Data":
    st.header("Upload TMDB CSV (movies_metadata.csv) or credits CSV (optional)")
    uploaded = st.file_uploader("Upload movies CSV", type=['csv'])
    if uploaded is not None:
        st.info("Reading uploaded CSV... this may take a moment for large files.")
        df_raw = read_csv_bytes(uploaded)
        st.success(f"Loaded {len(df_raw):,} rows.")
        st.write("Preview:")
        st.dataframe(df_raw.head(5))

        if st.button("Save processed (basic) to data/processed/movies_processed.parquet"):
            st.info("Cleaning & saving processed dataset...")
            df_clean = basic_cleaning(df_raw)
            df_clean.to_parquet("data/processed/movies_processed.parquet", index=False)
            st.success("Saved to data/processed/movies_processed.parquet")
            st.experimental_rerun()

# ---- EDA ----
elif mode == "EDA":
    st.header("Exploratory Data Analysis")
    df = load_tmdb()
    if df is None:
        st.warning("No dataset available. Upload one in 'Upload Data'.")
    else:
        st.subheader("Dataset preview")
        st.dataframe(df[['title','budget','revenue','runtime','popularity']].head(8))

        st.subheader("Distributions")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df[df['budget']>0], x="budget", nbins=60, title="Budget distribution (USD)", log_y=True)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.histogram(df[df['revenue']>0], x="revenue", nbins=60, title="Revenue distribution (USD)", log_y=True)
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Interactive filters")
        genres_available = sorted({g for lst in df['genres_parsed'] for g in lst if g})
        chosen = st.multiselect("Filter by genres", genres_available, default=genres_available[:6])
        df_filtered = df[df['genres_parsed'].apply(lambda lst: any(g in lst for g in chosen))]
        st.write(f"Filtered to {len(df_filtered)} rows.")
        fig3 = px.scatter(df_filtered[df_filtered['budget']>0], x="budget", y="revenue", hover_data=['title'], log_x=True, log_y=True, title="Budget vs Revenue (filtered)")
        st.plotly_chart(fig3, use_container_width=True)

# ---- Train Models ----
elif mode == "Train Models":
    st.header("Train Models — quick & reproducible")
    st.markdown("Train three models (Linear, RandomForest, XGBoost). A quick training is provided for speed — tune hyperparams later for production.")
    df = load_tmdb()
    if df is None:
        st.warning("No dataset: upload one in 'Upload Data'.")
    else:
        use_text = st.checkbox("Include text features (overview) via TF-IDF (slower)", value=False)
        top_n_genres = st.slider("Top N genres to use for one-hot", 4, 20, 10)
        train_btn = st.button("Start training (quick)")
        if train_btn:
            st.info("Building features...")
            X_num, X_text, tfidf, top_genres = build_feature_matrix(df, top_genres=extract_top_genres(df, top_n=top_n_genres), use_text=use_text, tfidf_max_features=1000)
            if use_text:
                X = combine_features(X_num, X_text)
            else:
                X = X_num
            # target
            y = df['revenue'].fillna(0)
            # remove rows where all zeros maybe
            valid_idx = ~y.isna()
            X = X.loc[valid_idx].fillna(0)
            y = y.loc[valid_idx]
            st.info(f"Training on {X.shape[0]} examples with {X.shape[1]} features.")
            # scale numeric for linear
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            # train
            results, models_trained = train_quick_models(X, y, save_prefix="models/movie_model")
            # Save metadata
            metadata = {"top_genres": top_genres, "features": list(X.columns), "tfidf": None}
            if tfidf:
                save_pickle(tfidf, "models/tfidf.joblib")
                metadata["tfidf"] = "models/tfidf.joblib"
            save_pickle(metadata, "models/metadata.joblib")
            st.success("Training complete — models saved to /models/")
            st.json(results)
            # show brief feature importance (from RF or XGB)
            if 'xgboost' in models_trained:
                model_for_imp = models_trained.get('xgboost') or models_trained.get('random_forest')
            else:
                model_for_imp = models_trained.get('random_forest')
            if model_for_imp is not None:
                try:
                    fi = getattr(model_for_imp, "feature_importances_", None)
                    if fi is not None:
                        fi_df = pd.DataFrame({"feature": X.columns, "importance": fi}).sort_values("importance", ascending=False).head(25)
                        fig = px.bar(fi_df, x='importance', y='feature', orientation='h', title="Top feature importances")
                        st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass

# ---- Predict ----
elif mode == "Predict":
    st.header("Predict box-office revenue for a single movie or batch upload")
    # single prediction form
    with st.form("single_predict"):
        st.subheader("Single movie prediction")
        title = st.text_input("Title", value="Untitled")
        budget = st.number_input("Budget (USD)", min_value=0, value=10_000_000, step=100_000)
        runtime = st.number_input("Runtime (minutes)", min_value=0, value=110)
        popularity = st.number_input("Popularity score", min_value=0.0, max_value=1000.0, value=10.0)
        vote_count = st.number_input("Vote count", min_value=0, value=100)
        vote_average = st.number_input("Vote average (IMDb-like)", min_value=0.0, max_value=10.0, value=6.5)
        cast_power = st.number_input("Cast power (approx)", min_value=0.0, max_value=500.0, value=20.0)
        genres_input = st.multiselect("Genres", ["Drama","Comedy","Action","Thriller","Romance","Adventure","Sci-Fi","Horror","Animation"])
        submit = st.form_submit_button("Predict single movie")
    if submit:
        # build input row consistent with metadata
        meta_path = Path("models/metadata.joblib")
        if meta_path.exists():
            meta = load_pickle("models/metadata.joblib")
            features_expected = meta.get("features", [])
            top_genres = meta.get("top_genres", [])
        else:
            # fallback features
            top_genres = genres_input
            features_expected = ["budget","runtime","popularity","vote_count","vote_average","cast_power"] + [f"genre_{g}" for g in top_genres]
        # create row
        row = {}
        row['budget'] = budget
        row['runtime'] = runtime
        row['popularity'] = popularity
        row['vote_count'] = vote_count
        row['vote_average'] = vote_average
        row['cast_power'] = cast_power
        for g in top_genres:
            row[f"genre_{g}"] = int(g in genres_input)
        # ensure all expected columns present
        for c in features_expected:
            if c not in row:
                row[c] = 0
        X_row = pd.DataFrame([row], columns=features_expected)
        # load model
        model = load_best_model()
        if model is None:
            st.error("No trained model found. Train a model in 'Train Models' first.")
        else:
            try:
                pred = model.predict(X_row)[0]
                st.success(f"Predicted box-office revenue: ${pred:,.0f}")
                # simple ROI guidance
                roi = (pred - budget) / (budget + 1e-9)
                st.metric("Estimated ROI", f"{roi*100:.1f}%")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    st.markdown("---")
    st.subheader("Batch predictions (CSV upload)")
    uploaded = st.file_uploader("Upload CSV with columns (title,budget,runtime,popularity,vote_count,vote_average,genres)", type=['csv'])
    if uploaded:
        df_batch = read_csv_bytes(uploaded)
        st.write("Preview:", df_batch.head(3))
        if st.button("Run batch prediction"):
            # prepare df_batch features
            df_batch['genres_parsed'] = df_batch.get('genres', "").apply(parse_genres_field)
            meta_path = Path("models/metadata.joblib")
            if not meta_path.exists():
                st.error("No metadata/models available. Train first.")
            else:
                meta = load_pickle("models/metadata.joblib")
                features_expected = meta.get("features", [])
                top_genres = meta.get("top_genres", [])
                rows = []
                for _, r in df_batch.iterrows():
                    row = {}
                    row['budget'] = float(r.get('budget',0) or 0)
                    row['runtime'] = float(r.get('runtime',0) or 0)
                    row['popularity'] = float(r.get('popularity',0) or 0)
                    row['vote_count'] = float(r.get('vote_count',0) or 0)
                    row['vote_average'] = float(r.get('vote_average',0) or 0)
                    row['cast_power'] = float(r.get('cast_power',0) or 0)
                    for g in top_genres:
                        row[f"genre_{g}"] = int(g in r.get('genres_parsed', []))
                    # ensure columns
                    for c in features_expected:
                        if c not in row:
                            row[c] = 0
                    rows.append(row)
                Xp = pd.DataFrame(rows, columns=features_expected)
                model = load_best_model()
                preds = model.predict(Xp)
                df_batch['predicted_revenue'] = preds
                out_path = f"predictions/batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                Path("predictions").mkdir(exist_ok=True)
                df_batch.to_csv(out_path, index=False)
                st.success(f"Batch predictions saved to {out_path}")
                st.dataframe(df_batch[['title','predicted_revenue']].head(10))

# ---- Model Interpretability ----
elif mode == "Model Interpretability":
    st.header("Model Interpretability & Feature Importance")
    model = load_best_model()
    if model is None:
        st.warning("No model found — train models first.")
    else:
        st.subheader("Feature importance (model-provided)")
        # attempt to load metadata features list
        meta = None
        if Path("models/metadata.joblib").exists():
            meta = load_pickle("models/metadata.joblib")
        feat_names = None
        if meta:
            feat_names = meta.get("features", None)
        # try model.feature_importances_
        fi = getattr(model, "feature_importances_", None)
        if fi is not None and feat_names:
            fi_df = pd.DataFrame({"feature": feat_names, "importance": fi}).sort_values("importance", ascending=False).head(30)
            fig = px.bar(fi_df, x='importance', y='feature', orientation='h', title="Top 30 features")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Model does not expose feature_importances_ or metadata missing. Showing coefficients if linear model.")
            coef = getattr(model, "coef_", None)
            if coef is not None and feat_names:
                coef_df = pd.DataFrame({"feature": feat_names, "coef": coef}).sort_values("coef", ascending=False).head(30)
                fig = px.bar(coef_df, x='coef', y='feature', orientation='h', title="Top coefficients (Linear Model)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Unable to show feature importance. Make sure model and metadata are present.")

        if HAS_SHAP:
            st.subheader("SHAP explanations (local & global)")
            try:
                # use small sample for speed
                if meta and meta.get("features"):
                    feat_names = meta["features"]
                    # attempt to build a dataset from processed or recreate features using processed parquet
                    if Path("data/processed/movies_processed.parquet").exists():
                        dff = pd.read_parquet("data/processed/movies_processed.parquet")
                        X_num, X_text, tfidf, top_genres = build_feature_matrix(dff, top_genres=meta.get("top_genres", []), use_text=False)
                        X_use = combine_features(X_num, None)
                        X_sample = X_use.sample(min(300, len(X_use)), random_state=42)
                        explainer = shap.Explainer(model.predict, X_sample) if hasattr(shap, "Explainer") else shap.TreeExplainer(model)
                        shap_values = explainer(X_sample)
                        st.write("SHAP summary plot (may take a moment)...")
                        fig_shap = shap.plots.beeswarm(shap_values, show=False)
                        st.pyplot(bbox_inches='tight')
                    else:
                        st.info("Processed data not available to compute SHAP. Upload or create processed dataset first.")
            except Exception as e:
                st.error(f"SHAP computation failed: {e}")

# ---- Export / Deploy ----
elif mode == "Export / Deploy":
    st.header("Export, Demo & Deploy")
    st.markdown("""
    **Recommended next steps for production**
    - Containerize: create Dockerfile that installs Python deps and runs `streamlit run app/streamlit_app.py`.
    - Experiment tracking: add MLflow to log experiments and models.
    - CI: add tests for data processing and model sanity checks.
    - Auth & sharing: add simple auth via Streamlit sharing or put behind a small FastAPI with auth if needed.
    """)
    if Path("models/metadata.joblib").exists():
        st.success("models/metadata.joblib found")
    else:
        st.info("No metadata found — train a model to generate metadata.")

    st.markdown("### Download trained model (if exists)")
    for p in Path("models").glob("*.joblib"):
        with open(p, "rb") as f:
            b = f.read()
        b64 = base64.b64encode(b).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{p.name}">Download {p.name}</a>'
        st.markdown(href, unsafe_allow_html=True)

# ---- default footer ----
st.markdown("---")
st.markdown("Built with ❤️ for a professional, investor-ready workflow.  \nTips: upload the TMDB CSV in Upload Data → Train Models → Predict.")
