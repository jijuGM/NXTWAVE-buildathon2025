# app.py
'''import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import sklearn
from packaging import version

st.set_page_config(page_title="Carbon Emission - Simple App", layout="wide")
st.title("🌱 Carbon Emission Predictor (with automatic encoding)")

DATA_FILE = "Carbon Emission.csv"
PICKLE_FILE = "carbon_model.pkl"

# -- helper for OneHotEncoder compatibility --
def make_onehot(**kwargs):
    if version.parse(sklearn.__version__) >= version.parse("1.2"):
        return OneHotEncoder(sparse_output=False, **kwargs)
    else:
        return OneHotEncoder(sparse=False, **kwargs)

st.write("This app will train a model from your CSV (must include target column 'CarbonEmission'), encode text columns automatically, save the model as a pickle, and let you predict from the sidebar.")

# -------------------------
# 1) Load CSV
# -------------------------
if not os.path.exists(DATA_FILE):
    st.error(f"Place your dataset as '{DATA_FILE}' in this folder (CSV).")
    st.stop()

df = pd.read_csv(DATA_FILE)
st.subheader("Data preview")
st.dataframe(df.head())

if "CarbonEmission" not in df.columns:
    st.error("CSV must contain target column named 'CarbonEmission'. Rename the target column or update the code.")
    st.stop()

# -------------------------
# 2) Simple automatic column detection
# -------------------------
X = df.drop("CarbonEmission", axis=1)
y = df["CarbonEmission"]

# Treat columns with numeric dtype as numeric; everything else as categorical
numeric_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=["int64","float64"]).columns.tolist()

st.write(f"Detected numeric columns: {numeric_cols}")
st.write(f"Detected categorical columns: {categorical_cols}")

# -------------------------
# 3) Build preprocessing + model pipeline
# -------------------------
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", make_onehot(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, numeric_cols),
    ("cat", cat_pipeline, categorical_cols)
], remainder="drop")

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# -------------------------
# 4) Train button (trains and saves pickle)
# -------------------------
if st.button("🚀 Train model and save pickle"):
    with st.spinner("Training..."):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, preds)

        # save pipeline (includes encoder + regressor)
        with open(PICKLE_FILE, "wb") as f:
            pickle.dump(pipeline, f)

    st.success(f"Model trained. Test MSE: {mse:.3f}")
    st.info(f"Saved pipeline to {PICKLE_FILE}")

# -------------------------
# 5) Load model for prediction
# -------------------------
st.sidebar.header("Predict with trained model")
uploaded_model = st.sidebar.file_uploader("Or upload a trained pipeline (.pkl)", type=["pkl"])
model = None

if uploaded_model is not None:
    try:
        model = pickle.load(uploaded_model)
        st.sidebar.success("Loaded uploaded model.")
    except Exception as e:
        st.sidebar.error(f"Failed to load uploaded model: {e}")

elif os.path.exists(PICKLE_FILE):
    try:
        with open(PICKLE_FILE, "rb") as f:
            model = pickle.load(f)
        st.sidebar.success(f"Loaded local model: {PICKLE_FILE}")
    except Exception as e:
        st.sidebar.error(f"Failed to load local model: {e}")

# -------------------------
# 6) Sidebar input widgets — auto-generate sensible widgets from detected columns
# -------------------------
st.sidebar.subheader("Enter your details")

def make_widget(col_name, sample_series):
    # numeric => number_input, categorical => selectbox with unique values (or text input)
    if pd.api.types.is_numeric_dtype(sample_series):
        # provide default as median
        default = float(sample_series.median() if not sample_series.dropna().empty else 0)
        return st.sidebar.number_input(col_name, value=default)
    else:
        # take top unique values as choices if not too many
        uniques = sample_series.dropna().unique().tolist()
        if 1 < len(uniques) <= 50:
            # use selectbox for convenience
            default = uniques[0]
            return st.sidebar.selectbox(col_name, uniques, index=0)
        else:
            # fallback to text input
            return st.sidebar.text_input(col_name, value="")

# build input dict in same column order as X
input_dict = {}
for col in X.columns:
    input_dict[col] = make_widget(col, X[col])

input_df = pd.DataFrame([input_dict])
st.sidebar.write("Preview of input:")
st.sidebar.write(input_df.T.rename(columns={0: "value"}))

# Ensure numeric columns are numeric
for c in numeric_cols:
    if c in input_df.columns:
        input_df[c] = pd.to_numeric(input_df[c], errors="coerce")

# -------------------------
# 7) Predict
# -------------------------
if st.sidebar.button("🌍 Predict Carbon Emission"):
    if model is None:
        st.sidebar.error("No trained model available. Train or upload a pipeline .pkl file first.")
    else:
        try:
            pred = model.predict(input_df)[0]
            st.success(f"Estimated Carbon Emission: **{pred:.2f} kg CO₂/year**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.write("Input dataframe (for debugging):")
            st.write(input_df)

st.markdown("---")
st.caption("This simple app automatically encodes text columns and uses a pipeline so predictions won't fail on string categories like 'overweight'.")'''
# app_clean.py
import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import sklearn
from packaging import version

# ---------- Config ----------
DATA_FILE = "Carbon Emission.csv"
PICKLE_FILE = "carbon_model.pkl"

st.set_page_config(page_title="Carbon Footprint esitimator", layout="centered")
st.title("🌿 Carbon Footprint esitimator")
st.markdown("Enter a few simple details and click **Estimate** — quick, friendly, and private.")

# ---------- helpers ----------
def make_onehot(**kwargs):
    if version.parse(sklearn.__version__) >= version.parse("1.2"):
        return OneHotEncoder(sparse_output=False, **kwargs)
    else:
        return OneHotEncoder(sparse=False, **kwargs)

def build_pipeline_from_df(df):
    X = df.drop("CarbonEmission", axis=1)
    y = df["CarbonEmission"]
    numeric_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="missing")), ("onehot", make_onehot(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", num_pipe, numeric_cols), ("cat", cat_pipe, categorical_cols)], remainder="drop")
    pipe = Pipeline([("preprocessor", pre), ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)
    return pipe, mean_squared_error(y_test, pipe.predict(X_test))

# ---------- Load data quietly (used to build UI widgets) ----------
if os.path.exists(DATA_FILE):
    try:
        df = pd.read_csv(DATA_FILE)
        if "CarbonEmission" not in df.columns:
            df = None
    except Exception:
        df = None
else:
    df = None

# ---------- Admin: training/upload (collapsed by default) ----------
with st.expander("👩‍💻 Admin (train / upload model)", expanded=False):
    st.write("Training and upload tools. Customers won't see this unless expanded.")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Train model from CSV and save .pkl"):
            if df is None:
                st.error(f"Cannot find usable '{DATA_FILE}'. Make sure the CSV is present and has a 'CarbonEmission' column.")
            else:
                with st.spinner("Training..."):
                    pipeline, mse = build_pipeline_from_df(df)
                    try:
                        with open(PICKLE_FILE, "wb") as f:
                            pickle.dump(pipeline, f)
                        st.success(f"Model trained and saved. Test MSE: {mse:.3f}")
                    except Exception as e:
                        st.error(f"Saved training failed: {e}")

    with col2:
        uploaded_model = st.file_uploader("Or upload a trained pipeline (.pkl)", type=["pkl"])
        if uploaded_model is not None:
            try:
                temp = pickle.load(uploaded_model)
                # save uploaded model to local file for app use
                with open(PICKLE_FILE, "wb") as f:
                    pickle.dump(temp, f)
                st.success("Uploaded model saved for app use.")
            except Exception as e:
                st.error(f"Failed to load uploaded model: {e}")

    st.markdown("---")
    st.caption("Admin area: use this only if you want to retrain or replace the model.")

# ---------- Load model for prediction (silent success/fail) ----------
model = None
if os.path.exists(PICKLE_FILE):
    try:
        with open(PICKLE_FILE, "rb") as f:
            model = pickle.load(f)
    except Exception:
        model = None

# ---------- Build minimal sidebar UI for customer ----------
st.sidebar.header("Tell us about you")
st.sidebar.markdown("Just a few easy choices — casual answers are fine.")

# If we have df, use its columns to build widgets; otherwise show a small default form.
def make_widget(col_name, sample_series):
    if pd.api.types.is_numeric_dtype(sample_series):
        default = float(sample_series.median() if not sample_series.dropna().empty else 0)
        return st.sidebar.number_input(col_name, value=default)
    else:
        uniques = sample_series.dropna().unique().tolist()
        if 1 < len(uniques) <= 50:
            return st.sidebar.selectbox(col_name, uniques)
        else:
            return st.sidebar.text_input(col_name, value="")

if df is not None:
    X_cols = [c for c in df.columns if c != "CarbonEmission"]
    input_vals = {}
    for c in X_cols:
        input_vals[c] = make_widget(c, df[c])
else:
    # Fallback small form — friendly and short
    input_vals = {
        "Transport": st.sidebar.selectbox("Main transport", ["car","bike","bus","train","walk"]),
        "Vehicle": st.sidebar.selectbox("Do you own a vehicle?", ["yes","no"]),
        "Vehicle Distance": st.sidebar.number_input("Vehicle km/week", value=50, min_value=0),
        "Flights": st.sidebar.selectbox("Flights per year", ["never","rarely","sometimes","often"]),
        "Grocery": st.sidebar.selectbox("Grocery type", ["mixed","organic","non-organic"]),
        "Waste Weekly": st.sidebar.number_input("Waste kg/week", value=5, min_value=0),
        "Energy Eff": st.sidebar.selectbox("Home energy efficiency", ["high","medium","low"])
    }

# Turn into DataFrame with correct column order (if df available)
input_df = pd.DataFrame([input_vals])
if df is not None:
    # ensure same column order
    input_df = input_df[[c for c in df.columns if c != "CarbonEmission"]]

# Ensure numeric columns are numeric
if df is not None:
    numeric_cols = df.drop("CarbonEmission", axis=1).select_dtypes(include=["int64","float64"]).columns.tolist()
    for c in numeric_cols:
        if c in input_df.columns:
            input_df[c] = pd.to_numeric(input_df[c], errors="coerce")

# ---------- Big friendly Predict button ----------
st.markdown("")
if st.button("🔮 Estimate my carbon"):
    if model is None:
        # friendly fallback heuristic if no model available
        transport = input_vals.get("Transport", "")
        vehicle = input_vals.get("Vehicle", "no")
        vehicle_distance = float(input_vals.get("Vehicle Distance", 0))
        flights = input_vals.get("Flights", "never") or input_vals.get("Flight", "never")
        grocery = input_vals.get("Grocery", "mixed")
        waste = float(input_vals.get("Waste Weekly", 0))
        score = 5.0
        if transport == "car": score += 5
        if vehicle == "yes": score += 4
        score += (vehicle_distance / 100) * 2
        if flights == "often": score += 20
        if grocery == "non-organic": score += 3
        score += (waste / 10) * 2
        est = max(1.0, score * 2.5)
        st.success(f"Estimated Carbon Emission: **{est:.1f} kg CO₂/year**")
        st.info("Quick heuristic estimate — upload or train a model for a data-driven result (Admin).")
    else:
        try:
            pred = model.predict(input_df)[0]
            st.success(f"Estimated Carbon Emission: **{pred:.2f} kg CO₂/year**")
        except Exception as e:
            st.error("Prediction failed. Please contact the administrator or try training/uploading a model in Admin.")
            # optional: show input for debugging (hidden by default)
            with st.expander("Show input (for debugging)", expanded=False):
                st.write(input_df)

# ---------- Footer ----------
st.markdown("---")
st.caption("Privacy-friendly: your inputs stay in your browser/session.")
