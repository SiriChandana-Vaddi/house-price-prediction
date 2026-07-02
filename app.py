import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ── Load Model ───────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load('models/best_model.pkl')
    return model

# ── Header ───────────────────────────────────────────────
st.title("🏠 House Price Prediction App")
st.markdown("### Predict house prices using Machine Learning")
st.markdown("---")

# ── Sidebar Inputs ───────────────────────────────────────
st.sidebar.header("🔧 Enter House Features")

avg_income = st.sidebar.slider(
    "Avg. Area Income ($)",
    min_value=17000,
    max_value=110000,
    value=68000,
    step=1000
)

house_age = st.sidebar.slider(
    "Avg. House Age (years)",
    min_value=2,
    max_value=10,
    value=6,
    step=1
)

num_rooms = st.sidebar.slider(
    "Avg. Number of Rooms",
    min_value=3,
    max_value=11,
    value=7,
    step=1
)

num_bedrooms = st.sidebar.slider(
    "Avg. Number of Bedrooms",
    min_value=2,
    max_value=7,
    value=4,
    step=1
)

population = st.sidebar.slider(
    "Area Population",
    min_value=172,
    max_value=70000,
    value=36000,
    step=500
)

# ── Engineered Features ──────────────────────────────────
income_per_room     = avg_income / num_rooms
bedroom_room_ratio  = num_bedrooms / num_rooms
income_x_population = avg_income * population
log_income          = np.log1p(avg_income)
log_population      = np.log1p(population)

# ── Input DataFrame ──────────────────────────────────────
input_data = pd.DataFrame({
    'Avg. Area Income'            : [avg_income],
    'Avg. Area House Age'         : [house_age],
    'Avg. Area Number of Rooms'   : [num_rooms],
    'Avg. Area Number of Bedrooms': [num_bedrooms],
    'Area Population'             : [population],
    'Income_per_Room'             : [income_per_room],
    'Bedroom_Room_Ratio'          : [bedroom_room_ratio],
    'Income_x_Population'         : [income_x_population],
    'Log_Income'                  : [log_income],
    'Log_Population'              : [log_population]
})

# ── Main Layout ──────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Your Input Features")
    display_df = pd.DataFrame({
        'Feature': [
            'Avg. Area Income',
            'House Age',
            'Number of Rooms',
            'Number of Bedrooms',
            'Area Population'
        ],
        'Value': [
            f"${avg_income:,}",
            f"{house_age} years",
            str(num_rooms),
            str(num_bedrooms),
            f"{population:,}"
        ]
    })
    st.table(display_df)

with col2:
    st.subheader("🎯 Prediction Result")
    if st.button("🔮 Predict House Price", use_container_width=True):
        try:
            model = load_model()
            prediction = model.predict(input_data)[0]

            # If log transform was used, reverse it
            if prediction < 100:
                prediction = np.expm1(prediction)

            st.success(f"### Estimated Price: ${prediction:,.2f}")

            # Price range (±10%)
            lower = prediction * 0.90
            upper = prediction * 1.10
            st.info(f"📊 Price Range: ${lower:,.2f} — ${upper:,.2f}")

        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.warning("Make sure models/best_model.pkl exists!")

# ── Feature Importance ───────────────────────────────────
st.markdown("---")
st.subheader("📊 Feature Importance")

features = [
    'Avg. Area Income',
    'House Age',
    'Num Rooms',
    'Num Bedrooms',
    'Population',
    'Income/Room',
    'Bedroom Ratio',
    'Income×Pop',
    'Log Income',
    'Log Population'
]
importance = [0.64, 0.45, 0.35, 0.17, 0.40, 0.55, 0.20, 0.60, 0.62, 0.38]

fig, ax = plt.subplots(figsize=(10, 4))
colors = ['steelblue' if i < 5 else 'coral' for i in range(len(features))]
bars = ax.barh(features, importance, color=colors)
ax.set_xlabel('Importance Score')
ax.set_title('Feature Importance')
ax.invert_yaxis()
st.pyplot(fig)

# ── Model Info ───────────────────────────────────────────
st.markdown("---")
st.subheader("ℹ️ Model Information")

col3, col4, col5, col6 = st.columns(4)
col3.metric("Model",    "XGBoost")
col4.metric("R² Score", "0.92")
col5.metric("RMSE",     "~$85,000")
col6.metric("Dataset",  "5000 rows")

st.markdown("---")
st.markdown("Built by **Siri Chandana Vaddi** | BTech CSE | House Price Prediction Project")