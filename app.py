from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="EcoType Forest Cover Predictor", page_icon="🌲", layout="wide")

MODEL_PATH = Path("C:/Users/Gokul/OneDrive/Desktop/EcoType_Forest_Cover/notebooks/models/ecotype_best_model.pkl")

COVER_TYPE_NAMES = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz",
}

INPUT_FEATURES = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
    "Wilderness_Area", "Soil_Type",
]


def engineer_forest_features(frame):
    """Keep this transformation identical to the training notebook."""
    result = frame.copy()
    aspect_radians = np.deg2rad(result["Aspect"] % 360)
    result["Aspect_Sin"] = np.sin(aspect_radians)
    result["Aspect_Cos"] = np.cos(aspect_radians)
    result["Hydrology_Distance_3D"] = np.sqrt(
        result["Horizontal_Distance_To_Hydrology"] ** 2
        + result["Vertical_Distance_To_Hydrology"] ** 2
    )
    result["Hydrology_Vertical_Ratio"] = (
        result["Vertical_Distance_To_Hydrology"]
        / (result["Horizontal_Distance_To_Hydrology"] + 1.0)
    )
    shade_cols = ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
    result["Hillshade_Mean"] = result[shade_cols].mean(axis=1)
    result["Hillshade_Range"] = (
        result[shade_cols].max(axis=1) - result[shade_cols].min(axis=1)
    )
    result["Hillshade_Morning_Evening_Diff"] = (
        result["Hillshade_9am"] - result["Hillshade_3pm"]
    )
    result["Hillshade_Noon_Morning_Diff"] = (
        result["Hillshade_Noon"] - result["Hillshade_9am"]
    )
    return result


def prepare_input(raw_row, bundle):
    """Apply training-derived caps, log transforms and feature engineering."""
    row = raw_row[ bundle["input_features"] ].copy()

    caps = bundle.get("iqr_caps", {})
    for feature, bounds in caps.items():
        if feature in row.columns:
            row[feature] = row[feature].clip(lower=bounds[0], upper=bounds[1])

    for feature in bundle.get("log1p_features", []):
        if feature in row.columns:
            row[feature] = np.log1p(row[feature].clip(lower=0))

    return engineer_forest_features(row)


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


st.title("🌲 EcoType — Forest Cover Type Prediction")
st.write(
    "Enter the 12 cartographic measurements to predict the dominant forest cover type. "
    "The application uses the saved model and the same preprocessing steps used during training."
)

if not MODEL_PATH.exists():
    st.error(
        "Trained model not found. Run the complete notebook first. "
        "It creates models/ecotype_best_model.pkl."
    )
    st.stop()

try:
    bundle = load_bundle()
except Exception as exc:
    st.error(f"Could not load the model package: {exc}")
    st.stop()

st.caption(f"Loaded model: {bundle.get('model_name', 'Saved EcoType model')}")

with st.form("prediction_form"):
    st.subheader("Terrain measurements")
    c1, c2, c3 = st.columns(3)

    with c1:
        elevation = st.number_input("Elevation (m)", min_value=0, max_value=5000, value=2500)
        aspect = st.number_input("Aspect (0–360°)", min_value=0, max_value=360, value=180)
        slope = st.number_input("Slope (degrees)", min_value=0, max_value=90, value=15)
        hydrology_h = st.number_input("Horizontal distance to hydrology (m)", min_value=0, value=100)

    with c2:
        hydrology_v = st.number_input("Vertical distance to hydrology (m)", min_value=-1000, max_value=1000, value=0)
        road = st.number_input("Horizontal distance to roadways (m)", min_value=0, value=1000)
        shade9 = st.number_input("Hillshade at 9 AM (0–255)", min_value=0, max_value=255, value=200)
        shadenoon = st.number_input("Hillshade at noon (0–255)", min_value=0, max_value=255, value=220)

    with c3:
        shade3 = st.number_input("Hillshade at 3 PM (0–255)", min_value=0, max_value=255, value=150)
        fire = st.number_input("Horizontal distance to fire points (m)", min_value=0, value=1500)
        wilderness = st.selectbox("Wilderness Area code", options=[1, 2, 3, 4])
        soil = st.selectbox("Soil Type code", options=list(range(1, 41)))

    submitted = st.form_submit_button("Predict forest cover type", type="primary")

if submitted:
    values = {
        "Elevation": elevation,
        "Aspect": aspect,
        "Slope": slope,
        "Horizontal_Distance_To_Hydrology": hydrology_h,
        "Vertical_Distance_To_Hydrology": hydrology_v,
        "Horizontal_Distance_To_Roadways": road,
        "Hillshade_9am": shade9,
        "Hillshade_Noon": shadenoon,
        "Hillshade_3pm": shade3,
        "Horizontal_Distance_To_Fire_Points": fire,
        "Wilderness_Area": wilderness,
        "Soil_Type": soil,
    }
    raw = pd.DataFrame([values], columns=INPUT_FEATURES)

    try:
        prepared = prepare_input(raw, bundle)
        prediction = bundle["model"].predict(prepared)[0]

        # XGBoost labels are decoded to the original Cover_Type values.
        if bundle.get("model_name") == "XGBoost":
            encoder = bundle.get("xgb_target_encoder")
            if encoder is not None:
                prediction = encoder.inverse_transform(
                    np.asarray([int(prediction)])
                )[0]


        
        def get_cover_name(prediction):
            # Model may return a numeric class or an existing class name.
            if isinstance(prediction, (int, np.integer)):
                return COVER_TYPE_NAMES.get(int(prediction), str(prediction))

            prediction = str(prediction).strip()

            # Already a readable forest-cover name
            if prediction in COVER_TYPE_NAMES.values():
                return prediction

            # Numeric class stored as a string
            if prediction.isdigit():
                return COVER_TYPE_NAMES.get(int(prediction), prediction)

            return prediction


        cover_name = get_cover_name(prediction)
        
        st.success(f"Predicted forest cover type: **{cover_name}**")
        st.write(f"Cover_Type class: **{prediction}**")
        st.info(
            "Prediction generated successfully using the trained EcoType model and the supplied terrain measurements. "
            
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
