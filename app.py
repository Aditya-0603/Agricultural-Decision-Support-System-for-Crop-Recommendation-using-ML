import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Crop Recommendation", page_icon="🌾", layout="centered")

@st.cache_data
def load_data(path: str = "Crop_recommendation.csv") -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data
def get_available_crops(df: pd.DataFrame) -> list:
    return sorted(df["label"].unique())

@st.cache_data
def get_crop_ranges(df: pd.DataFrame) -> dict:
    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    ranges = {}
    grouped = df.groupby("label")[features].agg(["min", "max"])
    for label, row in grouped.iterrows():
        feature_ranges = {}
        for feature in features:
            minimum = row[(feature, "min")]
            maximum = row[(feature, "max")]
            feature_ranges[feature] = (minimum, maximum)
        ranges[label] = feature_ranges
    return ranges

@st.cache_resource
def train_crop_model(df: pd.DataFrame):
    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X = df[features]
    y = df["label"]
    encoder = LabelEncoder().fit(y)
    y_encoded = encoder.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, encoder, accuracy

try:
    data = load_data()
except FileNotFoundError:
    st.error("Could not find Crop_recommendation.csv in the current folder.")
    st.stop()

model, label_encoder, model_accuracy = train_crop_model(data)
available_crops = get_available_crops(data)
crop_ranges = get_crop_ranges(data)

feature_bounds = data[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]].agg(["min", "max", "median"])

st.title("Crop Recommendation System")
st.markdown(
    "Use nutrient, weather, and soil inputs to get a recommended crop based on the dataset."
)

with st.expander("Dataset Preview"):
    st.dataframe(data.head(10))
    st.write(f"Total records: {len(data)}")
    st.write(f"Available crop labels: {len(available_crops)}")


st.sidebar.header("Available Crops")
st.sidebar.write(", ".join(available_crops))

st.sidebar.markdown("---")
st.sidebar.header("Crop feature ranges")
with st.sidebar.expander("Show crop ranges"):
    for crop, feature_ranges in crop_ranges.items():
        st.markdown(f"**{crop}**")
        range_text = []
        for feature, (minimum, maximum) in feature_ranges.items():
            if feature in {"N", "P", "K"}:
                range_text.append(f"{feature}: {int(minimum)}–{int(maximum)}")
            else:
                range_text.append(f"{feature}: {minimum:.2f}–{maximum:.2f}")
        st.write(", ".join(range_text))

st.header("Input Features")
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        nitrogen = st.slider(
            "Nitrogen (N)",
            int(feature_bounds.loc["min", "N"]),
            int(feature_bounds.loc["max", "N"]),
            int(feature_bounds.loc["median", "N"]),
        )
        phosphorus = st.slider(
            "Phosphorus (P)",
            int(feature_bounds.loc["min", "P"]),
            int(feature_bounds.loc["max", "P"]),
            int(feature_bounds.loc["median", "P"]),
        )
        potassium = st.slider(
            "Potassium (K)",
            int(feature_bounds.loc["min", "K"]),
            int(feature_bounds.loc["max", "K"]),
            int(feature_bounds.loc["median", "K"]),
        )
    with col2:
        temperature = st.slider(
            "Temperature (°C)",
            float(feature_bounds.loc["min", "temperature"]),
            float(feature_bounds.loc["max", "temperature"]),
            float(feature_bounds.loc["median", "temperature"]),
            step=0.1,
        )
        humidity = st.slider(
            "Humidity (%)",
            float(feature_bounds.loc["min", "humidity"]),
            float(feature_bounds.loc["max", "humidity"]),
            float(feature_bounds.loc["median", "humidity"]),
            step=0.1,
        )
    with col3:
        ph = st.slider(
            "Soil pH",
            float(feature_bounds.loc["min", "ph"]),
            float(feature_bounds.loc["max", "ph"]),
            float(feature_bounds.loc["median", "ph"]),
            step=0.01,
        )
        rainfall = st.slider(
            "Rainfall (mm)",
            float(feature_bounds.loc["min", "rainfall"]),
            float(feature_bounds.loc["max", "rainfall"]),
            float(feature_bounds.loc["median", "rainfall"]),
            step=0.1,
        )

    submit = st.form_submit_button("Recommend Crop")

if submit:
    input_df = pd.DataFrame(
        [
            {
                "N": nitrogen,
                "P": phosphorus,
                "K": potassium,
                "temperature": temperature,
                "humidity": humidity,
                "ph": ph,
                "rainfall": rainfall,
            }
        ]
    )
    prediction_encoded = model.predict(input_df)[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

    st.success(f"Recommended crop: {prediction_label}")
    st.write("### Input summary")
    st.write(input_df.T)
    st.write(f"Model accuracy on hold-out set: {model_accuracy:.2%}")


