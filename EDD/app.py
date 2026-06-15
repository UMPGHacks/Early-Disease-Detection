import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch
from torch import nn

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

SYMPTOM_MODEL_PATH = MODELS_DIR / "symptom_model.pkl"
HEART_MODEL_PATH = MODELS_DIR / "heart_model.pkl"
LIVER_MODEL_PATH = MODELS_DIR / "liver_model.pkl"
LUNG_MODEL_PATH = MODELS_DIR / "lung_model.pt"

HEART_FEATURES = [
    "age",
    "anaemia",
    "creatinine_phosphokinase",
    "diabetes",
    "ejection_fraction",
    "high_blood_pressure",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "sex",
    "smoking",
    "time",
]

LIVER_FEATURES = [
    "Age",
    "Gender",
    "TB",
    "DB",
    "Alkphos",
    "Sgpt",
    "Sgot",
    "TP",
    "ALB",
    "A/G Ratio",
]

SYMPTOM_FEATURES = [
    "fever",
    "headache",
    "nausea",
    "vomiting",
    "fatigue",
    "joint_pain",
    "skin_rash",
    "cough",
    "weight_loss",
    "yellow_eyes",
]

SYMPTOM_LABELS = {
    "fever": "Fever",
    "headache": "Headache",
    "nausea": "Nausea",
    "vomiting": "Vomiting",
    "fatigue": "Fatigue",
    "joint_pain": "Joint Pain",
    "skin_rash": "Skin Rash",
    "cough": "Cough",
    "weight_loss": "Weight Loss",
    "yellow_eyes": "Yellow Eyes",
}

SYMPTOM_PRECAUTIONS = {
    "Dengue": ["Drink fluids", "Avoid mosquito bites", "Consult a doctor quickly"],
    "Jaundice": ["Avoid oily food", "Take proper rest", "Drink plenty of water"],
    "Malaria": ["Use mosquito nets", "Seek medical help", "Follow prescribed treatment"],
    "Typhoid": ["Drink safe water", "Eat hygienic food", "Complete the medicine course"],
    "Flu": ["Rest well", "Drink warm fluids", "Monitor fever and breathing"],
    "Pneumonia": ["Get medical evaluation", "Rest and hydrate", "Watch breathing symptoms"],
    "Heart attack": ["Seek emergency care immediately", "Do not ignore chest symptoms", "Follow urgent medical advice"],
    "Hypertension": ["Reduce salt intake", "Monitor blood pressure", "Exercise regularly"],
    "Diabetes": ["Monitor blood sugar", "Follow diet plan", "Stay physically active"],
    "Tuberculosis": ["Visit a doctor promptly", "Avoid close-contact spread", "Follow treatment fully"],
}

HEART_RISK_HINTS = [
    ("Low ejection fraction", "Below 40 may indicate weakened heart pumping."),
    ("High serum creatinine", "Higher values can suggest kidney stress linked to worse outcomes."),
    ("Low serum sodium", "Lower sodium may be associated with poor heart failure prognosis."),
    ("Anaemia or high blood pressure", "These can increase cardiovascular burden."),
]

LIVER_RISK_HINTS = [
    ("High bilirubin", "Raised total or direct bilirubin can point to impaired liver processing."),
    ("High enzymes", "Higher SGPT, SGOT, or alkaline phosphotase may suggest liver inflammation or damage."),
    ("Low albumin", "Lower albumin can be associated with reduced liver synthetic function."),
    ("Low A/G ratio", "A lower albumin-globulin ratio can appear in several liver disorders."),
]


class SimpleLungCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


@st.cache_resource
def load_model(model_path: Path):
    if not model_path.exists():
        return None
    return joblib.load(model_path)


@st.cache_resource
def load_lung_model(model_path: Path):
    if not model_path.exists():
        return None, None, None
    checkpoint = torch.load(model_path, map_location="cpu")
    class_names = checkpoint["class_names"]
    image_size = checkpoint.get("image_size", 128)
    model = SimpleLungCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names, image_size


def predict_with_confidence(model, values):
    prediction = model.predict(values)[0]
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba(values)[0]) * 100)
    return prediction, confidence


def make_single_row_dataframe(columns, values):
    return pd.DataFrame([values], columns=columns)


def show_model_missing(model_name: str, command_name: str):
    st.warning(
        f"{model_name} model not found. Run `{command_name}` inside the EDD folder to create it."
    )


def render_symptom_prediction(symptom_model):
    st.subheader("Symptom-Based Disease Prediction")
    st.write("Choose the symptoms present in the patient.")

    cols = st.columns(3)
    symptom_values = []
    for index, feature in enumerate(SYMPTOM_FEATURES):
        with cols[index % 3]:
            symptom_values.append(int(st.checkbox(SYMPTOM_LABELS[feature], key=f"symptom_{feature}")))

    if st.button("Predict Symptom-Based Disease", use_container_width=True):
        if symptom_model is None:
            show_model_missing("Symptom", "python model_training.py")
        elif sum(symptom_values) == 0:
            st.error("Select at least one symptom before prediction.")
        else:
            input_data = np.array([symptom_values])
            prediction, confidence = predict_with_confidence(symptom_model, input_data)
            st.success(f"Predicted disease: {prediction}")
            if confidence is not None:
                st.info(f"Confidence: {confidence:.2f}%")

            if hasattr(symptom_model, "predict_proba"):
                probabilities = symptom_model.predict_proba(input_data)[0]
                top_indices = np.argsort(probabilities)[::-1][:3]
                top_labels = symptom_model.classes_[top_indices]
                top_scores = probabilities[top_indices] * 100
                st.write("Top probable diseases")
                for label, score in zip(top_labels, top_scores):
                    st.write(f"- {label}: {score:.2f}%")

            precautions = SYMPTOM_PRECAUTIONS.get(prediction)
            if precautions:
                st.write("Suggested precautions")
                for item in precautions:
                    st.write(f"- {item}")


def render_heart_prediction(heart_model):
    st.subheader("Heart Failure Risk Screening")
    st.markdown("""<div class="section-card"><h3 style="margin-top:0; color:#0f172a;">Patient Details</h3></div>""", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        smoking = st.selectbox("Smoking", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with col2:
        anaemia = st.selectbox("Anaemia", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        high_blood_pressure = st.selectbox("High blood pressure", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        time = st.number_input("Follow-up time", min_value=1.0, value=130.0)

    st.markdown("""<div class="section-card"><h3 style="margin-top:0; color:#0f172a;">Clinical Measurements</h3></div>""", unsafe_allow_html=True)
    col3, col4, col5 = st.columns(3)
    with col3:
        creatinine_phosphokinase = st.number_input("Creatinine phosphokinase", min_value=1.0, value=250.0)
        ejection_fraction = st.number_input("Ejection fraction", min_value=1.0, max_value=100.0, value=38.0)
    with col4:
        platelets = st.number_input("Platelets", min_value=1000.0, value=263358.03)
        serum_creatinine = st.number_input("Serum creatinine", min_value=0.1, value=1.1)
    with col5:
        serum_sodium = st.number_input("Serum sodium", min_value=100.0, max_value=200.0, value=136.0)

    with st.expander("Heart risk interpretation help"):
        for title, detail in HEART_RISK_HINTS:
            st.write(f"- {title}: {detail}")

    if st.button("Predict Heart Risk", use_container_width=True):
        if heart_model is None:
            show_model_missing("Heart", "python model_training.py")
        else:
            input_data = make_single_row_dataframe(
                HEART_FEATURES,
                [
                    age,
                    anaemia,
                    creatinine_phosphokinase,
                    diabetes,
                    ejection_fraction,
                    high_blood_pressure,
                    platelets,
                    serum_creatinine,
                    serum_sodium,
                    sex,
                    smoking,
                    time,
                ],
            )
            prediction, confidence = predict_with_confidence(heart_model, input_data)
            risk_label = "Higher heart failure outcome risk" if int(prediction) == 1 else "Lower heart failure outcome risk"

            metric_col1, metric_col2 = st.columns(2)
            risk_value = "High" if int(prediction) == 1 else "Low"
            confidence_value = f"{confidence:.2f}%" if confidence is not None else "N/A"
            with metric_col1:
                st.markdown(f"""<div class="metric-card"><h4>Risk Class</h4><p>{risk_value}</p></div>""", unsafe_allow_html=True)
            with metric_col2:
                st.markdown(f"""<div class="metric-card"><h4>Model Confidence</h4><p>{confidence_value}</p></div>""", unsafe_allow_html=True)

            if int(prediction) == 1:
                st.markdown(f'<div class="pill-high">{risk_label}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="pill-low">{risk_label}</div>', unsafe_allow_html=True)

            flagged_points = []
            if ejection_fraction < 40:
                flagged_points.append("Ejection fraction is below 40.")
            if serum_creatinine > 1.3:
                flagged_points.append("Serum creatinine is above a typical reference level.")
            if serum_sodium < 135:
                flagged_points.append("Serum sodium is below 135.")
            if high_blood_pressure == 1:
                flagged_points.append("High blood pressure is present.")
            if anaemia == 1:
                flagged_points.append("Anaemia is present.")
            if smoking == 1:
                flagged_points.append("Smoking is present as a risk factor.")

            st.markdown("""<div class="section-card"><h3 style="margin-top:0; color:#0f172a;">Result Summary</h3></div>""", unsafe_allow_html=True)
            if flagged_points:
                for item in flagged_points:
                    st.write(f"- {item}")
            else:
                st.write("- No major quick-check warning markers were triggered from the entered values.")

            st.info("Use this result only as a screening signal with doctor supervision.")


def render_liver_prediction(liver_model):
    st.subheader("Liver Disease Detection")
    st.markdown("""<div class="section-card"><h3 style="margin-top:0; color:#0f172a;">Patient Profile</h3></div>""", unsafe_allow_html=True)
    profile_col1, profile_col2 = st.columns(2)
    with profile_col1:
        liver_age = st.number_input("Age ", min_value=1, max_value=120, value=45)
    with profile_col2:
        gender = st.selectbox("Gender", ["Male", "Female"])

    st.markdown("""<div class="section-card"><h3 style="margin-top:0; color:#0f172a;">Liver Function Inputs</h3></div>""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        tb = st.number_input("Total Bilirubin", min_value=0.1, value=1.0)
        db = st.number_input("Direct Bilirubin", min_value=0.1, value=0.3)
        alkphos = st.number_input("Alkaline Phosphotase", min_value=1.0, value=210.0)
    with col2:
        sgpt = st.number_input("Alamine Aminotransferase", min_value=1.0, value=35.0)
        sgot = st.number_input("Aspartate Aminotransferase", min_value=1.0, value=40.0)
        tp = st.number_input("Total Proteins", min_value=0.1, value=6.8)
    with col3:
        alb = st.number_input("Albumin", min_value=0.1, value=3.3)
        agr = st.number_input("Albumin and Globulin Ratio", min_value=0.1, value=0.9)

    with st.expander("Liver result interpretation help"):
        for title, detail in LIVER_RISK_HINTS:
            st.write(f"- {title}: {detail}")

    if st.button("Predict Liver Disease", use_container_width=True):
        if liver_model is None:
            show_model_missing("Liver", "python model_training.py")
        else:
            input_data = make_single_row_dataframe(
                LIVER_FEATURES,
                [liver_age, gender, tb, db, alkphos, sgpt, sgot, tp, alb, agr],
            )
            prediction, confidence = predict_with_confidence(liver_model, input_data)
            label = "Liver disease likely" if int(prediction) == 1 else "Liver disease less likely"

            metric_col1, metric_col2 = st.columns(2)
            risk_value = "High" if int(prediction) == 1 else "Low"
            confidence_value = f"{confidence:.2f}%" if confidence is not None else "N/A"
            with metric_col1:
                st.markdown(f"""<div class="metric-card"><h4>Risk Class</h4><p>{risk_value}</p></div>""", unsafe_allow_html=True)
            with metric_col2:
                st.markdown(f"""<div class="metric-card"><h4>Model Confidence</h4><p>{confidence_value}</p></div>""", unsafe_allow_html=True)

            if int(prediction) == 1:
                st.markdown(f'<div class="pill-high">{label}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="pill-low">{label}</div>', unsafe_allow_html=True)

            flagged_points = []
            if tb > 1.2:
                flagged_points.append("Total bilirubin is above a common reference level.")
            if db > 0.3:
                flagged_points.append("Direct bilirubin is above a common reference level.")
            if alkphos > 120:
                flagged_points.append("Alkaline phosphotase is elevated.")
            if sgpt > 56:
                flagged_points.append("SGPT is elevated.")
            if sgot > 40:
                flagged_points.append("SGOT is elevated.")
            if alb < 3.5:
                flagged_points.append("Albumin is lower than a common reference level.")
            if agr < 1.0:
                flagged_points.append("Albumin and globulin ratio is below 1.0.")

            st.markdown("""<div class="section-card"><h3 style="margin-top:0; color:#0f172a;">Result Summary</h3></div>""", unsafe_allow_html=True)
            if flagged_points:
                for item in flagged_points:
                    st.write(f"- {item}")
            else:
                st.write("- No major quick-check liver markers were triggered from the entered values.")

            st.info("Please confirm liver-related concerns with laboratory tests and medical advice.")


def render_lung_module(lung_model, lung_classes, lung_image_size):
    st.subheader("Lung X-Ray Detection")
    st.markdown("""<div class="section-card"><h3 style="margin-top:0; color:#0f172a;">Upload Chest X-Ray</h3></div>""", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg"])

    with st.expander("Lung class guide"):
        st.write("- Normal: model predicts no major abnormal opacity in this dataset label set.")
        st.write("- Lung_Opacity: model predicts opacity patterns present in the training data.")
        st.write("- Viral Pneumonia: model predicts viral pneumonia-like patterns from the dataset.")

    if lung_model is None:
        st.warning("Lung model not found. Run `python train_lung_model.py` inside the EDD folder to create it.")

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded X-ray image", use_container_width=True)

        if lung_model is not None and st.button("Predict Lung Condition", use_container_width=True):
            resized = image.resize((lung_image_size, lung_image_size))
            tensor = torch.tensor(list(resized.getdata()), dtype=torch.float32).view(lung_image_size, lung_image_size, 3)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
            with torch.no_grad():
                logits = lung_model(tensor)
                probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
            top_index = int(np.argmax(probabilities))
            prediction = lung_classes[top_index]
            confidence = float(probabilities[top_index] * 100)

            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.markdown(f"""<div class="metric-card"><h4>Predicted Class</h4><p>{prediction}</p></div>""", unsafe_allow_html=True)
            with metric_col2:
                st.markdown(f"""<div class="metric-card"><h4>Confidence</h4><p>{confidence:.2f}%</p></div>""", unsafe_allow_html=True)

            if prediction == "Normal":
                st.markdown(f'<div class="pill-low">{prediction}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="pill-high">{prediction}</div>', unsafe_allow_html=True)

            st.markdown("""<div class="section-card"><h3 style="margin-top:0; color:#0f172a;">Top Probabilities</h3></div>""", unsafe_allow_html=True)
            ranked = sorted(zip(lung_classes, probabilities), key=lambda item: item[1], reverse=True)
            for label, score in ranked:
                st.write(f"- {label}: {score * 100:.2f}%")

            st.info("This image result is an ML screening output and should be reviewed by a clinician.")


st.set_page_config(page_title="EDD Multi Disease Detection", layout="wide")
st.markdown("""
<style>
    .hero-card {
        background: linear-gradient(135deg, #0f172a, #1d4ed8);
        color: white;
        padding: 1.5rem;
        border-radius: 18px;
        margin-bottom: 1rem;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.18);
    }
    .hero-card h1 {
        margin: 0 0 0.35rem 0;
        font-size: 2rem;
    }
    .hero-card p {
        margin: 0;
        font-size: 1rem;
        opacity: 0.92;
    }
    .section-card {
        background: #f8fafc;
        border: 1px solid #dbeafe;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
    }
    .heart-card {
        background: linear-gradient(180deg, #fff7ed, #ffffff);
        border: 1px solid #fed7aa;
        border-radius: 18px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.07);
    }
    .metric-card h4 {
        margin: 0;
        color: #475569;
        font-size: 0.95rem;
        font-weight: 600;
    }
    .metric-card p {
        margin: 0.35rem 0 0 0;
        color: #0f172a;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .pill-high {
        display: inline-block;
        padding: 0.45rem 0.8rem;
        border-radius: 999px;
        background: #fee2e2;
        color: #b91c1c;
        font-weight: 700;
        margin-bottom: 0.7rem;
    }
    .pill-low {
        display: inline-block;
        padding: 0.45rem 0.8rem;
        border-radius: 999px;
        background: #dcfce7;
        color: #166534;
        font-weight: 700;
        margin-bottom: 0.7rem;
    }
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="hero-card">
    <h1>Early Disease Detection Web App</h1>
</div>
""", unsafe_allow_html=True)

symptom_model = load_model(SYMPTOM_MODEL_PATH)
heart_model = load_model(HEART_MODEL_PATH)
liver_model = load_model(LIVER_MODEL_PATH)
lung_model, lung_classes, lung_image_size = load_lung_model(LUNG_MODEL_PATH)

st.sidebar.header("About")
st.sidebar.write(
    "This unified app combines symptom-based prediction, heart failure risk screening, liver disease screening, and a lung X-ray placeholder."
)
st.sidebar.markdown("### Select Model")
selected_module = st.sidebar.radio(
    "Choose a prediction module",
    [
        "Symptom Disease Prediction",
        "Heart Failure Risk",
        "Liver Disease Detection",
        "Lung X-Ray",
    ],
)

if selected_module == "Symptom Disease Prediction":
    render_symptom_prediction(symptom_model)
elif selected_module == "Heart Failure Risk":
    render_heart_prediction(heart_model)
elif selected_module == "Liver Disease Detection":
    render_liver_prediction(liver_model)
else:
    render_lung_module(lung_model, lung_classes, lung_image_size)
