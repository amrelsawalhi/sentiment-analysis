import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json

# === Load Model Artifacts === #
MODEL_DIR = "model"
model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
metrics_path = os.path.join(MODEL_DIR, "metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        metrics = json.load(f)
    model_version = metrics.get("version", "?")
    model_accuracy = metrics.get("accuracy", "?")
    training_size = metrics.get("training_size", "?")
else:
    model_version = model_accuracy = training_size = "?"

# === UI Header === #
st.title("🧠 FeelyText - Sentiment Analysis Tool")
st.markdown("Upload a `.txt` file (no header, one line per row) **or** enter a text manually.")
st.markdown("---")

# === Input === #
text_input = st.text_area("Or type your text here (takes priority over file upload)", height=150)
uploaded_file = st.file_uploader("Upload .txt file", type="txt")

# === Helper: Predict + Confidence === #
def predict_sentiment(texts):
    X_vec = vectorizer.transform(texts)
    probs = model.predict_proba(X_vec)
    preds = model.predict(X_vec)
    confidences = probs.max(axis=1)
    pred_labels = label_encoder.inverse_transform(preds)
    return pred_labels, confidences

# === Single Text Input Mode === #
if text_input.strip():
    st.subheader("🔎 Prediction:")
    pred_label, confidence = predict_sentiment([text_input])
    st.write(f"**Predicted Sentiment:** {pred_label[0]}")
    st.write(f"**Confidence:** {confidence[0]:.2f}")

    corrected = st.selectbox(
        "Is this correct?",
        label_encoder.classes_,
        index=int(label_encoder.transform([pred_label[0]])[0])
    )

    if st.button("✅ Submit Feedback"):
        df = pd.DataFrame([{"text": text_input, "sentiment": corrected}])
        feedback_path = "data/user_feedback.csv"
        df.to_csv(feedback_path, mode="a", index=False, header=not os.path.exists(feedback_path))
        st.success("Feedback submitted.")

# === Multi-line TXT Upload Mode === #
elif uploaded_file:
    lines = [line.strip() for line in uploaded_file.readlines()]
    lines = [line.decode("utf-8") for line in lines if line.strip()]

    st.subheader("🔎 Predictions:")
    pred_labels, confidences = predict_sentiment(lines)

    corrected_labels = []
    for i, (text, pred, conf) in enumerate(zip(lines, pred_labels, confidences)):
        st.markdown(f"**Text {i+1}:** {text}")
        st.write(f"Predicted: `{pred}` | Confidence: `{conf:.2f}`")
        corrected = st.selectbox(
            f"Correct label for Text {i+1}?",
            label_encoder.classes_,
            index=int(label_encoder.transform([pred])[0]),
            key=f"select_{i}"
        )
        corrected_labels.append((text, corrected))
        st.markdown("---")

    if st.button("✅ Submit All Feedback"):
        df_feedback = pd.DataFrame(corrected_labels, columns=["text", "sentiment"])
        feedback_path = "data/user_feedback.csv"
        df_feedback.to_csv(feedback_path, mode="a", index=False, header=not os.path.exists(feedback_path))
        st.success("Feedback submitted.")

else:
    st.info("Please either type text or upload a `.txt` file.")

st.markdown(
    f"**Model v{model_version}** | "
    f"Trained on **{training_size:,}** samples | "
    f"Accuracy: **{model_accuracy:.2%}**"
)


st.markdown("---")
st.markdown(
    "🔒 **Disclaimer:** This app is for demonstration and educational purposes only. "
    "Predictions may be imperfect and should not be used as the sole basis for decisions. "
    "Your feedback is stored locally to enhance the accuarcy of the model")

st.markdown("""---""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center;">
    <a href="mailto:amr.elsawalhi.business@gmail.com" style="text-decoration: none; margin-right: 20px;">
        📧
    </a>
    <a href="https://www.linkedin.com/in/amrelsawalhi/" target="_blank" style="text-decoration: none; margin-right: 20px;">
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="25"/>
    </a>
    <a href="https://github.com/amrelsawalhi" target="_blank" style="text-decoration: none;">
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" width="25"/>
    </a>
</div>
""", unsafe_allow_html=True)