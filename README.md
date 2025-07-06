# 🧠 Sentiment Analysis Pipeline with Feedback Loop

This repository contains an end-to-end sentiment analysis pipeline built with Python. It covers:

- Raw data ingestion from multiple formats (CSV/JSON)
- Schema detection and data preparation
- Text cleaning and lemmatization
- Model training and versioning
- Interactive Streamlit app with feedback loop
- Model metadata tracking

---

## 📁 Project Structure
```
├── data/
│ ├── raw/ # Raw data files (Kaggle, Amazon, News)
│ ├── clean/ # Cleaned and lemmatized data
│ ├── raw_state.json # Tracks last modification timestamps
│ └── user_feedback.csv # User corrections collected via Streamlit
│
├── model/ # Production model artifacts
│ ├── model.pkl
│ ├── vectorizer.pkl
│ ├── label_encoder.pkl
│ └── metrics.json # Metadata: accuracy, version, training size
│
├── models/ # Versioned models (archived)
│
├── Scripts/
│ ├── Prepare_data.py # Merges and standardizes raw data
│ └── Clean_data.py # Lemmatizes text, removes stopwords, etc.
│
├── train_model.py # Checks for new data → triggers retraining
├── streamlit_app.py # Front-end app for predictions and feedback
├── requirements.txt
└── README.md
```


---

## 🔄 Pipeline Overview

### 1. **Data Ingestion & Preparation**
- Supports multiple schemas: Amazon reviews, Kaggle sentiment data, and news headlines.
- `Prepare_data.py` detects schema and merges all into a standard format.

### 2. **Data Cleaning**
- `Clean_data.py` performs:
  - Lowercasing
  - Regex cleanup
  - Lemmatization (via spaCy)
  - Stopword filtering (customizable)

### 3. **Training & Versioning**
- `train_model.py`:
  - Detects new or updated files using `raw_state.json`
  - Triggers `Prepare_data.py` and `Clean_data.py`
  - Vectorizes text (TF-IDF), trains logistic regression
  - Saves new model/version only if accuracy improves
  - Updates production artifacts and `metrics.json`

### 4. **Interactive Streamlit App**
- Users can:
  - Input text or upload `.txt` files (no header, one row per text)
  - Get sentiment predictions with confidence scores
  - Submit corrected labels for feedback
- Feedback is stored for potential fine-tuning.

---

## 📊 Model Metadata Example
```
json
{
  "version": 4,
  "accuracy": 0.8381,
  "training_size": 147871
}
```
🚀 Running the App
Install dependencies:
```
pip install -r requirements.txt
```
Launch the app locally:
```
streamlit run streamlit_app.py
```
