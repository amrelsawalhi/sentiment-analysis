<p align="center">
  <img src="https://github.com/amrelsawalhi/sentiment-analysis/blob/3cc1c86d7477ad723bf516f18ae305c2b3ebfa07/textyfeel.png?raw=true" alt="FeelyText Banner" style="max-width: 100%;">
</p>

# 🧠 FeelyText – Real-time Sentiment Analyzer

FeelyText is an interactive sentiment analysis tool powered by machine learning and NLP. It lets users type or upload text, view predicted sentiment, and provide feedback to improve future predictions.


---

## 📁 Project Structure
```
├── data/
│ ├── raw/ 
│ ├── clean/ 
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
- Supports multiple input formats: Amazon reviews, Kaggle sentiment datasets, and news headlines.
- `prepare_data.py` automatically detects schema and standardizes all inputs into a unified format.

### 2. **Text Cleaning & Processing**
- `clean_data.py` applies a full NLP cleaning pipeline:
  - Lowercasing & punctuation removal (Regex)
  - Lemmatization (via spaCy)
  - Custom stopword filtering
- Output is saved to `/data/clean` for downstream use.

### 3. **Model Training & Version Control**
- `train_model.py`:
  - Detects new or modified data via `raw_state.json`
  - Triggers preparation and cleaning if needed
  - Trains a logistic regression model with TF-IDF vectorization
  - Evaluates performance — updates production model only if accuracy improves
  - Saves updated artifacts (`model.pkl`, `vectorizer.pkl`, `metrics.json`)

### 4. **Interactive Streamlit App**
- `streamlit_app.py` lets users:
  - Type or upload text for sentiment prediction
  - See predicted label and model confidence
  - Submit corrected labels to feed a user-driven feedback loop
- Feedback is saved locally (`user_feedback.csv`) for future fine-tuning.

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
