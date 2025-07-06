import os
import json
import joblib
import shutil
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import subprocess




def get_latest_lemmatized_file():
    clean_dir = "data/clean"
    files = [f for f in os.listdir(clean_dir) if f.startswith("lemmatized_data_v") and f.endswith(".csv")]
    if not files:
        raise FileNotFoundError("No lemmatized_data_v*.csv file found.")
    versions = [int(f.split("_v")[1].split(".")[0]) for f in files]
    latest_version = max(versions)
    return os.path.join(clean_dir, f"lemmatized_data_v{latest_version}.csv")

# === CONFIGURATION === #
RAW_DIR = "data/raw"
STATE_FILE = "data/raw_state.json"
CLEANED_DATA_PATH = None
MODEL_DIR = "models"
PROD_MODEL_DIR = "model"

# === HELPERS === #
def get_raw_file_state():
    return {
        f: os.path.getmtime(os.path.join(RAW_DIR, f))
        for f in os.listdir(RAW_DIR)
        if f.endswith((".csv", ".json"))
    }

def load_previous_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}

def is_data_changed(current_state, previous_state):
    return current_state != previous_state



def get_latest_model_version():
    versions = []
    for fname in os.listdir(MODEL_DIR):
        if fname.startswith("model_v") and fname.endswith(".pkl"):
            try:
                version = int(fname.split("_v")[1].split(".")[0])
                versions.append(version)
            except:
                pass
    return max(versions) if versions else 0

# === MAIN EXECUTION === #
current_state = get_raw_file_state()
previous_state = load_previous_state()

if not is_data_changed(current_state, previous_state):
    print("✅ No new data. Skipping retraining.")
    exit()

print("🔁 New data found. Starting training...")
print("📦 Running prepare_data.py...")
subprocess.run(["python", "Scripts/Prepare_data.py"], check=True)

print("🧹 Running clean_data.py...")
subprocess.run(["python", "Scripts/Clean_data.py"], check=True)
CLEANED_DATA_PATH = get_latest_lemmatized_file()

# Load cleaned lemmatized data
if not os.path.exists(CLEANED_DATA_PATH):
    raise FileNotFoundError(f"Cleaned dataset not found at {CLEANED_DATA_PATH}")

df = pd.read_csv(CLEANED_DATA_PATH)
df = df.dropna(subset=["text"])

print(f"🧮 Training on {len(df)} samples.")

# Encode labels
le = LabelEncoder()
X = df["text"]
y = le.fit_transform(df["sentiment"])

# Split and vectorize
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)
accuracy = accuracy_score(y_test, clf.predict(X_test_vec))
print(f"📊 New model accuracy: {accuracy:.4f}")

# Save model, vectorizer, label encoder with version
version = get_latest_model_version() + 1
model_path = os.path.join(MODEL_DIR, f"model_v{version}.pkl")
vec_path = os.path.join(MODEL_DIR, f"vectorizer_v{version}.pkl")
label_path = os.path.join(MODEL_DIR, f"label_encoder_v{version}.pkl")
metrics_path = os.path.join(MODEL_DIR, f"metrics_v{version}.json")

joblib.dump(clf, model_path)
joblib.dump(vectorizer, vec_path)
joblib.dump(le, label_path)
with open(metrics_path, "w") as f:
    json.dump({
        "accuracy": accuracy,
        "training_size": len(df),
        "version": version
    }, f)

# Load current prod metrics
prod_metrics_path = os.path.join(PROD_MODEL_DIR, "metrics.json")
if os.path.exists(prod_metrics_path):
    with open(prod_metrics_path) as f:
        prod_metrics = json.load(f)
    prod_acc = prod_metrics.get("accuracy", 0)
else:
    prod_acc = 0

# Promote model if better
if accuracy >= prod_acc:
    shutil.copy(model_path, os.path.join(PROD_MODEL_DIR, "model.pkl"))
    shutil.copy(vec_path, os.path.join(PROD_MODEL_DIR, "vectorizer.pkl"))
    shutil.copy(label_path, os.path.join(PROD_MODEL_DIR, "label_encoder.pkl"))
    shutil.copy(metrics_path, os.path.join(PROD_MODEL_DIR, "metrics.json"))
    print(f"✅ Promoted model_v{version}. Accuracy improved or equal.")
else:
    print(f"⚠️ Model_v{version} not promoted. Accuracy lower than current.")

# Save new raw file state
with open(STATE_FILE, "w") as f:
    json.dump(current_state, f)

print("✅ Done.")
