import os
import glob
import time
import pandas as pd
from nltk.corpus import stopwords
import spacy

# Load combined cleaned data
df = pd.read_csv('data/clean/combined_data.csv')

# Preprocessing: lowercase, remove non-alpha chars, trim spaces
df['text'] = df['text'].str.lower()
df['text'] = df['text'].str.replace(r"[^a-zA-Z\s]", "", regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()

# Define stopwords to keep
stop_words_to_keep = {
    "not", "no", "nor", "neither", "never", "n't", "very", "too", "so", "really", "just", 
    "only", "even", "still", "always", "ever", "ain", "aren", "aren't", "couldn", "couldn't", 
    "didn", "didn't", "doesn", "doesn't", "don", "don't", "hadn", "hadn't", "hasn", "hasn't",
    "haven", "haven't", "isn", "isn't", "mightn", "mightn't", "mustn", "mustn't", "needn",
    "needn't", "shan", "shan't", "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't",
    "won", "won't", "wouldn", "wouldn't", "more", "most", "own"
}
stop_words = set(stopwords.words('english')).difference(stop_words_to_keep)

# Remove stopwords
df['text'] = df['text'].apply(lambda d: " ".join([word for word in d.split() if word not in stop_words]))

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Lemmatization function
def lemmatize(text):
    doc = nlp(text)
    allowed_pos = {"ADJ", "NOUN", "PROPN", "ADV", "VERB"}
    return ' '.join([
        token.lemma_.lower()
        for token in doc
        if token.pos_ in allowed_pos and
           not token.is_punct and
           not token.is_space and
           token.lemma_.lower() not in stop_words
    ])

# Detect latest versioned output
lemm_dir = "data/clean"
existing = glob.glob(os.path.join(lemm_dir, "lemmatized_data_v*.csv"))
versions = [int(f.split("_v")[-1].split(".")[0]) for f in existing]
next_version = max(versions) + 1 if versions else 1
output_path = os.path.join(lemm_dir, f"lemmatized_data_v{next_version}.csv")

# Chunk processing
start_time = time.time()
chunk_size = 10000
total_rows = len(df)

for start_idx in range(0, total_rows, chunk_size):
    end_idx = start_idx + chunk_size
    chunk = df.iloc[start_idx:end_idx].copy()
    chunk["text"] = chunk["text"].apply(lemmatize)
    chunk.to_csv(output_path, mode="a", header=not bool(start_idx), index=False)
    print(f"✅ Processed rows {start_idx}–{end_idx}")

print(f"\n📁 Saved to {output_path}")
print(f"⏱️ Total time: {time.time() - start_time:.2f} seconds")
