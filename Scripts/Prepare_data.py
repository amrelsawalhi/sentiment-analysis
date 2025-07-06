import os
import pandas as pd
import numpy as np

RAW_DIR = "data/raw"
CLEAN_PATH = "data/clean/combined_data.csv"

def load_kaggle(path):
    return pd.read_csv(path, encoding="ISO-8859-1", header=None, names=["sentiment", "text"])

def load_amazon(path):
    df = pd.read_json(path, lines=True)
    
    df['text'] = df['summary'].fillna('') + " " + df['reviewText'].fillna('')
    
    sentiment_map = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}
    
    df['sentiment'] = df['overall'].map(sentiment_map)
    
    return df[['sentiment', 'text']]

def load_news(path):
    df = pd.read_csv(path,encoding="ISO-8859-1")
    df.drop(columns=[df.columns[0], 'Time'], inplace=True)
    df.rename(columns={'Headlines': 'text', 'Sentiment': 'sentiment'}, inplace=True)

    conditions = [
        df['sentiment'] >= 4,
        df['sentiment'] == 3,
        df['sentiment'] <= 2
    ]
    choices = ['positive', 'neutral', 'negative']

    df['sentiment'] = np.select(conditions, choices, default='neutral')

    df = df[['sentiment', 'text']]
    return df



def detect_schema(file_name):
    name = file_name.lower()
    if name.startswith("amazon"):
        return "amazon"
    elif name.startswith("kaggle"):
        return "kaggle"
    elif name.startswith("news"):
        return "news"
    return None

def prepare_all():
    dfs = []
    for file in os.listdir(RAW_DIR):
        path = os.path.join(RAW_DIR, file)
        schema = detect_schema(file)
        if not schema:
            print(f"⚠️ Skipping unrecognized file: {file}")
            continue
        if schema == "kaggle":
            dfs.append(load_kaggle(path))
        elif schema == "amazon":
            dfs.append(load_amazon(path))
        elif schema == "news":
            dfs.append(load_news(path))
        else:
            print(f"❌ No loader defined for schema: {schema}")

    if not dfs:
        print("❌ No data loaded. Check RAW_DIR and schema detection.")
        return

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(CLEAN_PATH, index=False)
    print(f"✅ Combined data saved to {CLEAN_PATH} ({len(combined_df)} rows)")

if __name__ == "__main__":
    prepare_all()
