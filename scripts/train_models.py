"""
Word2Vec training for diachronic analysis (1980–2019)

"""

import pandas as pd
import os
from gensim.models import Word2Vec
import multiprocessing

# === Config ===
INPUT_FILE = "data/cleaned_dataset.csv"   #processed data goes here
OUTPUT_DIR = "outputs/models"             #trained Word2Vec models :)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Detect CPU cores and assign a safe number of parallel workers
# :)
num_cores = multiprocessing.cpu_count()
safe_workers = max(1, num_cores // 2)
print(f"Detected {num_cores} logical cores. Using {safe_workers} workers for training.\n")

# Training parameters
W2V_PARAMS = {
    "vector_size": 100,  # Good balance for moderate corpora
    "window": 10,         # Moderate context, balances syntax & semantics
    "min_count": 2,      # Lowered to capture rare metaphorical terms
    "workers": safe_workers,
    "epochs": 15,        # Fixed for consistency across time bins
    "sg": 1              # Skip-gram: better for rare words & metaphor detection
}

# Terms
KEY_TERMS = ["sperm", "egg", "penetrate", "rescue", "attack", "bind", "fusion", "passive", "active"]

# === Load dataset ===
df = pd.read_csv(INPUT_FILE)
df["Publication Year"] = pd.to_numeric(df["Publication Year"], errors="coerce")
df = df.dropna(subset=["Publication Year", "Text"])

# === Define time bins (1980–2019 only) ===
bins = {
    "1980_1984": (1980, 1984),
    "1985_1989": (1985, 1989),
    "1990_1994": (1990, 1994),
    "1995_1999": (1995, 1999),
    "2000_2004": (2000, 2004),
    "2005_2009": (2005, 2009),
    "2010_2014": (2010, 2014),
    "2015_2019": (2015, 2019)
    
}

# === Train and save model for each bin ===
for label, (start_year, end_year) in bins.items():
    subset = df[(df["Publication Year"] >= start_year) & (df["Publication Year"] <= end_year)]
    sentences = [text.split() for text in subset["Text"] if isinstance(text, str)]

    if len(sentences) == 0:
        print(f"Skipping {label} — no text available.\n")
        continue

    print(f"Training Word2Vec model for {label} on {len(sentences)} documents using {W2V_PARAMS['epochs']} epochs...")

    model = Word2Vec(sentences=sentences, **W2V_PARAMS)

    # Log missing key terms
    missing = [w for w in KEY_TERMS if w not in model.wv.key_to_index]
    if missing:
        print(f" Missing terms in {label}: {missing}")

    # Save model
    model_path = os.path.join(OUTPUT_DIR, f"word2vec_{label}.model")
    model.save(model_path)
    print(f"Saved model to: {model_path}\n")
