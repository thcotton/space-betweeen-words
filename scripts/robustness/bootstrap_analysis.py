"""
Bootstrapping
"""
import os
import random
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from tqdm import tqdm
import multiprocessing

# === CONFIGURATION ===
DATA_DIR = "data/binned_data"
OUTPUT_CSV = "outputs/robustness/bootstrap_results.csv"
BOOTSTRAP_SAMPLES = 500
VECTOR_SIZE = 100
WINDOW = #set somewhere between 5-10. 10 matches the rest of the models.
MIN_COUNT = 2
SG = 1  # Skip-gram
#SEED = 42

# safe workers
safe_workers = max(1, multiprocessing.cpu_count() - 1)
print(f"Detected {safe_workers} safe workers.")

# === TERM GROUPS ===
term_groups = {
    "Sperm_Aggression": {
        "anchor": "sperm",
        "targets": ["penetrate", "assault", "attack", "harpoon", "burrow", "invade", "aggressive"]
    },
    "Sperm_Heroic": {
        "anchor": "sperm",
        "targets": ["mission", "journey", "quest"]
    },
    "Egg_Passive": {
        "anchor": "egg",
        "targets": ["sweep", "drift", "wait", "receive", "passive", "receptive"]
    },
    "Egg_Royalty": {
        "anchor": "egg",
        "targets": ["prize", "dormant", "corona", "vestments"]
    }
}

# === LOAD BINS ===
def load_binned_sentences(data_dir, bins):
    bin_data = {}
    for b in bins:
        fname = b + ".txt"
        with open(os.path.join(data_dir, fname), "r", encoding="utf-8") as f:
            lines = f.readlines()
            sentences = [line.strip().split() for line in lines if line.strip()]
            bin_data[b] = sentences
    return bin_data

# === TRAIN MODEL ===
def train_model(sentences):
    model = Word2Vec(
        sentences,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        sg=SG,
        seed=SEED,
        workers=safe_workers
    )
    print("Vocab size:", len(model.wv))
    return model

# === BOOTSTRAP ===
def bootstrap_drift_two_bins(bin_data, term_groups, samples):
    results = []

    base_bin = "1990_1994"
    comp_bin = "2015_2019"

    for group_name, group in term_groups.items():
        anchor = group["anchor"]
        for target in group["targets"]:
            diffs = []

            for _ in tqdm(range(samples), desc=f"{group_name}: {target}"):
                base_sample = random.choices(bin_data[base_bin], k=len(bin_data[base_bin]))
                comp_sample = random.choices(bin_data[comp_bin], k=len(bin_data[comp_bin]))

                base_model = train_model(base_sample)
                comp_model = train_model(comp_sample)

                if anchor in base_model.wv and target in base_model.wv:
                    base_sim = cosine_similarity(
                        base_model.wv[anchor].reshape(1, -1),
                        base_model.wv[target].reshape(1, -1)
                    )[0][0]
                else:
                    base_sim = np.nan

                if anchor in comp_model.wv and target in comp_model.wv:
                    comp_sim = cosine_similarity(
                        comp_model.wv[anchor].reshape(1, -1),
                        comp_model.wv[target].reshape(1, -1)
                    )[0][0]
                else:
                    comp_sim = np.nan

                if not np.isnan(base_sim) and not np.isnan(comp_sim):
                    diffs.append(comp_sim - base_sim)

            diffs = np.array(diffs)
            if len(diffs) > 0:
                p_val = (np.sum(diffs <= 0) + 1) / (len(diffs) + 1)
                drift = np.nanmean(diffs)
            else:
                p_val = np.nan
                drift = np.nan

            results.append({
                "Group": group_name,
                "Anchor": anchor,
                "Target": target,
                "Baseline_Bin": base_bin,
                "Compare_Bin": comp_bin,
                "Drift": drift,
                "P_Value": p_val
            })

    return pd.DataFrame(results)

# === MAIN EXECUTION ===
bins_to_use = ["1990_1994", "2015_2019"]
bin_data = load_binned_sentences(DATA_DIR, bins_to_use)
df_results = bootstrap_drift_two_bins(bin_data, term_groups, BOOTSTRAP_SAMPLES)
df_results.to_csv(OUTPUT_CSV, index=False)

print("Results saved to:", OUTPUT_CSV)