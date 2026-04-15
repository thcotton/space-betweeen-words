"""
Heatmap
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
from collections import defaultdict

# === Paths ===
MODEL_DIR = "outputs/models"   
OUTPUT_DIR = "outputs/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Bins ===
BIN_ORDER = [
    "1980_1984", "1985_1989", "1990_1994",
    "1995_1999", "2000_2004", "2005_2009", "2010_2014", "2015_2019"
]

# === Term Groups ===
term_group_map = {
    "sperm": "Core_Nouns", "egg": "Core_Nouns", # Something here is funny, but it doesn't affect the output
    "penetrate": "Sperm_Aggression", "assault": "Sperm_Aggression", "attack": "Sperm_Aggression",
    "harpoon": "Sperm_Aggression", "burrow": "Sperm_Aggression", "invade": "Sperm_Aggression", "aggressive": "Sperm_Aggression",
    "mission": "Sperm_Heroic", "journey": "Sperm_Heroic", "quest": "Sperm_Heroic",
    "sweep": "Egg_Passive", "drift": "Egg_Passive", "wait": "Egg_Passive", "receive": "Egg_Passive",
    "passive": "Egg_Passive", "receptive": "Egg_Passive",
    "prize": "Egg_Royalty", "dormant": "Egg_Royalty", "corona": "Egg_Royalty", "vestments": "Egg_Royalty"
}

# === Anchor-to-group mapping ===
anchor_to_groups = {
    "sperm": ["Sperm_Aggression", "Sperm_Heroic"],
    "egg": ["Egg_Passive", "Egg_Royalty"]
}

# === Colors === 
#There really aren't enough colorblind-friendly color schemes. I think this one is friendly?
color_map = {
    "Sperm_Aggression": "#17becf",
    "Sperm_Heroic": "#ff7f0e",
    "Egg_Passive": "#e377c2",
    "Egg_Royalty": "#2ca02c"
}

label_display = {
    "Sperm_Aggression": "Aggression",
    "Sperm_Heroic": "Heroic Journey",
    "Egg_Passive": "Passivity",
    "Egg_Royalty": "Royalty"
}

# === Reverse group map ===
group_to_terms = defaultdict(list)
for term, group in term_group_map.items():
    group_to_terms[group].append(term)

# === Load models ===
models = {}
for label in BIN_ORDER:
    path = os.path.join(MODEL_DIR, f"word2vec_{label}.model")
    if os.path.exists(path):
        models[label] = Word2Vec.load(path)

# === Cosine Similarity Calculation ===
def mean_std_cosine(anchor, group_terms, model):
    values = [
        model.wv.similarity(anchor, term)
        for term in group_terms
        if anchor in model.wv and term in model.wv
    ]
    return (np.mean(values) if values else np.nan, np.std(values) if values else np.nan, len(values))

# === Mean and Std Cosine Table ===
summary_rows = []
for anchor, groups in anchor_to_groups.items():
    for group in groups:
        group_terms = group_to_terms[group]
        row = {"anchor": anchor, "group": group}
        for bin_label in BIN_ORDER:
            if bin_label in models:
                mean, std, count = mean_std_cosine(anchor, group_terms, models[bin_label])
                row[f"{bin_label}_mean"] = mean
                row[f"{bin_label}_std"] = std
                row[f"{bin_label}_n"] = count
        summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "semantic_similarity_summary.csv"), index=False)

# === Correlation Heatmap ===
mean_only = summary_df[[col for col in summary_df.columns if col.endswith("_mean")]]
pretty_labels = [f"{r['anchor'].capitalize()}~{label_display[r['group']]}" for _, r in summary_df.iterrows()]
mean_only.index = pretty_labels

# Define logical order for rows/columns
desired_order = [
    "Sperm~Aggression", "Sperm~Heroic Journey",
    "Egg~Passivity", "Egg~Royalty"
]

# Reorder and compute correlations
mean_only = mean_only.reindex(desired_order)
corr = mean_only.transpose().corr()
corr = corr.loc[desired_order, desired_order]

# Plot heatmap 
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    linewidths=0.5,
    square=True,
    cbar_kws={"label": "Pearson Correlation"}
)
plt.title("Correlation Between Similarity Trends Across Anchor–Category Pairs", fontsize=16, fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "semantic_similarity_correlation_heatmap.png"))
plt.close()

# === Per-Term Cosine Table ===
term_rows = []
for anchor, groups in anchor_to_groups.items():
    for group in groups:
        for term in group_to_terms[group]:
            for bin_label in BIN_ORDER:
                model = models.get(bin_label)
                if model and anchor in model.wv and term in model.wv:
                    sim = model.wv.similarity(anchor, term)
                    term_rows.append({
                        "anchor": anchor,
                        "term": term,
                        "group": group,
                        "bin": bin_label,
                        "similarity": sim
                    })

term_df = pd.DataFrame(term_rows)
term_df.to_csv(os.path.join(OUTPUT_DIR, "per_term_similarity_trends.csv"), index=False)

# === Placeholder for Frequency Table ===
#dw about this
freq_df = pd.DataFrame(columns=["term", "group"] + BIN_ORDER)
for group, terms in group_to_terms.items():
    for term in terms:
        row = {"term": term, "group": group}
        for bin in BIN_ORDER:
            row[bin] = np.nan  # Replace with actual frequencies later
        freq_df = pd.concat([freq_df, pd.DataFrame([row])], ignore_index=True)

freq_df.to_csv(os.path.join(OUTPUT_DIR, "term_frequencies_placeholder.csv"), index=False)

print("Outputs saved to:", OUTPUT_DIR)
