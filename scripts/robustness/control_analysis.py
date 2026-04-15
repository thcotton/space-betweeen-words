"""
Control Analysis (Messy)
Run cosine-similarity pipeline for frequency-matched neutral controls
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
from collections import defaultdict

# === Paths ===
MODEL_DIR = "outputs/models"                       
OUTPUT_DIR = "outputs/controls"                     
CONTROL_CSV = os.path.join(OUTPUT_DIR, "control_results.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Settings ===
BIN_ORDER = [
    "1980_1984", "1985_1989", "1990_1994",
    "1995_1999", "2000_2004", "2005_2009", "2010_2014", "2015_2019"
]
BASE_PERIOD = "1990_1994"
anchor_to_groups = {
    "sperm": ["Sperm_Aggression", "Sperm_Heroic"],
    "egg":   ["Egg_Passive", "Egg_Royalty"]
}

label_display = {
    "Sperm_Aggression": "Aggression (control)",
    "Sperm_Heroic": "Heroic Journey (control)",
    "Egg_Passive": "Passivity (control)",
    "Egg_Royalty": "Royalty (control)"
}
color_map = {
    "Sperm_Aggression": "#555555",
    "Sperm_Heroic": "#888888",
    "Egg_Passive": "#999999",
    "Egg_Royalty": "#AAAAAA"
}

# === Load control term mapping ===
df_controls = pd.read_csv(CONTROL_CSV)
group_to_terms = defaultdict(list)
term_group_map = {}
for _, r in df_controls.iterrows():
    g = str(r["target_group"])
    t = str(r["control_term"])
    group_to_terms[g].append(t)
    term_group_map[t] = g

# === Load models ===
models = {}
for label in BIN_ORDER:
    p = os.path.join(MODEL_DIR, f"word2vec_{label}.model")
    if os.path.exists(p):
        models[label] = Word2Vec.load(p)
    else:
        print(f"[warn] Missing model: {p}")

# === Cosine Similarity Calculation ===
def mean_std_cosine(anchor, group_terms, model):
    vals = [
        model.wv.similarity(anchor, term)
        for term in group_terms
        if anchor in model.wv and term in model.wv
    ]
    return (np.mean(vals) if vals else np.nan,
            np.std(vals) if vals else np.nan,
            len(vals))

# === Mean/Std Table ===
summary_rows = []
for anchor, groups in anchor_to_groups.items():
    for group in groups:
        row = {"anchor": anchor, "group": group}
        for bin_label in BIN_ORDER:
            if bin_label in models:
                mean, std, count = mean_std_cosine(anchor, group_to_terms[group], models[bin_label])
                row[f"{bin_label}_mean"] = mean
                row[f"{bin_label}_std"] = std
                row[f"{bin_label}_n"] = count
        summary_rows.append(row)
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "semantic_similarity_summary_controls.csv"), index=False)

# === Per-term Cosine Table ===
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
term_df.to_csv(os.path.join(OUTPUT_DIR, "per_term_similarity_trends_controls.csv"), index=False)

# === (3) Wide Mean Table for Plotting ===
wide_rows = []
for anchor, groups in anchor_to_groups.items():
    for group in groups:
        row = {"anchor~group": f"{anchor}~{group}"}
        for b in BIN_ORDER:
            row[b] = summary_df.loc[
                (summary_df.anchor==anchor) & (summary_df.group==group),
                f"{b}_mean"
            ].values[0]
        wide_rows.append(row)
wide_df = pd.DataFrame(wide_rows).set_index("anchor~group")[BIN_ORDER]
wide_df.to_csv(os.path.join(OUTPUT_DIR, "cosine_between_anchors_and_categories_controls.csv"))

# === Trend Plots ===
sns.set_theme(style="whitegrid")
BG = "#e1e1e3"
plot_dir = os.path.join(OUTPUT_DIR, "CosineAnchorCategory")
os.makedirs(plot_dir, exist_ok=True)

for anchor, groups in anchor_to_groups.items():
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_facecolor(BG)
    ax.grid(True, axis='y', color='white', linewidth=1)
    ax.grid(True, axis='x', color='white', linewidth=1)

    for group in groups:
        means, stds = [], []
        for b in BIN_ORDER:
            vals = [
                models[b].wv.similarity(anchor, t)
                for t in group_to_terms[group]
                if anchor in models[b].wv and t in models[b].wv
            ] if b in models else []
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if vals else 0)

        x = range(len(BIN_ORDER))
        ax.plot(x, means, marker='o', linewidth=3,
                label=label_display[group], color=color_map[group])
        ax.fill_between(x, np.array(means)-np.array(stds),
                        np.array(means)+np.array(stds),
                        color=color_map[group], alpha=0.2)

    ax.axvline(x=2, color='#EE3377', linestyle='--', linewidth=2,
               label="Martin's Critique (1991)")
    ax.set_title(f"Cosine Similarity Between {anchor.capitalize()} and Control Terms Over Time",
                 fontsize=20, fontweight='bold')
    ax.set_xlabel("5-Year Period")
    ax.set_ylabel("Average Cosine Similarity")
    ax.set_ylim(0, 0.5)
    plt.xticks(ticks=range(len(BIN_ORDER)), labels=[b.replace("_", "-") for b in BIN_ORDER], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"combinedplot_{anchor}_controls.png"))
    plt.close()

# === Delta Cosine Plot ===
delta_rows = []
for idx, row in wide_df.iterrows():
    anchor, group = idx.split("~")
    base_value = row[BASE_PERIOD]
    if pd.isna(base_value): continue
    for bin_label in BIN_ORDER:
        if bin_label == BASE_PERIOD or pd.isna(row[bin_label]): continue
        delta = row[bin_label] - base_value
        delta_rows.append({
            "Time Bin": bin_label.replace("_", "-"),
            "Category": f"{anchor.capitalize()}~{label_display[group]}",
            "Delta Cosine Similarity": delta,
            "Color": color_map[group]
        })
plot_df = pd.DataFrame(delta_rows)
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_facecolor(BG)
ax.grid(True, axis='y', color='white', linewidth=1)
ax.grid(True, axis='x', color='white', linewidth=1)
palette = plot_df.drop_duplicates("Category").set_index("Category")["Color"].to_dict()
sns.barplot(data=plot_df, x="Time Bin", y="Delta Cosine Similarity", hue="Category",
            palette=palette, ax=ax)
ax.axhline(0, color="black", linewidth=1)
ax.set_title("Change in Semantic Similarity (Controls) Relative to 1990–1994", fontsize=16, fontweight='bold')
ax.set_ylabel("Δ Cosine Similarity")
ax.set_xlabel("5-Year Period")
ax.set_ylim(-0.25, 0.25)
ax.tick_params(axis='x', rotation=45)
ax.legend(title="Anchor–Category Pair", bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "delta_cosine_similarity_controls.png"))
plt.close()

# === (6) Correlation Heatmap ===
mean_cols = [c for c in summary_df.columns if c.endswith("_mean")]
mean_only = summary_df[mean_cols].copy()
pretty_labels = [f"{r['anchor'].capitalize()}~{label_display[r['group']]}" for _, r in summary_df.iterrows()]
mean_only.index = pretty_labels
corr = mean_only.transpose().corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="viridis",
            linewidths=0.5, square=True, cbar_kws={"label": "Pearson Correlation"})
plt.title("Correlation Between Control Term Similarity Trends", fontsize=16, fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "semantic_similarity_correlation_heatmap_controls.png"))
plt.close()

print(f"All control-run outputs saved to: {OUTPUT_DIR}")
