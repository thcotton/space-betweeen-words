"""
Normalized Freq Trends
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === File Paths ===
INPUT_PATH = "data/normalized_frequency.csv"
OUTPUT_DIR = "outputs/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load Data ===
df = pd.read_csv(INPUT_PATH, index_col=0)  # Use first column as term names

# === Group Mapping ===
group_map = {
    "penetrate": "Sperm_Aggression", "assault": "Sperm_Aggression", "attack": "Sperm_Aggression",
    "harpoon": "Sperm_Aggression", "burrow": "Sperm_Aggression", "invade": "Sperm_Aggression",
    "aggressive": "Sperm_Aggression",
    "mission": "Sperm_Heroic", "journey": "Sperm_Heroic", "quest": "Sperm_Heroic",
    "sweep": "Egg_Passive", "drift": "Egg_Passive", "wait": "Egg_Passive", "receive": "Egg_Passive",
    "passive": "Egg_Passive", "receptive": "Egg_Passive",
    "prize": "Egg_Royalty", "dormant": "Egg_Royalty", "corona": "Egg_Royalty", "vestments": "Egg_Royalty"
}

color_map = {
    "Sperm_Aggression": "#CC3311",
    "Sperm_Heroic":     "#009988",
    "Egg_Passive":      "#0077BB",
    "Egg_Royalty":      "#EE7733",
}

label_map = {
    "Sperm_Aggression": "Aggression",
    "Sperm_Heroic": "Heroic Journey",
    "Egg_Passive": "Passivity",
    "Egg_Royalty": "Royalty"
}

# === Attach group column ===
df["group"] = df.index.map(group_map)

# === Identify time bins (all columns except the new 'group' column)
bins = [col for col in df.columns if col != "group"]

# === Plotting ===
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_facecolor("#e1e1e3")  # Optional: gray background
ax.grid(True, which='major', axis='y', color='white', linestyle='-', linewidth=1, zorder=1)
ax.grid(True, which='major', axis='x', color='white', linestyle='-', linewidth=1, zorder=1)

grouped = df.groupby("group")

for group, data in grouped:
    means = data[bins].mean()
    stds = data[bins].std()
    label = label_map.get(group, group)
    color = color_map.get(group, "#888888")

    ax.plot(bins, means, marker="o", label=label, color=color, zorder=3)
    ax.fill_between(bins, means - stds, means + stds, alpha=0.3, color=color, zorder=2)

# === Add vertical line for Martin's critique
critique_bin = "1990_1994"
if critique_bin in bins:
    critique_index = bins.index(critique_bin)
    ax.axvline(
        x=critique_index,
        color="#EE3377",
        linestyle="--",
        linewidth=2,
        label="Martin's Critique (1991)",
        zorder=4
    )

# === Final plot ===
ax.set_title("Normalized Frequency of Metaphorical Terms Over Time", fontsize=14, weight="bold")
ax.set_xlabel("5-Year Period")
ax.set_ylabel("Frequency per Million Words")
ax.set_xticks(range(len(bins)))

# FIX: replace underscores with en dashes for prettier date labels
xtick_labels = [label.replace("_", "-") for label in bins]
ax.set_xticklabels(xtick_labels, rotation=45)

plt.legend(title="Metaphor Category")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "normalized_frequency_trends.png"))
plt.show()

