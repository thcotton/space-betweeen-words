"""
Frquency trends
"""
"""
Plotting:
Plots three measures over time for metaphor categories:
  1) Raw term counts
  2) Frequency per million corpus words
  3) Relative share within tracked metaphor terms

For EACH measure, saves three plots:
  - sperm-only (Aggression + Heroic)
  - egg-only   (Passivity + Royalty)
  - combined   (all four categories)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# === File Paths ===
ABSOLUTE_PATH = "data/absolute_counts.csv"
PMW_PATH = "data/pmw_counts.csv"
RELATIVE_PATH = "data/relative_counts.csv"
OUTPUT_DIR = "outputs/frequency_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Group Mapping ===
group_map = {
    "penetrate": "Sperm_Aggression", "assault": "Sperm_Aggression", "attack": "Sperm_Aggression",
    "harpoon": "Sperm_Aggression",   "burrow": "Sperm_Aggression",  "invade": "Sperm_Aggression",
    "aggressive": "Sperm_Aggression",
    "mission": "Sperm_Heroic", "journey": "Sperm_Heroic", "quest": "Sperm_Heroic",
    "sweep": "Egg_Passive", "drift": "Egg_Passive", "wait": "Egg_Passive", "receive": "Egg_Passive",
    "passive": "Egg_Passive", "receptive": "Egg_Passive",
    "prize": "Egg_Royalty", "dormant": "Egg_Royalty", "corona": "Egg_Royalty", "vestments": "Egg_Royalty"
}

label_map = {
    "Sperm_Aggression": "Aggression",
    "Sperm_Heroic":     "Heroic Journey",
    "Egg_Passive":      "Passivity",
    "Egg_Royalty":      "Royalty"
}

color_map = {
    "Sperm_Aggression": "#CC3311",
    "Sperm_Heroic":     "#009988",
    "Egg_Passive":      "#0077BB",
    "Egg_Royalty":      "#EE7733",
}

# === Utility: compute relative share (from raw counts) if not present ===
def ensure_relative_share(raw_path: str, out_path: str):
    if os.path.exists(out_path):
        return out_path
    raw = pd.read_csv(raw_path, index_col=0)
    # divide each column (bin) by its column sum; handle 0-division by leaving zeros
    col_sums = raw.sum(axis=0)
    rel = raw.div(col_sums.where(col_sums != 0, other=1), axis=1)
    rel.to_csv(out_path)
    print(f"[OK] Relative share saved -> {out_path}")
    return out_path

RELATIVE_PATH = ensure_relative_share(ABSOLUTE_PATH, RELATIVE_PATH)

# === Plotter ===
def plot_trends(filepath: str, ylabel: str, filename_prefix: str):
    df = pd.read_csv(filepath, index_col=0)
    # attach groups
    df["group"] = df.index.map(group_map)
    df = df[df["group"].notna()]  # keep only mapped terms

    # identify time bins (all non-'group' columns), keep chronological order
    bins = [c for c in df.columns if c != "group"]
    try:
        bins = sorted(bins, key=lambda s: int(s.split("_")[0]))
    except Exception:
        # fallback to existing order
        pass

    sns.set_theme(style="whitegrid")
    subsets = {
        "sperm":    ["Sperm_Aggression", "Sperm_Heroic"],
        "egg":      ["Egg_Passive", "Egg_Royalty"],
        "combined": ["Sperm_Aggression", "Sperm_Heroic", "Egg_Passive", "Egg_Royalty"],
    }

    for subset_name, groups in subsets.items():
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_facecolor("#e1e1e3")
        ax.grid(True, which='major', axis='y', color='white', linestyle='-', linewidth=1, zorder=1)
        ax.grid(True, which='major', axis='x', color='white', linestyle='-', linewidth=1, zorder=1)

        for grp in groups:
            data = df[df["group"] == grp]
            if data.empty:
                continue
            means = data[bins].mean().astype(float)
            stds  = data[bins].std().fillna(0).astype(float)

            ax.plot(bins, means, marker="o", label=label_map.get(grp, grp),
                    color=color_map.get(grp, "#666666"), zorder=3)
            ax.fill_between(bins, (means - stds), (means + stds),
                            alpha=0.3, color=color_map.get(grp, "#666666"), zorder=2)

        # critique marker
        critique_bin = "1990_1994"
        if critique_bin in bins:
            ax.axvline(x=bins.index(critique_bin), color="#EE3377", linestyle="--",
                       linewidth=2, label="Martin's Critique (1991)", zorder=4)

        # labels & ticks
        ax.set_title(f"{ylabel} of Metaphorical Terms Over Time", fontsize=14, weight="bold")
        ax.set_xlabel("5-Year Period")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels([b.replace("_", "-") for b in bins], rotation=45)
        ax.legend(title="Metaphor Category")
        plt.tight_layout()

        out_name = f"{filename_prefix}_{subset_name}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, out_name))
        plt.close()
        print(f"[OK] Saved -> {out_name}")

# === Run all three measures, each with sperm/egg/combined ===
plot_trends(ABSOLUTE_PATH, ylabel="Raw Term Counts",               filename_prefix="absolute_frequency_trends")
plot_trends(PMW_PATH,      ylabel="Frequency per Million Words",   filename_prefix="normalized_pmw_trends")
plot_trends(RELATIVE_PATH, ylabel="Relative Share of Terms",       filename_prefix="relative_share_trends")
