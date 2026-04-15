"""
Control term selection based on frequency matching
"""

import os, re
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

# === Paths ===
MODEL_DIR = "outputs/models"   
OUTPUT_DIR = "outputs/controls"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Time bins / model filenames ===
BIN_ORDER = ["1980_1984","1985_1989","1990_1994","1995_1999","2000_2004","2005_2009","2010_2014","2015_2019"]
MODEL_NAME = "word2vec_{bin}.model"

# === Your metaphor terms (targets to match)
group_map = {
    "penetrate":"Sperm_Aggression","assault":"Sperm_Aggression","attack":"Sperm_Aggression",
    "harpoon":"Sperm_Aggression","burrow":"Sperm_Aggression","invade":"Sperm_Aggression","aggressive":"Sperm_Aggression",
    "mission":"Sperm_Heroic","journey":"Sperm_Heroic","quest":"Sperm_Heroic",
    "sweep":"Egg_Passive","drift":"Egg_Passive","wait":"Egg_Passive","receive":"Egg_Passive",
    "passive":"Egg_Passive","receptive":"Egg_Passive",
    "prize":"Egg_Royalty","dormant":"Egg_Royalty","corona":"Egg_Royalty","vestments":"Egg_Royalty",
    "sperm":"Core_Nouns","egg":"Core_Nouns"
}

# === Control selection params ===
TOLERANCE = 0.10         # ±10% mean-frequency window
N_MATCHES = 5            # how many controls per target term
MIN_NONZERO_BINS = 6     # require presence in ≥ this many bins
SEED = 13
rng = np.random.default_rng(SEED)

# === Domain stoplist (exclude reproduction-ish vocab)
DOMAIN_PATTERNS = [
    r"sperm",r"egg",r"ovum",r"oocyte",r"zygote",r"embryo",r"gamet",
    r"spermat",r"fertil",r"inseminat",r"follic",r"oviduct",r"uter",
    r"placent",r"gonad",r"androgen",r"estrogen",r"oestrogen",r"semen",
    r"oogenesis",r"testis",r"ovary"
]
def is_domain_term(term:str)->bool:
    t = term.lower()
    for p in DOMAIN_PATTERNS:
        if re.search(p, t):
            return True
    return False

# === Helper to get raw counts from a model (gensim 4.x or fallback 3.x)
def get_count(model: Word2Vec, word: str) -> int:
    try:
        return int(model.wv.get_vecattr(word, "count"))  # gensim >= 4.0
    except Exception:
        try:
            return int(model.wv.vocab[word].count)        # gensim 3.x fallback
        except Exception:
            return 0

# === Load models and build per-million frequency table
models = {}
for b in BIN_ORDER:
    path = os.path.join(MODEL_DIR, MODEL_NAME.format(bin=b))
    if os.path.exists(path):
        models[b] = Word2Vec.load(path)
    else:
        print(f"[warn] Missing model for {b}: {path}")

# union vocabulary across bins
vocab = set()
for m in models.values():
    vocab.update(m.wv.index_to_key)

# per-bin totals (sum of counts across vocab)
bin_totals = {}
for b, m in models.items():
    total = 0
    # summing all counts directly is faster by vectorized grab, but loop is fine
    for w in m.wv.index_to_key:
        total += get_count(m, w)
    bin_totals[b] = max(total, 1)  # guard divide-by-zero

# build freq (per milli) dataframe
freq_df = pd.DataFrame(index=sorted(vocab), columns=BIN_ORDER, dtype=float)
for b, m in models.items():
    tot = float(bin_totals[b])
    col = []
    for w in freq_df.index:
        c = get_count(m, w) if w in m.wv else 0
        col.append((c / tot) * 1_000_000.0)
    freq_df[b] = col

# Quick stats
nonzero_counts = (freq_df[BIN_ORDER] > 0).sum(axis=1)
means = freq_df[BIN_ORDER].mean(axis=1)

# === Build candidate universe: not metaphor terms, not domain, present enough
metaphor_terms = set(group_map.keys())
universe = [
    t for t in freq_df.index
    if t not in metaphor_terms
    and not is_domain_term(t)
    and nonzero_counts.loc[t] >= MIN_NONZERO_BINS
]

# === Per-term matching
rows_term = []
for target in metaphor_terms:
    if target not in freq_df.index:
        # skip targets not in any model vocab
        continue
    if nonzero_counts.loc[target] < MIN_NONZERO_BINS:
        # target too sparse to match reliably
        continue
    m_t = float(means.loc[target])
    if m_t <= 0:
        continue

    low, high = m_t*(1.0 - TOLERANCE), m_t*(1.0 + TOLERANCE)
    pool = [u for u in universe if low <= means.loc[u] <= high]

    if not pool:
        # widen window a bit if empty
        low2, high2 = m_t*0.85, m_t*1.15
        pool = [u for u in universe if low2 <= means.loc[u] <= high2]

    if not pool:
        continue

    cand = pd.DataFrame({
        "control_term": pool,
        "mean_freq": means.loc[pool].values
    })
    cand["abs_diff"] = (cand["mean_freq"] - m_t).abs()
    cand = cand.sort_values(["abs_diff","mean_freq"])

    top_slice = cand.head(max(20, N_MATCHES*3))
    choose = top_slice.sample(n=min(N_MATCHES, len(top_slice)), random_state=SEED)

    for _, r in choose.iterrows():
        rows_term.append({
            "target_term": target,
            "target_group": group_map.get(target, "NA"),
            "control_term": r["control_term"],
            "target_mean": m_t,
            "control_mean": float(r["mean_freq"]),
            "rel_diff": float((r["mean_freq"] - m_t)/m_t)
        })

per_term_df = pd.DataFrame(rows_term)
per_term_path = os.path.join(OUTPUT_DIR, "neutral_controls_per_term_from_models.csv")
per_term_df.to_csv(per_term_path, index=False)
print("Wrote:", per_term_path)

# === Per-category matching (match to category's average mean freq)
group_to_terms = {}
for t,g in group_map.items():
    if t in freq_df.index and nonzero_counts.loc[t] >= MIN_NONZERO_BINS:
        group_to_terms.setdefault(g, []).append(t)

rows_cat = []
for g, terms in group_to_terms.items():
    if not terms:
        continue
    m_cat = float(means.loc[terms].mean())
    if m_cat <= 0:
        continue

    low, high = m_cat*(1.0 - TOLERANCE), m_cat*(1.0 + TOLERANCE)
    pool = [u for u in universe if low <= means.loc[u] <= high]

    if not pool:
        low2, high2 = m_cat*0.85, m_cat*1.15
        pool = [u for u in universe if low2 <= means.loc[u] <= high2]

    if not pool:
        continue

    cand = pd.DataFrame({
        "control_term": pool,
        "mean_freq": means.loc[pool].values
    })
    cand["abs_diff"] = (cand["mean_freq"] - m_cat).abs()
    cand = cand.sort_values(["abs_diff","mean_freq"])

    k = min(len(terms), max(N_MATCHES, 5))
    top_slice = cand.head(max(30, k*3))
    choose = top_slice.sample(n=min(k, len(top_slice)), random_state=SEED)

    for _, r in choose.iterrows():
        rows_cat.append({
            "target_category": g,
            "control_term": r["control_term"],
            "category_mean": m_cat,
            "control_mean": float(r["mean_freq"]),
            "rel_diff": float((r["mean_freq"] - m_cat)/m_cat)
        })

per_cat_df = pd.DataFrame(rows_cat)
per_cat_path = os.path.join(OUTPUT_DIR, "neutral_controls_per_category_from_models.csv")
per_cat_df.to_csv(per_cat_path, index=False)
print("Wrote:", per_cat_path)

# === Sanity checks
if not per_term_df.empty:
    print("Per-term unique control terms:", per_term_df["control_term"].nunique())
    print(per_term_df.groupby("target_group")["control_term"].nunique())
if not per_cat_df.empty:
    print("Per-category control counts:")
    print(per_cat_df.groupby("target_category")["control_term"].nunique()