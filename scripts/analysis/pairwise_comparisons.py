"""
Pairwise vector comps.
"""
"""
Standard (alignment-based) with period-specific neighbor clouds
- Loads two Word2Vec models (1990–1994, 2015–2019)
- Aligns 2015–2019 onto 1990–1994 via Orthogonal Procrustes
- Computes single-word drift and relational drift (Δcosine anchor–target)
- Saves metrics to Excel
- Plots PCA with unlabeled nearest neighbors from each period as background
"""

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from gensim.models import Word2Vec
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#=== Reproducibility ===
random.seed(42)
np.random.seed(42)

#=== Paths ===
MODEL_DIR = "outputs/models"
OUTPUT_DIR = "outputs/results"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

#=== Models (align 2015–2019 onto 1990–1994) ===
MODELS = {
    "1990–1994": "word2vec_1990_1994.model",
    "2015–2019": "word2vec_2015_2019.model",
}

#=== Anchor/Target sets ===
SPERM_TERMS = {
    "penetrate": "Sperm_Aggression", "assault": "Sperm_Aggression", "attack": "Sperm_Aggression",
    "harpoon": "Sperm_Aggression", "burrow": "Sperm_Aggression", "invade": "Sperm_Aggression", "aggressive": "Sperm_Aggression",
    "mission": "Sperm_Heroic", "journey": "Sperm_Heroic", "quest": "Sperm_Heroic"
}
EGG_TERMS = {
    "sweep": "Egg_Passive", "drift": "Egg_Passive", "wait": "Egg_Passive", "receive": "Egg_Passive",
    "passive": "Egg_Passive", "receptive": "Egg_Passive",
    "prize": "Egg_Royalty", "dormant": "Egg_Royalty", "corona": "Egg_Royalty", "vestments": "Egg_Royalty"
}
ANCHORS = ["sperm", "egg"]

#=== Plot toggles ===
MAKE_PLOTS = True            
N_CONTEXT = 200                 
FIGSIZE = (8, 8)                

#=== Helpers ===
def safe_name(s: str) -> str:
    return s.replace("–", "_").replace("—", "_").replace(" ", "_")

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return np.nan
    return float(np.dot(a, b) / (na * nb))

def build_matrix_for_words(model, words):
    vecs = []
    keep = []
    for w in words:
        if w in model.wv:
            vecs.append(model.wv[w])
            keep.append(w)
    if not vecs:
        return np.zeros((0, model.wv.vectors.shape[1])), []
    return np.vstack(vecs), keep

def get_neighbors(word, model, topn):
    #return topn nearest neighbor words from the given model
    try:
        return [w for (w, sim) in model.wv.most_similar(word, topn=topn)]
    except KeyError:
        return []

#=== Load models ===
m90 = Word2Vec.load(os.path.join(MODEL_DIR, MODELS["1990–1994"]))
m15 = Word2Vec.load(os.path.join(MODEL_DIR, MODELS["2015–2019"]))

#=== Shared vocab (intersection, optionally cap by rank for stability) ===
v90 = set(m90.wv.index_to_key)
v15 = set(m15.wv.index_to_key)
shared_vocab = [w for w in m90.wv.index_to_key if w in v15]
TOPK = 50000  
shared_vocab = shared_vocab[:min(TOPK, len(shared_vocab))]

#=== Build embedding matrices (same word order) ===
X90, keep90 = build_matrix_for_words(m90, shared_vocab)    
X15, keep15 = build_matrix_for_words(m15, shared_vocab)    
assert keep90 == keep15, "Shared vocab alignment mismatch"

#=== Orthogonal Procrustes: find R mapping 2015–2019 -> 1990–1994 ===
# Solve: X15 @ R ≈ X90  (R is orthogonal)
if X15.shape[0] == 0 or X90.shape[0] == 0:
    raise RuntimeError("No shared vocabulary available for alignment.")
R, _ = orthogonal_procrustes(X15, X90)

def get_vec_1990(w):
    return m90.wv[w] if w in m90.wv else None

def get_vec_2015_aligned(w):
    if w not in m15.wv: return None
    return m15.wv[w] @ R

#=== Build (anchor,target,category) list ===
ALL_TARGETS = {**SPERM_TERMS, **EGG_TERMS}
pairs = []
for anchor in ANCHORS:
    targets = SPERM_TERMS if anchor == "sperm" else EGG_TERMS
    for target, category in targets.items():
        pairs.append((anchor, target, category))

#=== Compute metrics ===
records = []
for anchor, target, category in pairs:
    v_a_90 = get_vec_1990(anchor)
    v_t_90 = get_vec_1990(target)
    v_a_15 = get_vec_2015_aligned(anchor)
    v_t_15 = get_vec_2015_aligned(target)

    if any(v is None for v in [v_a_90, v_t_90, v_a_15, v_t_15]):
        records.append({
            "Anchor": anchor, "Target": target, "Category": category,
            "HasAllVectors": False
        })
        continue

    drift_anchor = 1 - cosine_sim(v_a_90, v_a_15)                
    drift_target = 1 - cosine_sim(v_t_90, v_t_15)                

    cos_90 = cosine_sim(v_a_90, v_t_90)                          
    cos_15 = cosine_sim(v_a_15, v_t_15)                         
    delta_cos = cos_15 - cos_90                                  

    euclid_90 = float(np.linalg.norm(v_a_90 - v_t_90))          
    euclid_15 = float(np.linalg.norm(v_a_15 - v_t_15))          
    delta_euclid = euclid_15 - euclid_90                       

    records.append({
        "Anchor": anchor,
        "Target": target,
        "Category": category,
        "HasAllVectors": True,
        "Cosine_1990_1994": cos_90,
        "Cosine_2015_2019_aligned": cos_15,
        "ΔCosine(RelationalDrift)": delta_cos,
        "Euclid_1990_1994": euclid_90,
        "Euclid_2015_2019_aligned": euclid_15,
        "ΔEuclid": delta_euclid,
        "Drift_Anchor(cosine_distance)": drift_anchor,
        "Drift_Target(cosine_distance)": drift_target
    })

#=== Save metrics ===
df = pd.DataFrame(records)
excel_path = os.path.join(OUTPUT_DIR, "alignment_based_semantic_drift_metrics.xlsx")
df.to_excel(excel_path, index=False)
print(f"Saved metrics to: {excel_path}")

#=== Plots: PCA drift arrows + unlabeled nearest neighbors per period ===
# This is one of those weird not pretty things I mentioned in the README
def make_plot_for_pair(anchor, target, category):
    v_a_90 = get_vec_1990(anchor)
    v_t_90 = get_vec_1990(target)
    v_a_15 = get_vec_2015_aligned(anchor)
    v_t_15 = get_vec_2015_aligned(target)
    if any(v is None for v in [v_a_90, v_t_90, v_a_15, v_t_15]):
        return

    # Collect nearest neighbors for each period and word
    topn = max(1, N_CONTEXT // 2)  #half for anchor, half for target, per period
    nbrs_a_90 = get_neighbors(anchor, m90, topn)
    nbrs_t_90 = get_neighbors(target, m90, topn)
    nbrs_a_15 = get_neighbors(anchor, m15, topn)
    nbrs_t_15 = get_neighbors(target, m15, topn)

    # Deduplicate within each set and exclude focal words
    ctx_90 = list({w for w in (nbrs_a_90 + nbrs_t_90) if w not in {anchor, target}})
    ctx_15 = list({w for w in (nbrs_a_15 + nbrs_t_15) if w not in {anchor, target}})

    # Build matrix for PCA: 1990 neighbors in 1990 space, 2015 neighbors in aligned space
    mat = []
    labels = []

    # 1990 neighbor vectors
    for w in ctx_90:
        v = get_vec_1990(w)
        if v is None: 
            continue
        mat.append(v); labels.append(("__ctx90__", "ctx90"))

    # 2015 neighbor vectors (aligned)
    for w in ctx_15:
        v = get_vec_2015_aligned(w)
        if v is None: 
            continue
        mat.append(v); labels.append(("__ctx15__", "ctx15"))

    # Anchor/Target at each time
    mat.extend([v_a_90, v_t_90, v_a_15, v_t_15])
    labels.extend([(anchor, "1990–1994"), (target, "1990–1994"),
                   (anchor, "2015–2019"), (target, "2015–2019")])

    if len(mat) < 4:
        return

    M = np.vstack(mat)
    pca = PCA(n_components=2, random_state=42)
    P = pca.fit_transform(M)

    # Split back into groups
    pts_ctx90 = []
    pts_ctx15 = []
    pts = dict()
    for (w, tag), pt in zip(labels, P):
        if tag == "ctx90":
            pts_ctx90.append(pt)
        elif tag == "ctx15":
            pts_ctx15.append(pt)
        else:
            pts.setdefault(w, dict())
            pts[w][tag] = pt

    # Plot
    fig = plt.figure(figsize=FIGSIZE, facecolor='black')
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_facecolor('black')
    ax.axis('off')

    # Draw neighbor clouds (unlabeled, labels were too messy)
    if len(pts_ctx90):
        pts_ctx90 = np.vstack(pts_ctx90)
        ax.scatter(pts_ctx90[:,0], pts_ctx90[:,1], s=18, alpha=0.18, color='white')  #1990 neighbors

    if len(pts_ctx15):
        pts_ctx15 = np.vstack(pts_ctx15)
        ax.scatter(pts_ctx15[:,0], pts_ctx15[:,1], s=18, alpha=0.18, color='white')  #2015 neighbors

    # Draw 1990 points
    for w in [anchor, target]:
        if "1990–1994" in pts.get(w, {}):
            x, y = pts[w]["1990–1994"]
            ax.scatter(x, y, s=120, color='white', edgecolors='black', zorder=3)
            ax.text(x+0.06, y+0.06, f"{w} (90s)", color='white', fontsize=11, weight='bold')

    # Draw 2015 points
    for w in [anchor, target]:
        if "2015–2019" in pts.get(w, {}):
            x, y = pts[w]["2015–2019"]
            ax.scatter(x, y, s=120, color='white', edgecolors='black', zorder=3)
            ax.text(x+0.06, y+0.06, f"{w} (15–19)", color='white', fontsize=11, weight='bold')

    # Arrows showing drift
    for w in [anchor, target]:
        if "1990–1994" in pts.get(w, {}) and "2015–2019" in pts.get(w, {}):
            x0, y0 = pts[w]["1990–1994"]
            x1, y1 = pts[w]["2015–2019"]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color="white", lw=3))

    title = f"Alignment-Based Drift with Period Neighbors: {anchor} & {target}  ({category})"
    fig.text(0.5, 0.94, title, ha='center', fontsize=15, color='white', weight='bold')
    fname = f"aligned_drift_neighbors_{safe_name(anchor)}_{safe_name(target)}.png"
    fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300, facecolor='black')
    plt.close()

if MAKE_PLOTS:
    for anchor, target, category in pairs:
        make_plot_for_pair(anchor, target, category)
