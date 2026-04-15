"""
Agency axis
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

# === Agency Lexicons (from WordNet) ===

ACTIVE = {
    "active", "agent", "actor", "doer", "worker", "participating", 
    "hands-on", "proactive", "involved", "lively", "energetic", 
    "dynamic"
}

PASSIVE = {
    "inactive", "inactivity", "passive", "passivity", "passiveness", 
    "motionless", "static", "dormant", "hands-off", "resistless", 
    "unresisting", "supine", "idle", "unused", "inert", "sedentary", 
    "still", "abeyant", "hypoactive", "underactive", "torpid", 
    "sluggish", "soggy", "lifeless"
}


# === Helper Functions Below: ===

# === Load models ===
models = {}
for label in BIN_ORDER:
    path = os.path.join(MODEL_DIR, f"word2vec_{label}.model")
    if os.path.exists(path):
        models[label] = Word2Vec.load(path)

def mean_vector(model, terms):
    """Calculate mean vector for a set of terms (mean pooling)
    """
    vectors = [
        model.wv[term]
        for term in terms
        if term in model.wv
    ]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return None  # or raise an error

# === Normalize vector by unit length ===
def unit(vec):
    """Normalize a vector to unit length ... need this for agency axis...
    """
    norm = np.sqrt((np.sum(np.square(vec))))
    #listen I know there's a better way to do this with norm = np.linalg.norm(vec), but the way I wrote it (above) makes the math more clear to me
    # If the norm is zero, return the original vector
    if norm == 0:
        return vec
    # Divide the vector by its norm
    return vec / norm

# === CosSimilarity ===
def mean_std_cosine(anchor, group_terms, model):
    values = [
        model.wv.similarity(anchor, term)
        for term in group_terms
        if anchor in model.wv and term in model.wv
    ]
    return (np.mean(values) if values else np.nan, np.std(values) if values else np.nan, len(values))

# === Build Agency Axis ===

def make_agency_axis(model, agency_pos_terms, agency_neg_terms):
    """
    Build the agency axis vector from term sets
    """
    agency_pos_mean = mean_vector(model, agency_pos_terms)
    agency_neg_mean = mean_vector(model, agency_neg_terms)

    #guard against empty coverage
    if agency_pos_mean is None or agency_neg_mean is None:
        return None

    axis = agency_pos_mean - agency_neg_mean
    axis = unit(axis)
    return axis


# === Test 1 : Projection onto Agency Axis ===

def project_onto_axis(vec, axis):
    """
    Project a vector onto the agency axis
    Returns a scalar - reps. how much the vector aligns with agency
    """
    projection = np.dot(vec, unit(axis))
    # - Positive = more aligned with agency_positive
    # - Negative = more aligned with agency_negative
    return projection

def test_positioning_on_axis(model, agency_axis):
    """
    GOAL: To Show that sperm and egg are positioned differently on agency axis
    """
    sperm_vec = model.wv['sperm']
    egg_vec = model.wv['egg']
    sperm_project = project_onto_axis(sperm_vec,agency_axis)
    egg_project = project_onto_axis(egg_vec,agency_axis)
    difference = (sperm_project - egg_project)
    # Compare the projection scores
    # Consider maybe testing other agent/patient pairs for validation
    print(f"Sperm projection: {sperm_project}")
    print(f"Egg projection: {egg_project}")
    print(f"Difference: {difference}")
    if sperm_project > egg_project:
        print ("The literature generally treats sperm as more agential.")
    else:
        print ("Test does not show sperm is treated as more agential.")
    return sperm_project, egg_project
        
# === Test 2 : Is sperm-egg difference aligned with the agency axis? ===

def test_difference_alignment(model, agency_axis):
    """
    Goal: To Show that the difference between sperm and egg IS actually the agency dimension
    """
    sperm_vec = model.wv['sperm']
    egg_vec = model.wv['egg']
    difference_vec = (sperm_vec - egg_vec)
    cosine_sim = np.dot(unit(difference_vec), unit(agency_axis))
    # High similarity = agency is the primary difference between them
    print(f"Cosine similarity: {cosine_sim}")
    return cosine_sim

# === Validate ===
def validate_agency_axis(model, agency_axis):
    agent_patient_pairs = [('enzyme', 'substrate')]
    null_pairs = [('tissue', 'organ')]
    print("\n=== VALIDATION: Agent/Patient Pair ===")
    enzyme_vec = model.wv['enzyme']
    substrate_vec = model.wv['substrate']
    tissue_vec = model.wv['tissue']
    organ_vec = model.wv['organ']
    enzyme_project = project_onto_axis(enzyme_vec,agency_axis)
    substrate_project = project_onto_axis(substrate_vec,agency_axis)
    difference = (enzyme_project - substrate_project)
    print(f"Enzyme projection: {enzyme_project}")
    print(f"Substrate projection: {substrate_project}")
    print("\n=== VALIDATION: Null Pair ===")
    tissue_project = project_onto_axis(tissue_vec,agency_axis)
    organ_project = project_onto_axis(organ_vec,agency_axis)
    difference = (tissue_project - organ_project)
    print(f"Tissue projection: {tissue_project}")
    print(f"Organ projection: {organ_project}")

# ===  Run Across Bins ===

def report_coverage(model, terms, label):
    """Check how many terms are actually in the model"""
    in_vocab = [t for t in terms if t in model.wv]
    print(f"{label}: {len(in_vocab)}/{len(terms)} terms in vocab")
    print(f"Missing: {set(terms) - set(in_vocab)}")
    return in_vocab

def analyze_across_time():
    """
    Run all analyses across your different time bins
    """
    results = []
    
    for bin_id in BIN_ORDER:
        print(f"\n=== Analyzing {bin_id} ===")
        
        model = models[bin_id]
        
        # CHECK COVERAGE HERE:
        active_in_vocab = report_coverage(model, ACTIVE, "ACTIVE")
        passive_in_vocab = report_coverage(model, PASSIVE, "PASSIVE")
        
        agency_axis = make_agency_axis(model, ACTIVE, PASSIVE)
        if agency_axis is None:
            print("Skipping bin: agency axis undefined (coverage too low)")
            continue
        
        validate_agency_axis(model, agency_axis)
        
        # Run tests and capture results
        sperm_proj, egg_proj = test_positioning_on_axis(model, agency_axis)
        alignment = test_difference_alignment(model, agency_axis)
        
        # Store results (maybe add coverage counts too?)
        results.append({
            'bin': bin_id,
            'sperm_projection': sperm_proj,
            'egg_projection': egg_proj,
            'alignment': alignment,
            'active_coverage': len(active_in_vocab),  # Optional: track this
            'passive_coverage': len(passive_in_vocab)
        })
    
    return results
    
def main():
    results = analyze_across_time()
    
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, 'agency_analysis_results2.csv'), index=False)
    print(f"\nResults saved to {OUTPUT_DIR}/agency_analysis_results2.csv")
    print("\n=== SUMMARY ===")
    print(df)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, 'agency_analysis_results2.csv'), index=False)

if __name__ == "__main__":
    main()