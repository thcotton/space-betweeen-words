\# The Space Between Words – Code Repository

This repository contains the code used to generate the analyses reported in the paper.**

## Repository Structure

###Core

\- scripts/train_models.py — trains Word2Vec models across bins

###Analysis

\- scripts/analysis/frequency_trends.py — raw, normalized, and relative frequency trends

\- scripts/analysis/normalized_frequency.py — visualization of normalized frequencies

\- scripts/analysis/pairwise_comparisons.py — alignment-based semantic drift analysis

\- scripts/analysis/heatmap.py — correlation heatmap of similarity trends

\- scripts/analysis/agency_axis.py — agency–passivity axis analysis

###Robustness

\-scripts/robustness/bootstrap_analysis.py — bootstrapping for pairwise drift

\-scripts/robustness/control_analysis.py — cosine similarity analysis on matched control terms

#Preprocessing&Helpers
\-scripts/preprocessing/helper_extract_pdf_to_text.py — PDF text extraction

\-scripts/preprocessing/clean_and_lemmatize.txt — tokenization and lemmatization

\-scripts/robustness/helper_control_term_selector.py — frequency-matched control term selection

\- Semantic Axis: `scripts/vector_arithmetic_agency_axis.py`

## Notes
This repository represents the research workflow used to generate the reported results. It is not intended as a polished software package. Regardless, this should get someone where they need to produce the analyses described in the paper, should they want to. Please note that for anonymity reasons, the file paths in the scripts have been replaced with generic placeholders (e.g., data/ and outputs/). Users should update these paths to match their local setup and provide the necessary input data.


## Training parameters

All Word2Vec models were trained using a consistent set of hyperparameters:

```text
vector_size: 100        # dimensionality of word vectors
window: 10              # context window size
min_count: 2            # minimum frequency for inclusion
workers: half CPU cores # parallel worker threads (minimum 1)
epochs: 15              # training iterations
sg: 1                   # skip-gram architecture

# The-Space-Between-Words-GitHub-Repo
