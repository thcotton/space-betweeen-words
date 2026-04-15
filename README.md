\# The Space Between Words – Code Repository

This repository contains the code used to generate the analyses reported in the paper.**

## Repository Structure

### Core

\- Model training: `scripts/train_models.py`

### Analysis

\- Frequency trends: `scripts/analysis/frequency_trends.py`

\- Normalized frequencies: `scripts/analysis/normalized_frequency.py`

\- Semantic drift: `scripts/analysis/pairwise_comparisons.py`

\- Correlation heatmap of similarity trends: `scripts/analysis/heatmap.py`

\- Axis analysis: `scripts/analysis/agency_axis.py`

### Robustness

\- Bootstrapping: `scripts/robustness/bootstrap_analysis.py`

\- Cosine similarity analysis on matched control terms: `scripts/robustness/control_analysis.py`

# Preprocessing&Helpers
\- PDF to text : `scripts/preprocessing/helper_extract_pdf_to_text.py`

\- tokenization and lemmatization: `scripts/preprocessing/clean_and_lemmatize.txt`

\- Frequency-matched control term selection: `scripts/robustness/helper_control_term_selector.py`

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
