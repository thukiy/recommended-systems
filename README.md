# Food.com Recipe Recommender

**Course:** CDS121 - Recommender Systems  
**Institution:** Fachhochschule Graubünden (FHGR)  
**Team:** Isabelle, Philip, Thuvaraka  

## Project Overview
This project builds and evaluates a recommender system for Food.com recipes.  
The goal is to recommend relevant recipes from historical user preferences while comparing different recommendation paradigms under a strict offline evaluation setup.

Our current pipeline covers:
- data exploration on raw Food.com data
- conversion of explicit ratings into positive implicit feedback
- strict temporal leave-last-out evaluation
- popularity, random, trending, matrix factorization, and content-based baselines

## Dataset
We use the **Food.com Recipes and User Interactions** dataset from Kaggle:

https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions

For our project, we work with the raw files:
- `RAW_interactions.csv`
- `RAW_recipes.csv`

### Why This Dataset
- It contains long-term user histories from 2000 to 2018.
- It includes both interaction data and rich recipe metadata.
- It is well suited for comparing collaborative filtering and content-based methods.

### Modeling Assumption
Although the dataset contains explicit ratings, we currently transform it into a Top-N recommendation problem:
- positive feedback: `rating >= 4`
- users with fewer than 2 positive interactions are excluded from evaluation

### Important
The dataset is too large for version control and is ignored via `.gitignore`.

To run the project locally:
1. Download the dataset from Kaggle.
2. Extract the archive.
3. Place `RAW_interactions.csv` and `RAW_recipes.csv` into `data/raw/`.

## Repository Structure

```text
recommended-systems/
│
├── data/
│   └── raw/
│       ├── RAW_interactions.csv
│       └── RAW_recipes.csv
│
├── evaluation/
│   ├── metrics.py                  # Recall@K, NDCG@K, Coverage@K, Novelty@K
│   └── split.py                    # Temporal leave-last-out split
│
├── models/
│   ├── baselines.py                # Popularity, Random, Trending
│   ├── content_based.py            # TF-IDF content-based recommender + hybrid scaffold
│   ├── knn.py                      # Item-item kNN
│   └── mf.py                       # Biased MF and BPR MF
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Exploration of interactions + recipe metadata
│   └── 02_baseline_evaluation.ipynb # Baseline training, evaluation, diagnostics
│
├── runs/
│   └── runs.csv                    # Experiment logging
│
├── environment.yml
├── requirements.txt
└── README.md
```

## Setup

### Conda / Mamba
```bash
conda env create -f environment.yml
conda activate recommended-systems
```

or

```bash
mamba env create -f environment.yml
conda activate recommended-systems
```

### Pip
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How To Run

### 1. Data Exploration
Open and run:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

This notebook explores:
- rating distribution
- sparsity
- cold users / cold items
- metadata quality
- join coverage between interactions and recipes
- candidate interaction rules for modeling

### 2. Baseline Evaluation
Open and run:

```bash
jupyter notebook notebooks/02_baseline_evaluation.ipynb
```

This notebook currently:
- loads `RAW_interactions.csv` and `RAW_recipes.csv`
- creates positive implicit feedback from `rating >= 4`
- applies a strict temporal leave-last-out split
- drops users with tied final timestamps to avoid ambiguous train/test boundaries
- trains baseline recommenders
- computes Recall@20, NDCG@20, Coverage@20, and Novelty@20
- logs runs to `runs/runs.csv`

## Current Results

Current strict Food.com baseline results (`K=20`):

| Model | Recall@20 | NDCG@20 | Coverage@20 | Novelty@20 |
| :--- | :--- | :--- | :--- | :--- |
| Popularity | 0.0312 | 0.0125 | 0.0367% | 10.3203 |
| Trending (180d) | 0.0029 | 0.0018 | 0.0173% | 15.6854 |
| Random | 0.0001 | 0.0000 | 99.1702% | 18.3655 |
| Content-Based | 0.0083 | 0.0035 | 9.9376% | 14.4408 |
| Biased MF (SQ) | 0.0264 | 0.0090 | 0.1091% | 11.1796 |
| **BPR MF** | **0.0289** | **0.0105** | **0.2473%** | **10.5064** |

## Key Findings So Far

1. **Popularity is a very strong baseline.**  
   On Food.com, recommending globally popular recipes is surprisingly hard to beat on accuracy.

2. **BPR MF is the strongest personalized collaborative model so far.**  
   It performs close to popularity while remaining a true personalized recommender.

3. **Content-Based dramatically improves coverage.**  
   It is much weaker than BPR on accuracy, but vastly better in catalog coverage and novelty.

4. **The current kNN implementation does not scale to the full Food.com item catalog.**  
   For this reason, it is disabled in the full baseline notebook run.

5. **Strict temporal splitting matters.**  
   We explicitly remove users with tied final timestamps so that the last interaction is temporally identifiable.

## Next Steps
- improve content feature preprocessing for Food.com list-like metadata
- test a hybrid of BPR MF and Content-Based recommendations
- tune BPR hyperparameters
- compare stronger diversity / coverage trade-offs
