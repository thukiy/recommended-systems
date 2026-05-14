# Food.com Recipe Recommender

**Course:** CDS121 - Recommender Systems
**Institution:** Fachhochschule Graubünden (FHGR)
**Team:** Isabelle, Philip, Thuvaraka

---

# Project Overview

This project builds and evaluates a modern recommender system pipeline for Food.com recipes.

The goal is to recommend relevant recipes from historical user interactions while comparing different recommendation paradigms under a strict offline evaluation setup.

The project evolved from simple baseline recommenders into a complete retrieval-and-ranking framework inspired by modern recommender-system architectures.

Our pipeline includes:

* temporal leave-last-out evaluation
* collaborative filtering
* content-based recommendation
* matrix factorization
* BPR optimization for implicit feedback
* ANN-based retrieval
* sequential recommendation
* graph-based recommendation
* hybrid retrieval
* learning-to-rank reranking
* retrieval evaluation
* explainability and diagnostics

---

# Dataset

We use the **Food.com Recipes and User Interactions** dataset from Kaggle:

https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions

For the project, we use:

* `RAW_interactions.csv`
* `RAW_recipes.csv`

## Why This Dataset

* contains long-term user histories from 2000–2018
* includes both interaction data and rich recipe metadata
* supports collaborative, content-based, sequential, and graph-based recommendation experiments
* contains strong head-tail and popularity effects

## Modeling Assumption

Although the dataset contains explicit ratings, we transform the task into an implicit Top-N recommendation problem:

* positive feedback: `rating >= 4`
* users with fewer than 2 positive interactions are excluded from evaluation

## Important

The dataset is too large for version control and is excluded via `.gitignore`.

To run the project locally:

1. Download the dataset from Kaggle
2. Extract the archive
3. Place the files into:

```text
data/raw/
```

Required files:

* `RAW_interactions.csv`
* `RAW_recipes.csv`

---

# Repository Structure

```text
recommended-systems/
│
├── data/
│   └── raw/
│       ├── RAW_interactions.csv
│       └── RAW_recipes.csv
│
├── evaluation/
│   ├── metrics.py
│   ├── retrieval_metrics.py
│   └── split.py
│
├── models/
│   ├── ann_retriever.py
│   ├── baselines.py
│   ├── content_based.py
│   ├── graph_recommender.py
│   ├── knn.py
│   ├── ltr_ranker.py
│   ├── mf.py
│   └── sequential.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_baseline_evaluation.ipynb
│
├── runs/
│   └── runs.csv
│
├── environment.yml
├── requirements.txt
└── README.md
```

---

# Implemented Recommendation Models

## Baselines

* Popularity Recommender
* Trending Recommender
* Random Recommender

## Collaborative Filtering

* Item-Item kNN
* Biased Matrix Factorization
* BPR Matrix Factorization

## Content-Based Recommendation

* TF-IDF recipe representation
* metadata-aware retrieval
* hybrid content + collaborative retrieval

## Sequential Recommendation

* First-Order Markov Recommender

## Graph-Based Recommendation

* Personalized PageRank (PPR) Recommender

## ANN Retrieval

* FAISS / ANN retrieval over learned BPR embeddings
* scalable nearest-neighbor candidate generation

## Learning-to-Rank

* Logistic Regression reranker
* feature-based reranking over retrieved candidates

---

# Evaluation Strategy

The project uses a strict temporal evaluation setup:

1. interactions are sorted chronologically
2. the final interaction of each user becomes the test item
3. earlier interactions form the training history
4. users with ambiguous final timestamps are removed

This prevents temporal leakage and simulates realistic recommendation scenarios.

---

# Evaluation Metrics

## Ranking Metrics

* Recall@K
* NDCG@K
* Catalog Coverage@K
* Novelty@K

## Retrieval Metrics

* Candidate Recall@K
* Candidate Coverage@K
* Average Candidate Set Size
* Duplicate Rate
* Retrieval Latency

---

# Setup

## Conda / Mamba

```bash
conda env create -f environment.yml
conda activate recommended-systems
```

or

```bash
mamba env create -f environment.yml
conda activate recommended-systems
```

## Pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

# How To Run

## 1. Data Exploration

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

This notebook explores:

* rating distribution
* sparsity
* cold users / cold items
* popularity distribution
* temporal effects
* head-tail structure
* metadata quality

---

## 2. Main Evaluation Pipeline

```bash
jupyter notebook notebooks/02_baseline_evaluation.ipynb
```

The notebook contains:

1. Setup & Imports
2. Data Pipeline
3. Baseline, Sequential, Graph & Retrieval Model Training
4. Ranking Evaluation
5. Retrieval Evaluation & ANN Analysis
6. Learning-to-Rank
7. Slice Analysis & Hybrid Analysis
8. Diagnostics & Explainability
9. Experiment Logging

---

# Current Results

Strict Food.com offline evaluation (`K=20`):

| Model                | Recall@20 | NDCG@20 | Coverage@20 | Novelty@20 |
| -------------------- | --------- | ------- | ----------- | ---------- |
| Popularity           | 0.0312    | 0.0125  | 0.0367%     | 10.3203    |
| Trending (180d)      | 0.0029    | 0.0018  | 0.0173%     | 15.6854    |
| Random               | 0.0001    | 0.0000  | 99.1464%    | 18.3655    |
| Content-Based        | 0.0083    | 0.0035  | 9.9376%     | 14.4408    |
| Biased MF (SQ)       | 0.0264    | 0.0090  | 0.1091%     | 11.1796    |
| BPR MF               | 0.0311    | 0.0117  | 0.0367%     | 10.4011    |
| Hybrid BPR + Content | 0.0225    | 0.0083  | 9.4522%     | 12.4656    |

---

# Key Findings

1. **Popularity is a very strong baseline.**
   On Food.com, globally popular recipes are surprisingly difficult to outperform.

2. **BPR MF is currently the strongest personalized collaborative model.**
   It achieves the best balance between ranking quality and personalization.

3. **Content-Based retrieval improves coverage and novelty.**
   Accuracy is lower, but the recommender explores a much larger portion of the catalog.

4. **Retrieval quality limits ranking quality.**
   If relevant items do not survive candidate generation, the reranker cannot recover them later.

5. **ANN retrieval separates retrieval from ranking.**
   The ANN retriever keeps the learned BPR embedding space while enabling scalable nearest-neighbor search.

6. **Cold-start remains difficult.**
   Pure collaborative models fail completely on unseen items.

7. **Sequential and graph-based models extend the project beyond classic baselines.**
   They allow experiments with temporal transitions and multi-hop user-item graph signals.

---

# Future Work

Possible next steps:

* stronger hybrid retrieval
* hard negative sampling for BPR
* improved cold-start handling
* deep sequential recommenders
* graph neural networks
* neural two-tower retrieval models
* better ANN indexing strategies
* diversification-aware reranking

---

# License

This project was developed for academic purposes as part of the FHGR Recommender Systems course.
