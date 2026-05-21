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
│   └── 03_fullrun_baseline_eval.ipynb
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

* Pairwise XGBoost Ranker (LambdaMART objective)
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
* SNIPS@K

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
jupyter notebook notebooks/03_fullrun_baseline_eval.ipynb
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

Strict Food.com offline evaluation (`K=20`) evaluated on all users with parallelized joblib processing:

| Model                | Recall@20 | NDCG@20 | Coverage@20 | Novelty@20 | SNIPS@20 |
|----------------------|-----------|---------| ----------- | ---------- | -------- |
| Popularity           | 0.0343    | 0.0139  | 0.02%       | 10.3201    | 0.0343   |
| Trending (180d)      | 0.0038    | 0.0023  | 0.01%       | 15.6851    | 0.0038   |
| Markov Sequential    | 0.0196    | 0.0075  | 20.41%      | 12.9805    | 0.0196   |
| Graph PPR            | 0.0064    | 0.0029  | 77.14%      | 15.6024    | 0.0029   |
| Random               | 0.0001    | 0.0000  | 72.66%      | 18.3646    | 0.0001   |
| Content-Based        | 0.0105    | 0.0042  | 8.49%       | 14.4969    | 0.0105   |
| Item-Item kNN        | 0.0199    | 0.0087  | 15.02%      | 13.2176    | 0.0045   |
| Biased MF (SQ)       | 0.0198    | 0.0068  | 0.19%       | 11.6514    | 0.0198   |
| BPR MF               | 0.0343    | 0.0132  | 0.02%       | 10.4009    | 0.0343   |
| BPR MF Debiased      | 0.0318    | 0.0110  | 0.02%       | 10.7210    | 0.0318   |
| ANN BPR Retrieval    | 0.0032    | 0.0009  | 4.60%       | 13.7976    | 0.0032   |
| Hybrid BPR + Content | 0.0257    | 0.0092  | 7.64%       | 12.5080    | 0.0257   |
| XGBoost LTR Ranker   | 0.0237    | 0.0087  | -           | -          | -        |

---

# Key Findings

1. **Popularity is a very strong baseline.**
   On Food.com, globally popular recipes are surprisingly difficult to outperform.

2. **Learning-to-Rank (LTR) successfully re-ranks candidates.**
   By injecting explicit positive items and using the retriever's unclicked items as "Hard Negatives," the Pairwise XGBoost ranker successfully optimized the NDCG and Recall over the base hybrid candidates.

3. **Content-Based retrieval improves coverage and novelty.**
   Accuracy is lower, but the recommender explores a much larger portion of the catalog.

4. **Debiasing reveals the true value of Hybrid models.**
   While traditional Recall favors the Popularity and BPR models (due to extreme popularity bias in the dataset), the newly introduced SNIPS metric proves that the Hybrid model is highly effective at surfacing long-tail items.

5. **ANN retrieval separates retrieval from ranking.**
   The ANN retriever keeps the learned BPR embedding space while enabling scalable nearest-neighbor search.

6. **Content-Based models solve the Cold-Start problem.**
   By removing the candidate limits on the TF-IDF vectorizer, the Content-Based model successfully recommended items that had exactly zero interactions in the training set (Day-Zero items), where all pure collaborative models completely failed.

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
