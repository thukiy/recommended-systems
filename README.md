# TravelBuddy: Context-Aware POI Recommender

**Course:** CDS121 - Recommender Systems  
**Institution:** Fachhochschule Graubünden (FHGR)  
**Team:** Isabelle, Philip, Thuvaraka  

## Project Overview
TravelBuddy is a recommender system designed to suggest exploratory activities and Points of Interest (POIs) to travelers. Instead of relying on explicit ratings, the system learns from sparse, implicit feedback (user check-ins) to generate highly personalized, localized recommendations. 

This repository contains the complete end-to-end pipeline, including data preprocessing, strict temporal evaluation, and the implementation of various collaborative filtering algorithms.

## Dataset
We use the **Foursquare NYC Check-in Dataset (TSMC2014)**. 
*   **Why this dataset?** It provides persistent user histories with exact timestamps, which is strictly required for preventing time-travel leakage during offline evaluation.
*   **Semantic Filtering:** We aggressively filtered out daily routine categories (e.g., *Home, Office, Subway, Train Station, Gym*) to align the candidate universe with our tourism business objective.

**Important:** The dataset is too large for version control and is ignored via `.gitignore`. 
To run this code locally, you must download the dataset manually:
1. Download the dataset from[Kaggle: Foursquare NYC and Tokyo Check-ins](https://www.kaggle.com/datasets/chetanism/foursquare-nyc-and-tokyo-checkin-dataset).
2. Extract the archive.
3. Place the file `dataset_TSMC2014_NYC.csv` into the `data/raw/` directory of this project.

## Repository Structure

```text
travelbuddy/
│
├── data/                           # Ignored by Git
│   └── raw/                        
│       └── dataset_TSMC2014_NYC.csv # Put the downloaded Kaggle data here!
│
├── evaluation/                     # The Evaluation Harness (Week 2)
│   ├── __init__.py
│   ├── metrics.py                  # Recall@K, NDCG@K, Coverage@K, Novelty@K
│   └── split.py                    # Strict Leave-Last-Out temporal split
│
├── models/                         # Recommender Algorithms
│   ├── __init__.py
│   ├── baselines.py                # Popularity, Random, Trending
│   ├── knn.py                      # Item-Item kNN (with Shrinkage & Explainability)
│   └── mf.py                       # Biased MF (Squared Loss) & BPR MF
│
├── notebooks/                      # Experimentation & Pipeline Execution
│   ├── 01_data_exploration.ipynb   # Sanity checks, sparsity analysis, long-tail plots
│   └── 02_baseline_evaluation.ipynb # Master evaluation loop & diagnostics
│
├── runs/                           # Experiment Tracking
│   └── runs.csv                    # Automatically logs model configs and metrics
│
├── requirements.txt                # Python dependencies
├── .gitignore                      
└── README.md                       # Project documentation
```

## Setup & Installation

To run this project locally, ensure you have Python 3.9+ installed.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/thukiy/recommended-systems.git
    cd travelbuddy
    ```
2.  **Create a virtual environment (Optional but recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Required packages: `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `jupyter`)*
4.  **Add the dataset:**
    Ensure `dataset_TSMC2014_NYC.csv` is placed inside `data/raw/`.
5.  **Run the Evaluation Pipeline:**
    Open `notebooks/02_baseline_evaluation.ipynb` and execute all cells to train the models and generate the final results table.

## Model Implementations & Results

This repository currently implements the core collaborative filtering algorithms covered in Weeks 1-4 of the CDS121 curriculum. All models are evaluated using a strict **Leave-Last-Out** temporal split to simulate real-world deployment.

### Current Baseline Metrics (K=20)
| Model | Recall@20 | NDCG@20 | Coverage@20 | Novelty@20 |
| :--- | :--- | :--- | :--- | :--- |
| Popularity | 0.0065 | 0.0021 | 0.10% | 9.46 |
| Trending (30d) | 0.0037 | 0.0012 | 0.07% | 10.92 |
| Random | 0.0009 | 0.0002 | 50.25% | 16.01 |
| Item-Item kNN | 0.0028 | 0.0010 | 16.20% | 13.37 |
| Biased MF (SQ) | 0.0018 | 0.0005 | 0.40% | 10.01 |
| **BPR MF** | **0.0166** | **0.0048** | **2.03%** | **10.20** |

### Key Engineering Diagnostics
1.  **The "Implicit Zero" Problem:** Standard Matrix Factorization using squared-loss (Biased MF) collapsed catastrophically on our implicit check-in data, treating all unobserved items as absolute negatives. This resulted in an unacceptable Catalog Coverage of 0.40%.
2.  **BPR Matrix Factorization:** By shifting the objective to Bayesian Personalized Ranking (pairwise ranking via negative sampling), we successfully solved the implicit zero problem. BPR MF currently stands as our champion model, beating the Popularity baseline in Accuracy (`0.0166`) while simultaneously increasing Catalog Coverage by 20x (`2.03%`).
3.  **Latent Semantics:** Diagnostic checks confirm that the BPR latent vectors successfully map human behavior into semantic space without using metadata (e.g., retrieving 'Coffee Shop' and 'Bakery' as the nearest latent neighbors for 'Park').
