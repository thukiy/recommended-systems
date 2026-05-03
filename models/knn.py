import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


class ItemItemKNN:
    """
    Week 3: Item-Item Collaborative Filtering using Implicit Feedback.
    Includes Shrinkage for noise reduction and Constructive Explainability.
    """

    def __init__(self, k_neighbors=100, shrinkage=10):
        self.k_neighbors = k_neighbors
        self.shrinkage = shrinkage  # Lambda parameter for trust weighting
        self.item_mapping = {}
        self.reverse_item_mapping = {}
        self.item_sim_matrix = None
        self.popular_fallback = []

    def fit(self, train_df, user_col='user_id', item_col='recipe_id'):
        print(f"Training Item-Item kNN (k={self.k_neighbors}, shrinkage={self.shrinkage})...")

        unique_users = train_df[user_col].unique()
        unique_items = train_df[item_col].unique()

        user_mapping = {u: i for i, u in enumerate(unique_users)}
        self.item_mapping = {item: i for i, item in enumerate(unique_items)}
        self.reverse_item_mapping = {i: item for item, i in self.item_mapping.items()}

        self.popular_fallback = train_df[item_col].value_counts().index.tolist()

        row_indices = train_df[user_col].map(user_mapping).values
        col_indices = train_df[item_col].map(self.item_mapping).values

        data = np.ones(len(train_df))
        R_ui = csr_matrix((data, (row_indices, col_indices)),
                          shape=(len(unique_users), len(unique_items)))

        print("Normalizing vectors and computing Cosine Similarity...")
        R_ui_normalized = normalize(R_ui, norm='l2', axis=0)
        S_ii = R_ui_normalized.T.dot(R_ui_normalized)

        S_ii.setdiag(0)
        S_ii.eliminate_zeros()

        # --- NEW: SHRINKAGE (Week 3, Slide 43) ---
        if self.shrinkage > 0:
            print("Applying shrinkage to downweight coincidental overlaps...")
            # Calculate raw co-occurrence counts
            C_ii = R_ui.T.dot(R_ui)
            C_ii.setdiag(0)
            C_ii.eliminate_zeros()

            # Since R_ui is binary, S_ii and C_ii have the exact same sparsity structure.
            # We can apply the penalty directly to the underlying data arrays for extreme speed.
            S_ii.data = S_ii.data * (C_ii.data / (C_ii.data + self.shrinkage))

        print("Pruning to Top-K neighborhoods to reduce noise...")
        for i in range(S_ii.shape[0]):
            row = S_ii.getrow(i).toarray()[0]
            if len(row.nonzero()[0]) > self.k_neighbors:
                top_k_indices = np.argsort(row)[-self.k_neighbors:]
                mask = np.ones(row.shape, dtype=bool)
                mask[top_k_indices] = False
                row[mask] = 0
                S_ii.data[S_ii.indptr[i]:S_ii.indptr[i + 1]] = row[S_ii.indices[S_ii.indptr[i]:S_ii.indptr[i + 1]]]

        S_ii.eliminate_zeros()
        self.item_sim_matrix = S_ii
        print("kNN Training complete!")

    def recommend(self, user_id, user_history, k=10, return_explanations=False):
        history_indices = [self.item_mapping[item] for item in user_history if item in self.item_mapping]

        if not history_indices:
            if return_explanations:
                return [(item, "Recommended due to general popularity (Cold Start).") for item in
                        self.popular_fallback[:k]]
            return self.popular_fallback[:k]

        scores = np.zeros(len(self.item_mapping))
        for item_idx in history_indices:
            scores += self.item_sim_matrix.getrow(item_idx).toarray()[0]

        scores[history_indices] = -1

        top_k_indices = np.argsort(scores)[-k:][::-1]

        # Fallback logic for extreme sparsity
        if scores[top_k_indices[0]] == 0:
            recommendations = [self.reverse_item_mapping[idx] for idx in top_k_indices if scores[idx] > 0]
            recs_set = set(recommendations)
            for item in self.popular_fallback:
                if item not in user_history and item not in recs_set:
                    recommendations.append(item)
                if len(recommendations) == k:
                    break
        else:
            recommendations = [self.reverse_item_mapping[idx] for idx in top_k_indices]

        # --- NEW: EXPLAINABILITY (Week 3, Slide 45) ---
        if return_explanations:
            explained_recs = []
            for rec_idx in top_k_indices:
                if scores[rec_idx] <= 0:
                    exp = "Recommended due to general popularity."
                else:
                    # Find which history item contributed the maximum similarity score
                    best_hist_idx = max(history_indices, key=lambda h: self.item_sim_matrix[h, rec_idx])
                    hist_name = self.reverse_item_mapping[best_hist_idx]
                    exp = f"Recommended because it is similar to a previously liked recipe (recipe_id={hist_name})."

                explained_recs.append((self.reverse_item_mapping[rec_idx], exp))
            return explained_recs[:k]

        return recommendations[:k]