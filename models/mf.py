import numpy as np
import pandas as pd
import random


class BiasedMatrixFactorization:
    """
    Matrix factorization baseline with bias terms, trained via SGD and negative sampling.
    Learns shared latent structure using user and item factors alongside global and per-entity biases.
    """

    def __init__(self, k_factors=16, learning_rate=0.01, reg=0.02, epochs=5):
        self.k = k_factors
        self.lr = learning_rate
        self.reg = reg
        self.epochs = epochs

        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_item_mapping = {}

        # Latent factors and bias terms
        self.P = None  # User factors
        self.Q = None  # Item factors
        self.b_u = None  # User bias
        self.b_i = None  # Item bias
        self.global_mean = 0.0

        self.popular_fallback = []
        self.all_items_set = set()

    def fit(self, train_df, user_col='user_id', item_col='venue_id'):
        print(f"Training Biased MF (k={self.k}, epochs={self.epochs})...")

        unique_users = train_df[user_col].unique()
        unique_items = train_df[item_col].unique()
        self.all_items_set = set(unique_items)

        self.user_mapping = {u: i for i, u in enumerate(unique_users)}
        self.item_mapping = {item: i for i, item in enumerate(unique_items)}
        self.reverse_item_mapping = {i: item for item, i in self.item_mapping.items()}

        num_users = len(unique_users)
        num_items = len(unique_items)

        # Initialize latent factors with small random values
        self.P = np.random.normal(scale=1. / self.k, size=(num_users, self.k))
        self.Q = np.random.normal(scale=1. / self.k, size=(num_items, self.k))
        self.b_u = np.zeros(num_users)
        self.b_i = np.zeros(num_items)

        # Keep a popularity-based fallback for cold-start users
        self.popular_fallback = train_df[item_col].value_counts().index.tolist()

        # Prepare interaction data for efficient training
        user_indices = train_df[user_col].map(self.user_mapping).values
        item_indices = train_df[item_col].map(self.item_mapping).values
        interactions = list(zip(user_indices, item_indices))

        # Precompute user histories for negative sampling
        user_history = train_df.groupby(user_col)[item_col].apply(lambda x: set(x.map(self.item_mapping))).to_dict()
        all_item_indices = list(range(num_items))

        # SGD training loop with one sampled negative per positive interaction
        for epoch in range(self.epochs):
            np.random.shuffle(interactions)

            for u, i in interactions:
                # Positive update: observed interaction should receive a high score
                pred_pos = self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i])
                err_pos = 1.0 - pred_pos

                # Update biases and latent factors
                self.b_u[u] += self.lr * (err_pos - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (err_pos - self.reg * self.b_i[i])

                P_u_old = self.P[u].copy()
                self.P[u] += self.lr * (err_pos * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err_pos * P_u_old - self.reg * self.Q[i])

                # Negative update: sample an unobserved item for the same user
                j = random.choice(all_item_indices)
                while j in user_history[unique_users[u]]:
                    j = random.choice(all_item_indices)

                pred_neg = self.b_u[u] + self.b_i[j] + np.dot(self.P[u], self.Q[j])
                err_neg = 0.0 - pred_neg

                self.b_u[u] += self.lr * (err_neg - self.reg * self.b_u[u])
                self.b_i[j] += self.lr * (err_neg - self.reg * self.b_i[j])

                P_u_old_neg = self.P[u].copy()
                self.P[u] += self.lr * (err_neg * self.Q[j] - self.reg * self.P[u])
                self.Q[j] += self.lr * (err_neg * P_u_old_neg - self.reg * self.Q[j])

            print(f"Epoch {epoch + 1}/{self.epochs} completed.")

        print("MF training complete!")

    def recommend(self, user_id, user_history, k=10):
        # Return popularity-based recommendations for unseen users
        if user_id not in self.user_mapping:
            return self.popular_fallback[:k]

        u_idx = self.user_mapping[user_id]

        # Score all items using user/item factors and bias terms
        scores = self.b_u[u_idx] + self.b_i + np.dot(self.Q, self.P[u_idx])

        # Filter out items already seen by the user
        for item in user_history:
            if item in self.item_mapping:
                item_idx = self.item_mapping[item]
                scores[item_idx] = -np.inf

        # Select the highest-scoring items
        top_k_indices = np.argsort(scores)[-k:][::-1]
        recommendations = [self.reverse_item_mapping[idx] for idx in top_k_indices]

        return recommendations


class BPRMatrixFactorization:
    """
    Week 4: Matrix Factorization using Bayesian Personalized Ranking (BPR).
    Optimizes for pairwise ranking (item i > item j) instead of point-wise scores.
    """

    def __init__(self, k_factors=16, learning_rate=0.05, reg=0.01, epochs=10):
        self.k = k_factors
        self.lr = learning_rate
        self.reg = reg
        self.epochs = epochs

        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_item_mapping = {}

        self.P = None  # User factors
        self.Q = None  # Item factors
        self.b_i = None  # Item bias (User bias cancels out in BPR)

        self.popular_fallback = []

    def fit(self, train_df, user_col='user_id', item_col='venue_id'):
        print(f"Training BPR MF (k={self.k}, epochs={self.epochs})...")

        unique_users = train_df[user_col].unique()
        unique_items = train_df[item_col].unique()

        self.user_mapping = {u: idx for idx, u in enumerate(unique_users)}
        self.item_mapping = {item: idx for idx, item in enumerate(unique_items)}
        self.reverse_item_mapping = {idx: item for item, idx in self.item_mapping.items()}

        num_users = len(unique_users)
        num_items = len(unique_items)

        # Initialize latent vectors with small random values
        self.P = np.random.normal(scale=1. / self.k, size=(num_users, self.k))
        self.Q = np.random.normal(scale=1. / self.k, size=(num_items, self.k))
        self.b_i = np.zeros(num_items)

        self.popular_fallback = train_df[item_col].value_counts().index.tolist()

        user_indices = train_df[user_col].map(self.user_mapping).values
        item_indices = train_df[item_col].map(self.item_mapping).values
        interactions = list(zip(user_indices, item_indices))

        user_history = train_df.groupby(user_col)[item_col].apply(lambda x: set(x.map(self.item_mapping))).to_dict()
        all_item_indices = list(range(num_items))

        # SGD for BPR
        for epoch in range(self.epochs):
            np.random.shuffle(interactions)

            for u, i in interactions:
                # Sample a negative item j that the user hasn't interacted with
                j = random.choice(all_item_indices)
                while j in user_history[unique_users[u]]:
                    j = random.choice(all_item_indices)

                # Calculate the pairwise difference: x_uij = score(u,i) - score(u,j)
                score_i = self.b_i[i] + np.dot(self.P[u], self.Q[i])
                score_j = self.b_i[j] + np.dot(self.P[u], self.Q[j])
                x_uij = score_i - score_j

                # Sigmoid function for gradient multiplier
                # Using np.clip to prevent overflow warnings in exp
                sigmoid = 1.0 / (1.0 + np.exp(np.clip(-x_uij, -20, 20)))
                grad_multiplier = 1.0 - sigmoid

                # Update parameters
                self.b_i[i] += self.lr * (grad_multiplier - self.reg * self.b_i[i])
                self.b_i[j] += self.lr * (-grad_multiplier - self.reg * self.b_i[j])

                P_u_copy = self.P[u].copy()
                self.P[u] += self.lr * (grad_multiplier * (self.Q[i] - self.Q[j]) - self.reg * self.P[u])
                self.Q[i] += self.lr * (grad_multiplier * P_u_copy - self.reg * self.Q[i])
                self.Q[j] += self.lr * (-grad_multiplier * P_u_copy - self.reg * self.Q[j])

            print(f"Epoch {epoch + 1}/{self.epochs} completed.")
        print("BPR Training complete!")

    def recommend(self, user_id, user_history, k=10):
        if user_id not in self.user_mapping:
            return self.popular_fallback[:k]

        u_idx = self.user_mapping[user_id]

        # Calculate scores for all items
        scores = self.b_i + np.dot(self.Q, self.P[u_idx])

        # Seen-item filtering
        for item in user_history:
            if item in self.item_mapping:
                scores[self.item_mapping[item]] = -np.inf

        top_k_indices = np.argsort(scores)[-k:][::-1]
        return [self.reverse_item_mapping[idx] for idx in top_k_indices]