import numpy as np
import random
import pandas as pd

class PopularityRecommender:
    """
    Popularity-based baseline.
    Ranks items by their interaction count in the training set.
    """

    def __init__(self):
        self.popular_items = []

    def fit(self, train_df, item_col='venue_id'):
        print("Training Popularity Baseline...")
        # Count how many times each item appears in the training data
        item_counts = train_df[item_col].value_counts()
        # Store the list of items sorted from most to least popular
        self.popular_items = item_counts.index.tolist()
        print(f"Learned popularity ranking for {len(self.popular_items)} unique items.")

    def recommend(self, user_id, user_history, k=10):
        """
        Returns the top-K recommendations while filtering out already seen items.
        """
        recommendations = []
        for item in self.popular_items:
            # Exclude items the user has already interacted with
            if item not in user_history:
                recommendations.append(item)

            # Stop once we have collected K items
            if len(recommendations) == k:
                break

        return recommendations


class RandomRecommender:
    """
    Random baseline.
    Samples items uniformly at random. Useful as a simple reference point for evaluation.
    """

    def __init__(self):
        self.candidate_universe = []

    def fit(self, train_df, item_col='venue_id'):
        print("Training Random Baseline...")
        # Build the candidate pool from all unique items observed during training
        self.candidate_universe = train_df[item_col].unique().tolist()

    def recommend(self, user_id, user_history, k=10):
        # Remove items the user has already seen from the candidate pool
        valid_candidates = list(set(self.candidate_universe) - set(user_history))

        # If enough candidates remain, sample K items uniformly at random
        if len(valid_candidates) >= k:
            return random.sample(valid_candidates, k)
        else:
            return valid_candidates



class TrendingRecommender:
    """
    Baseline 3: Trending.
    Ranks items by popularity, but ONLY counts interactions from a recent time window.
    """

    def __init__(self, days_window=30):
        self.days_window = days_window
        self.trending_items = []
        self.popular_fallback = []

    def fit(self, train_df, item_col='venue_id', time_col='date_time'):
        print(f"Training Trending Baseline (Last {self.days_window} days)...")

        # 1. Find the "current" date in the training set
        max_date = train_df[time_col].max()
        cutoff_date = max_date - pd.Timedelta(days=self.days_window)

        # 2. Filter data to only the recent window
        recent_df = train_df[train_df[time_col] >= cutoff_date]

        # 3. Calculate trending items
        item_counts = recent_df[item_col].value_counts()
        self.trending_items = item_counts.index.tolist()

        # 4. Calculate all-time popularity as a fallback (if trending isn't enough)
        self.popular_fallback = train_df[item_col].value_counts().index.tolist()
        print(f"Learned {len(self.trending_items)} trending items.")

    def recommend(self, user_id, user_history, k=10):
        recommendations = []

        # First, try to recommend trending items
        for item in self.trending_items:
            if item not in user_history:
                recommendations.append(item)
            if len(recommendations) == k:
                break

        # Fallback: If we don't have K trending items, fill the rest with all-time popularity
        if len(recommendations) < k:
            recs_set = set(recommendations)
            for item in self.popular_fallback:
                if item not in user_history and item not in recs_set:
                    recommendations.append(item)
                if len(recommendations) == k:
                    break

        return recommendations