import numpy as np
import random
import pandas as pd


class PopularityRecommender:
    """
    Popularity-based baseline for recipe recommendation.

    Ranks recipes by their interaction count in the training set.
    This is a strong non-personalized baseline and helps detect whether
    the dataset is dominated by head/popular recipes.
    """

    def __init__(self):
        self.popular_items = []

    def fit(self, train_df, item_col='recipe_id'):
        print("Training Popularity Baseline...")

        # Count how often each recipe appears in the training data
        item_counts = train_df[item_col].value_counts()

        # Store recipes sorted from most to least popular
        self.popular_items = item_counts.index.tolist()

        print(f"Learned popularity ranking for {len(self.popular_items)} unique recipes.")

    def recommend(self, user_id, user_history, k=10):
        """
        Returns top-K popular recipes while filtering out recipes
        the user has already interacted with in the training set.
        """
        user_history = set(user_history)
        recommendations = []

        for item in self.popular_items:
            if item not in user_history:
                recommendations.append(item)

            if len(recommendations) == k:
                break

        return recommendations


class RandomRecommender:
    """
    Random baseline for recipe recommendation.

    Samples recipes uniformly at random from the training candidate universe.
    This is mainly used as a sanity check for the evaluation pipeline.
    """

    def __init__(self, random_state=None):
        self.candidate_universe = []
        self.random_state = random_state

    def fit(self, train_df, item_col='recipe_id'):
        print("Training Random Baseline...")

        # Candidate universe: all recipes observed in the training data
        self.candidate_universe = train_df[item_col].unique().tolist()

        if self.random_state is not None:
            random.seed(self.random_state)

        print(f"Random candidate universe contains {len(self.candidate_universe)} recipes.")

    def recommend(self, user_id, user_history, k=10):
        user_history = set(user_history)

        # Remove recipes already seen by the user
        valid_candidates = list(set(self.candidate_universe) - user_history)

        if len(valid_candidates) >= k:
            return random.sample(valid_candidates, k)

        return valid_candidates


class TrendingRecommender:
    """
    Recent-popularity baseline for recipe recommendation.

    Ranks recipes by interaction count within a recent time window.
    If the recent window does not provide enough candidates, it falls back
    to all-time popularity.
    """

    def __init__(self, days_window=30):
        self.days_window = days_window
        self.trending_items = []
        self.popular_fallback = []

    def fit(self, train_df, item_col='recipe_id', time_col='date_time'):
        print(f"Training Trending Baseline (last {self.days_window} days)...")

        if time_col not in train_df.columns:
            raise ValueError(f"Missing time column: {time_col}")

        train_df = train_df.copy()
        train_df[time_col] = pd.to_datetime(train_df[time_col], errors='coerce')

        # Use the latest timestamp in training as the reference point
        max_date = train_df[time_col].max()
        cutoff_date = max_date - pd.Timedelta(days=self.days_window)

        # Count recipe interactions only within the recent window
        recent_df = train_df[train_df[time_col] >= cutoff_date]

        self.trending_items = recent_df[item_col].value_counts().index.tolist()

        # Fallback: all-time popularity from training only
        self.popular_fallback = train_df[item_col].value_counts().index.tolist()

        print(f"Learned {len(self.trending_items)} trending recipes.")

    def recommend(self, user_id, user_history, k=10):
        user_history = set(user_history)
        recommendations = []

        # First recommend recently popular recipes
        for item in self.trending_items:
            if item not in user_history:
                recommendations.append(item)

            if len(recommendations) == k:
                break

        # Fill remaining slots with all-time popular recipes
        if len(recommendations) < k:
            recs_set = set(recommendations)

            for item in self.popular_fallback:
                if item not in user_history and item not in recs_set:
                    recommendations.append(item)

                if len(recommendations) == k:
                    break

        return recommendations