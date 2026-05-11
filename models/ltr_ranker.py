import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression


class PointwiseLTRRanker:
    """
    Simple pointwise Learning-to-Rank model.

    The model learns:
        P(click | user, item, interaction features)

    using logistic regression.

    This ranker is designed for the second stage
    of a two-stage recommender system.

    Retrieval generates candidates.
    The LTR model reranks those candidates.
    """

    def __init__(self):
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )

        self.feature_columns = None

    def fit(self, train_df, feature_columns, label_column="label"):
        """
        Train the ranker on candidate rows.
        """

        self.feature_columns = feature_columns

        X = train_df[feature_columns].values
        y = train_df[label_column].values

        self.model.fit(X, y)

    def predict_scores(self, feature_df):
        """
        Predict ranking scores for candidate rows.
        """

        X = feature_df[self.feature_columns].values

        # probability of positive class
        scores = self.model.predict_proba(X)[:, 1]

        return scores

    def rerank(self, feature_df, top_k=20):
        """
        Rerank candidates by predicted click probability.
        """

        feature_df = feature_df.copy()

        feature_df["ltr_score"] = self.predict_scores(feature_df)

        reranked = (
            feature_df
            .sort_values("ltr_score", ascending=False)
            .head(top_k)
        )

        return reranked["recipe_id"].tolist()