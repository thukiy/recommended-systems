import re
import numpy as np
import pandas as pd


class ContentBasedRecommender:
    """
    Content-based recommender using TF-IDF item representations.
    Learns item vectors from metadata and builds user profiles as the mean of
    vectors from previously interacted items.
    """

    def __init__(
        self,
        feature_cols=None,
        text_cols=None,
        metadata_cols=None,
        min_df=1,
        ngram_range=(1, 2),
        recency_decay=0.0,
        event_col=None,
        time_col=None,
        event_weight_map=None,
    ):
        # Backward-compatible: feature_cols can still be passed as before.
        self.feature_cols = feature_cols
        self.text_cols = text_cols
        self.metadata_cols = metadata_cols
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.recency_decay = recency_decay
        self.event_col = event_col
        self.time_col = time_col
        self.event_weight_map = event_weight_map or {}

        self.vectorizer = None
        self.item_feature_matrix = None
        self.feature_names = np.array([])

        self.item_mapping = {}
        self.reverse_item_mapping = {}
        self.user_profiles = {}
        self.user_history_counts = {}

        self.popular_fallback = []
        self._feature_missing_rates = {}

    @staticmethod
    def _normalize_text(value):
        text = str(value).lower().strip()
        text = re.sub(r"[^a-z0-9\s]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _safe_strftime_days(delta):
        return float(delta.total_seconds()) / 86400.0

    def _item_to_document(self, row, item_col, text_cols, metadata_cols):
        tokens = []

        # Text features: plain normalized tokens (prefixed by column name).
        for col in text_cols:
            val = row[col]
            if pd.isna(val):
                continue

            normalized = self._normalize_text(val)
            if not normalized:
                continue

            for token in normalized.split():
                tokens.append(f"{col}_{token}")

        # Metadata features: explicit typed tokens.
        for col in metadata_cols:
            val = row[col]
            if pd.isna(val):
                continue

            if isinstance(val, (bool, np.bool_)):
                tokens.append(f"{col}_true" if val else f"{col}_false")
                continue

            if isinstance(val, (int, float, np.integer, np.floating)):
                # Quantize numeric values lightly to reduce token explosion.
                quantized = round(float(val), 1)
                tokens.append(f"{col}_{quantized}")
                continue

            normalized = self._normalize_text(val)
            if not normalized:
                continue

            for token in normalized.split():
                tokens.append(f"{col}_{token}")

        # Fallback token keeps item in vocabulary even with empty metadata.
        if not tokens:
            tokens.append(f"{item_col}_{row[item_col]}")

        return " ".join(tokens)

    def _resolve_feature_columns(self, items_unique, item_col):
        if self.feature_cols is not None:
            cols = [c for c in self.feature_cols if c in items_unique.columns and c != item_col]
            return cols, []

        resolved_text_cols = []
        resolved_metadata_cols = []

        if self.text_cols is not None:
            resolved_text_cols = [c for c in self.text_cols if c in items_unique.columns and c != item_col]
        if self.metadata_cols is not None:
            resolved_metadata_cols = [c for c in self.metadata_cols if c in items_unique.columns and c != item_col]

        if self.text_cols is None and self.metadata_cols is None:
            # Auto mode: string-like columns as text, bool/numeric as metadata.
            for c in items_unique.columns:
                if c == item_col:
                    continue
                if pd.api.types.is_object_dtype(items_unique[c]) or pd.api.types.is_string_dtype(items_unique[c]):
                    resolved_text_cols.append(c)
                elif pd.api.types.is_bool_dtype(items_unique[c]) or pd.api.types.is_numeric_dtype(items_unique[c]):
                    resolved_metadata_cols.append(c)

        # Avoid accidental duplicates.
        resolved_text_cols = [c for c in resolved_text_cols if c not in resolved_metadata_cols]
        return resolved_text_cols, resolved_metadata_cols

    def _interaction_weight(self, row, reference_time, event_col, time_col):
        base_weight = 1.0
        if event_col is not None and event_col in row and event_col in self.event_weight_map:
            base_weight = float(self.event_weight_map[row[event_col]])

        if self.recency_decay <= 0.0 or reference_time is None or time_col is None or pd.isna(row[time_col]):
            return base_weight

        age_days = self._safe_strftime_days(reference_time - row[time_col])
        recency_weight = np.exp(-self.recency_decay * max(0.0, age_days))
        return base_weight * recency_weight

    def _build_user_profiles(self, train_df, user_col, item_col, event_col, time_col):
        from scipy.sparse import csr_matrix

        self.user_profiles = {}
        self.user_history_counts = train_df.groupby(user_col)[item_col].size().to_dict()

        reference_time = None
        if time_col is not None:
            if not pd.api.types.is_datetime64_any_dtype(train_df[time_col]):
                train_df = train_df.copy()
                train_df[time_col] = pd.to_datetime(train_df[time_col], errors='coerce')
            reference_time = train_df[time_col].max()

        for user_id, user_events in train_df.groupby(user_col):
            mapped_indices = []
            weights = []
            for _, row in user_events.iterrows():
                item = row[item_col]
                if item not in self.item_mapping:
                    continue
                mapped_indices.append(self.item_mapping[item])
                weights.append(self._interaction_weight(row, reference_time, event_col, time_col))

            if not mapped_indices:
                continue

            weight_array = np.asarray(weights, dtype=float)
            denom = weight_array.sum()
            if denom <= 0:
                continue

            user_matrix = self.item_feature_matrix[mapped_indices]
            weighted = user_matrix.multiply(weight_array[:, None])
            profile = weighted.sum(axis=0) / denom
            self.user_profiles[user_id] = csr_matrix(profile)

    def fit(self, train_df, items_df, user_col='user_id', item_col='venue_id'):
        from sklearn.feature_extraction.text import TfidfVectorizer

        print("Training Content-Based Recommender...")

        if item_col not in train_df.columns:
            raise ValueError(f"'{item_col}' missing in train_df.")
        if user_col not in train_df.columns:
            raise ValueError(f"'{user_col}' missing in train_df.")
        if item_col not in items_df.columns:
            raise ValueError(f"'{item_col}' missing in items_df.")

        self.popular_fallback = train_df[item_col].value_counts().index.tolist()

        items_unique = items_df.drop_duplicates(subset=[item_col]).copy()
        items_unique = items_unique[items_unique[item_col].isin(train_df[item_col].unique())]

        text_cols, metadata_cols = self._resolve_feature_columns(items_unique, item_col)

        if not text_cols and not metadata_cols:
            raise ValueError("No valid feature columns found for content-based training.")

        print(f"Using text columns: {text_cols}")
        print(f"Using metadata columns: {metadata_cols}")

        self.item_mapping = {item: idx for idx, item in enumerate(items_unique[item_col].tolist())}
        self.reverse_item_mapping = {idx: item for item, idx in self.item_mapping.items()}

        all_feature_cols = text_cols + metadata_cols
        self._feature_missing_rates = {
            c: float(items_unique[c].isna().mean()) for c in all_feature_cols
        }

        item_docs = [
            self._item_to_document(
                row,
                item_col=item_col,
                text_cols=text_cols,
                metadata_cols=metadata_cols,
            )
            for _, row in items_unique.iterrows()
        ]

        self.vectorizer = TfidfVectorizer(min_df=self.min_df, ngram_range=self.ngram_range)
        self.item_feature_matrix = self.vectorizer.fit_transform(item_docs)
        self.feature_names = self.vectorizer.get_feature_names_out()
        print(f"Feature dimension: {self.item_feature_matrix.shape[1]}")
        print(f"Vocabulary size: {len(self.feature_names)}")
        if len(self._feature_missing_rates) > 0:
            print(f"Missing rates by feature: {self._feature_missing_rates}")

        effective_event_col = self.event_col if self.event_col in train_df.columns else None
        effective_time_col = self.time_col if self.time_col in train_df.columns else None
        self._build_user_profiles(
            train_df=train_df,
            user_col=user_col,
            item_col=item_col,
            event_col=effective_event_col,
            time_col=effective_time_col,
        )

        print(f"Content model trained with {len(self.item_mapping)} items and {len(self.user_profiles)} user profiles.")

    def _score_all_items(self, user_id):
        from sklearn.metrics.pairwise import cosine_similarity

        profile = self.user_profiles[user_id]
        return cosine_similarity(profile, self.item_feature_matrix).ravel()

    def _explain_item(self, profile, item_idx, top_n=3):
        item_vec = self.item_feature_matrix.getrow(item_idx)
        overlap = profile.multiply(item_vec)
        if overlap.nnz == 0:
            return "Recommended due to overall profile match."

        top = np.argsort(overlap.data)[-top_n:][::-1]
        feat_ids = overlap.indices[top]
        feats = [self.feature_names[i] for i in feat_ids]
        if not feats:
            return "Recommended due to overall profile match."
        return f"Recommended because it overlaps on: {', '.join(feats)}."

    def recommend(self, user_id, user_history, k=10, return_explanations=False):
        user_history_set = set(user_history)

        if user_id not in self.user_profiles:
            recs = []
            for item in self.popular_fallback:
                if item not in user_history_set:
                    recs.append(item)
                if len(recs) == k:
                    break
            if return_explanations:
                return [(r, "Recommended due to popularity fallback (cold user).") for r in recs]
            return recs

        profile = self.user_profiles[user_id]
        scores = self._score_all_items(user_id)

        for item in user_history_set:
            if item in self.item_mapping:
                scores[self.item_mapping[item]] = -np.inf

        top_indices = np.argsort(scores)[-k:][::-1]
        recs = [self.reverse_item_mapping[idx] for idx in top_indices if np.isfinite(scores[idx])]

        if len(recs) < k:
            recs_set = set(recs)
            for item in self.popular_fallback:
                if item not in user_history_set and item not in recs_set:
                    recs.append(item)
                if len(recs) == k:
                    break

        recs = recs[:k]
        if not return_explanations:
            return recs

        explained = []
        for item in recs:
            if item in self.item_mapping:
                item_idx = self.item_mapping[item]
                explained.append((item, self._explain_item(profile, item_idx)))
            else:
                explained.append((item, "Recommended due to popularity fallback."))
        return explained

    @staticmethod
    def cold_item_set(train_df, test_df, item_col='venue_id'):
        train_items = set(train_df[item_col].unique())
        return set(test_df[item_col].unique()) - train_items

    @staticmethod
    def sparse_user_set(train_df, user_col='user_id', max_interactions=5):
        counts = train_df.groupby(user_col).size()
        return set(counts[counts <= max_interactions].index.tolist())


class HybridMFContentRecommender:
    """
    Lightweight rank-based hybrid between an MF-like model and content-based model.
    Uses reciprocal-rank blending, so it can work with models that expose only
    recommend(...) but not raw score vectors.
    """

    def __init__(self, mf_model, content_model, alpha=0.5, adaptive=False, sparse_threshold=5):
        self.mf_model = mf_model
        self.content_model = content_model
        self.alpha = alpha
        self.adaptive = adaptive
        self.sparse_threshold = sparse_threshold

    @staticmethod
    def _rr_scores(ranked_items):
        return {item: 1.0 / (rank + 1.0) for rank, item in enumerate(ranked_items)}

    def _resolve_alpha(self, user_id):
        if not self.adaptive:
            return self.alpha
        count = self.content_model.user_history_counts.get(user_id, 0)
        if count <= self.sparse_threshold:
            return min(self.alpha, 0.3)  # Lean to content for sparse users.
        return self.alpha

    def recommend(self, user_id, user_history, k=10):
        # Pull larger pools from both models for better fusion.
        pool_k = max(k * 5, 50)
        mf_recs = self.mf_model.recommend(user_id, user_history, k=pool_k)
        cb_recs = self.content_model.recommend(user_id, user_history, k=pool_k)

        mf_scores = self._rr_scores(mf_recs)
        cb_scores = self._rr_scores(cb_recs)
        alpha = self._resolve_alpha(user_id)

        all_items = set(mf_scores.keys()) | set(cb_scores.keys())
        blended = []
        for item in all_items:
            score = alpha * mf_scores.get(item, 0.0) + (1.0 - alpha) * cb_scores.get(item, 0.0)
            blended.append((item, score))

        blended.sort(key=lambda x: x[1], reverse=True)

        recs = []
        seen = set(user_history)
        for item, _ in blended:
            if item in seen:
                continue
            recs.append(item)
            if len(recs) == k:
                break
        return recs


if __name__ == "__main__":
    print(
        "content_based.py defines model classes and is not a standalone training script.\n"
        "Use it via import, e.g.:\n"
        "from models.content_based import ContentBasedRecommender"
    )
