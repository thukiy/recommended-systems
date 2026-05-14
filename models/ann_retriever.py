"""
ANN / vector-index retriever for recommender-system candidate generation.

This module wraps an already trained matrix-factorization model (for example
BPRMatrixFactorization) and exposes the same recommend(user_id, user_history, k)
interface as the other models in this project.

Why this exists
---------------
Matrix factorization already learns user vectors P and item vectors Q. A two-tower
retriever uses exactly this idea operationally:

    user tower: user_id/history -> user vector
    item tower: item_id/features -> item vector
    retrieval:  nearest items by dot product

For small catalogs we can score every item exactly. For large catalogs we want an
index so that candidate generation becomes fast enough for serving. If faiss is
installed, this class uses a FAISS inner-product index. Otherwise it falls back to
an exact NumPy dot-product search with the same public API, so notebooks keep
working without extra system dependencies.
"""

from __future__ import annotations

import time
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # Optional dependency. Install with: pip install faiss-cpu
    import faiss  # type: ignore
except Exception:  # pragma: no cover - depends on local environment
    faiss = None


class ANNMatrixFactorizationRetriever:
    """
    Candidate retriever over matrix-factorization embeddings.

    Parameters
    ----------
    backend : {"auto", "faiss", "numpy"}
        "auto" uses FAISS when available and NumPy otherwise.
    include_item_bias : bool
        If True and the MF model has item biases ``b_i``, append an extra vector
        dimension so dot(user, item) also contains the item bias. This makes ANN
        rankings closer to BPR/MF recommend(...), where scores are Q @ P + b_i.
    normalize : bool
        If True, L2-normalize vectors before indexing. This turns inner product
        into cosine-style retrieval. For closest match to the current BPR model,
        keep this False.
    overfetch_factor : int
        Fetch more than k candidates before filtering already-seen items.
    fallback_to_popularity : bool
        Fill missing slots with popularity fallback if the index cannot return
        enough unseen candidates.
    """

    def __init__(
        self,
        backend: str = "auto",
        include_item_bias: bool = True,
        normalize: bool = False,
        overfetch_factor: int = 5,
        fallback_to_popularity: bool = True,
    ):
        valid_backends = {"auto", "faiss", "numpy"}
        if backend not in valid_backends:
            raise ValueError(f"backend must be one of {sorted(valid_backends)}")

        self.backend = backend
        self.include_item_bias = include_item_bias
        self.normalize = normalize
        self.overfetch_factor = max(1, int(overfetch_factor))
        self.fallback_to_popularity = fallback_to_popularity

        self.active_backend: Optional[str] = None
        self.index = None

        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_item_mapping = {}
        self.popular_fallback: List = []

        self.user_vectors: Optional[np.ndarray] = None
        self.item_vectors: Optional[np.ndarray] = None
        self.item_bias: Optional[np.ndarray] = None

    @staticmethod
    def _as_float32(matrix: np.ndarray) -> np.ndarray:
        return np.asarray(matrix, dtype=np.float32, order="C")

    @staticmethod
    def _l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, eps)

    def _prepare_vectors(
        self,
        user_vectors: np.ndarray,
        item_vectors: np.ndarray,
        item_bias: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        users = self._as_float32(user_vectors)
        items = self._as_float32(item_vectors)

        if self.include_item_bias and item_bias is not None:
            bias_col = self._as_float32(item_bias).reshape(-1, 1)
            items = np.hstack([items, bias_col])
            users = np.hstack([users, np.ones((users.shape[0], 1), dtype=np.float32)])

        if self.normalize:
            users = self._l2_normalize(users)
            items = self._l2_normalize(items)

        return users, items

    def fit_from_mf_model(self, mf_model) -> "ANNMatrixFactorizationRetriever":
        """
        Build a vector index from a trained MF/BPR model.

        The model must expose:
        - P: user-factor matrix
        - Q: item-factor matrix
        - user_mapping
        - item_mapping
        - reverse_item_mapping

        Optional:
        - b_i: item biases
        - popular_fallback: fallback ranking
        """
        required_attrs = ["P", "Q", "user_mapping", "item_mapping", "reverse_item_mapping"]
        missing = [attr for attr in required_attrs if not hasattr(mf_model, attr)]
        if missing:
            raise ValueError(f"MF model is missing required attributes: {missing}")
        if mf_model.P is None or mf_model.Q is None:
            raise ValueError("MF model vectors are empty. Fit the MF model before building the ANN index.")

        self.user_mapping = dict(mf_model.user_mapping)
        self.item_mapping = dict(mf_model.item_mapping)
        self.reverse_item_mapping = dict(mf_model.reverse_item_mapping)
        self.popular_fallback = list(getattr(mf_model, "popular_fallback", []))
        self.item_bias = getattr(mf_model, "b_i", None)

        self.user_vectors, self.item_vectors = self._prepare_vectors(
            user_vectors=mf_model.P,
            item_vectors=mf_model.Q,
            item_bias=self.item_bias,
        )

        self._build_index()
        return self

    def _build_index(self) -> None:
        if self.item_vectors is None:
            raise ValueError("No item vectors available. Call fit_from_mf_model first.")

        requested = self.backend
        if requested == "auto":
            requested = "faiss" if faiss is not None else "numpy"

        if requested == "faiss":
            if faiss is None:
                raise ImportError("FAISS is not installed. Use backend='numpy' or install faiss-cpu.")
            dim = self.item_vectors.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(self.item_vectors)
            self.index = index
            self.active_backend = "faiss"
        else:
            self.index = None
            self.active_backend = "numpy"

        print(
            f"Built ANN retriever over {self.item_vectors.shape[0]:,} items "
            f"with dim={self.item_vectors.shape[1]} using backend='{self.active_backend}'."
        )

    def _get_user_vector(self, user_id) -> Optional[np.ndarray]:
        if self.user_vectors is None:
            raise ValueError("Retriever has not been fitted yet.")
        if user_id not in self.user_mapping:
            return None
        return self.user_vectors[self.user_mapping[user_id]].reshape(1, -1)

    def _search_raw(self, user_vector: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.item_vectors is None:
            raise ValueError("Retriever has not been fitted yet.")

        n = min(int(n), self.item_vectors.shape[0])

        if self.active_backend == "faiss":
            scores, indices = self.index.search(self._as_float32(user_vector), n)
            return scores[0], indices[0]

        # NumPy fallback: exact inner-product search. Good for offline comparison.
        scores = self.item_vectors @ user_vector.ravel()
        if n >= len(scores):
            top_idx = np.argsort(scores)[::-1]
        else:
            unsorted = np.argpartition(scores, -n)[-n:]
            top_idx = unsorted[np.argsort(scores[unsorted])[::-1]]
        return scores[top_idx], top_idx

    def recommend(self, user_id, user_history: Iterable, k: int = 10) -> List:
        """
        Return top-k ANN candidates while filtering already-seen items.
        """
        user_history_set = set(user_history)
        user_vector = self._get_user_vector(user_id)

        if user_vector is None:
            return self._popularity_fallback(user_history_set, k)

        fetch_k = min(
            max(k * self.overfetch_factor, k + len(user_history_set)),
            len(self.reverse_item_mapping),
        )

        _, indices = self._search_raw(user_vector, fetch_k)

        recs = []
        for idx in indices:
            if idx < 0:
                continue
            item = self.reverse_item_mapping[int(idx)]
            if item in user_history_set:
                continue
            recs.append(item)
            if len(recs) == k:
                break

        if len(recs) < k and self.fallback_to_popularity:
            recs_set = set(recs)
            for item in self.popular_fallback:
                if item in user_history_set or item in recs_set:
                    continue
                recs.append(item)
                if len(recs) == k:
                    break

        return recs[:k]

    def recommend_with_latency(self, user_id, user_history: Iterable, k: int = 10) -> Tuple[List, float]:
        """
        Convenience method for serving-style diagnostics.
        Returns (recommendations, latency_ms).
        """
        start = time.perf_counter()
        recs = self.recommend(user_id=user_id, user_history=user_history, k=k)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return recs, latency_ms

    def _popularity_fallback(self, user_history_set: set, k: int) -> List:
        recs = []
        for item in self.popular_fallback:
            if item in user_history_set:
                continue
            recs.append(item)
            if len(recs) == k:
                break
        return recs

    def score(self, user_id, item_id) -> float:
        """
        Score one user-item pair with the same vector geometry used by retrieval.
        Useful for diagnostics or LTR features.
        """
        if (
            self.user_vectors is None
            or self.item_vectors is None
            or user_id not in self.user_mapping
            or item_id not in self.item_mapping
        ):
            return 0.0
        u = self.user_mapping[user_id]
        i = self.item_mapping[item_id]
        return float(np.dot(self.user_vectors[u], self.item_vectors[i]))


if __name__ == "__main__":
    print(
        "ann_retriever.py defines ANNMatrixFactorizationRetriever.\n"
        "Use it after fitting BPR MF, e.g.:\n"
        "ann_bpr = ANNMatrixFactorizationRetriever().fit_from_mf_model(bpr_model)"
    )
