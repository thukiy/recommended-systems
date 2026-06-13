from collections import defaultdict, Counter
import random
import hashlib


class PersonalizedPageRankRecommender:
    """
    Lightweight graph-based recommender using an approximate Personalized PageRank
    / Random Walk with Restart idea on a bipartite user-item interaction graph.

    The training data is interpreted as a graph:
        user -- interacted_with -- item

    At recommendation time, short random walks start from the target user.
    Items that are visited often are treated as relevant candidates.

    This is intentionally a simple graph baseline:
    - no neural training
    - no external dependencies
    - compatible with the existing recommend(user_id, user_history, k) interface

    Parameters
    ----------
    alpha : float
        Restart probability. Higher values keep walks closer to the user.
    max_steps : int
        Number of graph steps per walk. Odd values such as 3 or 5 can end on items.
    n_walks : int
        Number of random walks per recommendation call.
    random_state : int or None
        Seed for reproducible recommendations.
    cache_recommendations : bool
        Cache ranked candidates per user. Useful because evaluation calls recommend
        repeatedly for the same users with different k values.
    max_cache_items : int
        Maximum number of recommendations to store per user in the cache.
    """

    def __init__(
        self,
        alpha=0.15,
        max_steps=3,
        n_walks=200,
        random_state=None,
        cache_recommendations=True,
        max_cache_items=500,
    ):
        self.alpha = alpha
        self.max_steps = max_steps
        self.n_walks = n_walks
        self.random_state = random_state
        self.cache_recommendations = cache_recommendations
        self.max_cache_items = max_cache_items

        self.user_to_items = defaultdict(list)
        self.item_to_users = defaultdict(list)
        self.popular_fallback = []
        self._cache = {}

    def fit(self, train_df, user_col="user_id", item_col="recipe_id"):
        print(
            "Training Graph-Based Personalized PageRank Recommender "
            f"(alpha={self.alpha}, max_steps={self.max_steps}, n_walks={self.n_walks})..."
        )

        required_cols = {user_col, item_col}
        missing = required_cols - set(train_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        df = train_df[[user_col, item_col]].drop_duplicates().copy()

        # Keep a global fallback list for cold users or sparse graph neighborhoods.
        self.popular_fallback = train_df[item_col].value_counts().index.tolist()

        # Store the bipartite graph in both directions for cheap neighbor lookups.
        for user_id, item_id in zip(df[user_col].values, df[item_col].values):
            self.user_to_items[user_id].append(item_id)
            self.item_to_users[item_id].append(user_id)

        # Any previous cached rankings are invalid after rebuilding the graph.
        self._cache = {}

        n_users = len(self.user_to_items)
        n_items = len(self.item_to_users)
        n_edges = len(df)
        print(f"Graph contains {n_users:,} users, {n_items:,} items, and {n_edges:,} user-item edges.")

    def _stable_seed(self, user_id):
        base = 0 if self.random_state is None else int(self.random_state)
        # Derive a deterministic per-user seed so cached and repeated calls are stable.
        digest = hashlib.md5(str(user_id).encode("utf-8")).hexdigest()
        user_seed = int(digest[:8], 16)
        return (base + user_seed) % (2**32)

    def _neighbors(self, node_type, node_id):
        # The graph alternates between user nodes and item nodes.
        if node_type == "user":
            return [("item", item_id) for item_id in self.user_to_items.get(node_id, [])]
        if node_type == "item":
            return [("user", user_id) for user_id in self.item_to_users.get(node_id, [])]
        return []

    def _random_walk_scores(self, user_id, seen_items):
        rng = random.Random(self._stable_seed(user_id))
        scores = Counter()

        if user_id not in self.user_to_items:
            # Cold users cannot start a meaningful walk; recommend() will use fallback items.
            return scores

        for _ in range(self.n_walks):
            node_type = "user"
            node_id = user_id

            for _step in range(self.max_steps):
                # Random Walk with Restart: jump back to the target user.
                if rng.random() < self.alpha:
                    node_type = "user"
                    node_id = user_id
                    continue

                neighbors = self._neighbors(node_type, node_id)
                if not neighbors:
                    node_type = "user"
                    node_id = user_id
                    continue

                node_type, node_id = rng.choice(neighbors)

                # Only item visits are recommendation evidence.
                if node_type == "item" and node_id not in seen_items:
                    scores[node_id] += 1

        return scores

    def recommend(self, user_id, user_history, k=10):
        seen_items = set(user_history)
        cache_key = user_id

        # Evaluation often asks for the same user repeatedly with different k values.
        if self.cache_recommendations and cache_key in self._cache:
            cached = self._cache[cache_key]
            if len(cached) >= k:
                return cached[:k]

        scores = self._random_walk_scores(user_id, seen_items)

        ranked_items = [
            item_id
            for item_id, _ in scores.most_common()
            if item_id not in seen_items
        ]

        # Cache a larger candidate pool than requested so later larger k calls can reuse it.
        ranked_set = set(ranked_items)
        target_len = max(k, self.max_cache_items if self.cache_recommendations else k)

        # Popularity fallback ensures that every user receives k recommendations,
        # even if the local graph neighborhood is too small.
        for item_id in self.popular_fallback:
            if item_id not in seen_items and item_id not in ranked_set:
                ranked_items.append(item_id)
                ranked_set.add(item_id)

            if len(ranked_items) >= target_len:
                break

        if self.cache_recommendations:
            self._cache[cache_key] = ranked_items[:target_len]

        return ranked_items[:k]
