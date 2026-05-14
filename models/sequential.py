from collections import defaultdict, Counter


class MarkovRecommender:
    """
    First-order Markov recommender for sequential recommendation.

    The model learns item-to-item transition counts from chronological
    user histories. At recommendation time, it looks at the user's last
    interacted item and recommends the most frequent successors.

    If the last item is unknown or has no outgoing transitions, the model
    falls back to global popularity.
    """

    def __init__(self):
        self.transition_counts = defaultdict(Counter)
        self.transition_rankings = {}
        self.popular_fallback = []

    def fit(self, train_df, user_col="user_id", item_col="recipe_id", time_col="date_time"):
        print("Training First-Order Markov Sequential Recommender...")

        required_cols = {user_col, item_col, time_col}
        missing = required_cols - set(train_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        df = train_df[[user_col, item_col, time_col]].copy()
        df = df.sort_values([user_col, time_col])

        self.popular_fallback = df[item_col].value_counts().index.tolist()

        for _, user_events in df.groupby(user_col):
            items = user_events[item_col].tolist()

            if len(items) < 2:
                continue

            for current_item, next_item in zip(items[:-1], items[1:]):
                self.transition_counts[current_item][next_item] += 1

        self.transition_rankings = {
            item: [next_item for next_item, _ in counter.most_common()]
            for item, counter in self.transition_counts.items()
        }

        print(f"Learned transitions for {len(self.transition_rankings)} source recipes.")

    def recommend(self, user_id, user_history, k=10):
        user_history = list(user_history)
        seen = set(user_history)

        recommendations = []

        if len(user_history) > 0:
            last_item = user_history[-1]
            successors = self.transition_rankings.get(last_item, [])

            for item in successors:
                if item not in seen:
                    recommendations.append(item)

                if len(recommendations) == k:
                    return recommendations

        recs_set = set(recommendations)

        for item in self.popular_fallback:
            if item not in seen and item not in recs_set:
                recommendations.append(item)

            if len(recommendations) == k:
                break

        return recommendations