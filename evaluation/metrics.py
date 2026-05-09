import numpy as np
import math


def novelty_at_k(recommended_items, train_item_counts, total_train_interactions, k=10):
    """
    Week 2: Novelty@K (Slide 52).
    Rewards the model for recommending items that are rare in the training set.
    Formula: -log2(popularity(i))
    """
    top_k_recs = recommended_items[:k]
    novelty_score = 0.0

    for item in top_k_recs:
        # Calculate probability of item in training set (popularity)
        # If item was never seen in train (cold item), give it a tiny probability to avoid log(0)
        item_count = train_item_counts.get(item, 1e-9)
        item_pop = item_count / total_train_interactions

        # Add the novelty: -log2(p)
        novelty_score += -math.log2(item_pop)

    return novelty_score / len(top_k_recs) if len(top_k_recs) > 0 else 0.0


def recall_at_k(recommended_items, true_items, k=10):
    """
    Recall@K for leave-last-out evaluation.
    Measures whether the held-out item appears in the top-K recommendations.
    """
    # Truncate the recommendation list to the top K items
    top_k_recs = recommended_items[:k]

    # For leave-last-out, each user has at most one relevant test item.
    # If the item appears in the top-K list, it counts as a hit.
    hits = [1 for item in true_items if item in top_k_recs]

    if len(true_items) == 0:
        return 0.0  # Handle users without test items

    return sum(hits) / len(true_items)


def ndcg_at_k(recommended_items, true_items, k=10):
    """
    Normalized Discounted Cumulative Gain at K (NDCG@K).
    Rewards relevant items placed near the top of the ranked list.
    """
    top_k_recs = recommended_items[:k]
    dcg = 0.0

    for i, item in enumerate(top_k_recs):
        if item in true_items:
            # Relevance is binary; discount depends on the rank position.
            dcg += 1.0 / np.log2((i + 1) + 1)

    # In leave-last-out evaluation, each user has exactly one relevant test item.
    # Therefore, the ideal DCG is always 1.0 because the best possible rank is position 1.
    idcg = 1.0

    return dcg / idcg


def catalog_coverage_at_k(all_recommended_items_lists, total_catalog_size):
    """
    Catalog Coverage@K.
    Measures the share of the item catalog that appears in the model's recommendations.
    """
    unique_recommended_items = set()
    for rec_list in all_recommended_items_lists:
        unique_recommended_items.update(rec_list)

    return len(unique_recommended_items) / total_catalog_size