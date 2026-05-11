import time


def candidate_recall_at_k(candidate_items, true_items, k=300):
    """
    Candidate Recall@K for two-stage recommender evaluation.

    Measures whether the relevant held-out item survives the retrieval stage.

    In two-stage recommenders, this is important because:
    if the true item is not in the candidate set, the ranker cannot recover it later.

    Parameters
    ----------
    candidate_items : list
        Items returned by the candidate generator / retriever.
    true_items : list or set
        Relevant held-out items for the user.
        In leave-last-out evaluation this usually contains one item.
    k : int
        Candidate budget, e.g. 50, 100, 300, 500.

    Returns
    -------
    float
        Fraction of relevant items found in the top-K candidate set.
    """
    top_k_candidates = candidate_items[:k]
    candidate_set = set(top_k_candidates)

    if len(true_items) == 0:
        return 0.0

    hits = [1 for item in true_items if item in candidate_set]
    return sum(hits) / len(true_items)


def candidate_hit_at_k(candidate_items, true_item, k=300):
    """
    Candidate Hit@K for leave-last-out evaluation.

    Since we usually have exactly one held-out item per user,
    this is equivalent to Candidate Recall@K.

    Returns 1.0 if the true item survived retrieval, otherwise 0.0.
    """
    top_k_candidates = candidate_items[:k]
    return 1.0 if true_item in top_k_candidates else 0.0


def candidate_coverage_at_k(all_candidate_lists, total_catalog_size, k=300):
    """
    Candidate Coverage@K.

    Measures how much of the catalog appears in any retrieved candidate set.

    High coverage means the retriever explores a broader part of the catalog.
    Low coverage often indicates popularity bias or candidate concentration.
    """
    unique_candidates = set()

    for candidate_list in all_candidate_lists:
        unique_candidates.update(candidate_list[:k])

    if total_catalog_size == 0:
        return 0.0

    return len(unique_candidates) / total_catalog_size


def average_candidate_set_size(all_candidate_lists, k=300):
    """
    Average number of candidates returned per user.

    Useful to check whether a retriever actually fills the requested budget.
    For example, if candidate_k=300 but the model only returns 40 items,
    downstream ranking quality is limited.
    """
    if len(all_candidate_lists) == 0:
        return 0.0

    sizes = [len(candidate_list[:k]) for candidate_list in all_candidate_lists]
    return sum(sizes) / len(sizes)


def duplicate_rate_at_k(candidate_items, k=300):
    """
    Duplicate Rate@K for a single candidate list.

    Normally candidate lists should not contain duplicates.
    A high duplicate rate indicates a bug in branch merging or retrieval logic.
    """
    top_k_candidates = candidate_items[:k]

    if len(top_k_candidates) == 0:
        return 0.0

    unique_count = len(set(top_k_candidates))
    duplicate_count = len(top_k_candidates) - unique_count

    return duplicate_count / len(top_k_candidates)


def mean_duplicate_rate_at_k(all_candidate_lists, k=300):
    """
    Mean Duplicate Rate@K across all users.

    This is mainly a debugging metric for merged candidate generators.
    """
    if len(all_candidate_lists) == 0:
        return 0.0

    rates = [
        duplicate_rate_at_k(candidate_list, k=k)
        for candidate_list in all_candidate_lists
    ]

    return sum(rates) / len(rates)


def retrieval_latency_ms(retrieve_function, *args, **kwargs):
    """
    Measures retrieval latency in milliseconds for a single user request.

    Example
    -------
    latency = retrieval_latency_ms(
        model.recommend,
        user_id,
        user_history,
        k=300
    )

    This is useful for two-stage recommender analysis because retrieval
    must be fast enough to serve as the first stage.
    """
    start_time = time.perf_counter()
    result = retrieve_function(*args, **kwargs)
    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000
    return result, latency_ms


def summarize_retrieval_metrics(
    all_candidate_lists,
    true_items_by_user,
    user_ids,
    total_catalog_size,
    k=300,
):
    """
    Computes a compact metrics dictionary for retrieval evaluation.

    Parameters
    ----------
    all_candidate_lists : list of lists
        Candidate lists in the same order as user_ids.
    true_items_by_user : dict
        Mapping from user_id to held-out item or list of held-out items.
    user_ids : list
        Users evaluated.
    total_catalog_size : int
        Number of unique items in the catalog.
    k : int
        Candidate budget.

    Returns
    -------
    dict
        Retrieval metrics for logging in runs.csv or displaying in notebooks.
    """
    recalls = []

    for user_id, candidate_list in zip(user_ids, all_candidate_lists):
        true_items = true_items_by_user[user_id]

        # Allow both single item and list/set of items
        if not isinstance(true_items, (list, set, tuple)):
            true_items = [true_items]

        recalls.append(
            candidate_recall_at_k(
                candidate_items=candidate_list,
                true_items=true_items,
                k=k,
            )
        )

    return {
        f"candidate_recall_at_{k}": sum(recalls) / len(recalls) if recalls else 0.0,
        f"candidate_coverage_at_{k}": candidate_coverage_at_k(
            all_candidate_lists,
            total_catalog_size,
            k=k,
        ),
        f"avg_candidate_set_size_at_{k}": average_candidate_set_size(
            all_candidate_lists,
            k=k,
        ),
        f"mean_duplicate_rate_at_{k}": mean_duplicate_rate_at_k(
            all_candidate_lists,
            k=k,
        ),
    }