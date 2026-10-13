def recall_at_k(retrieved_is_gold, k):
    return 1.0 if any(retrieved_is_gold[:k]) else 0.0


def reciprocal_rank(retrieved_is_gold):
    for i, is_gold in enumerate(retrieved_is_gold, start=1):
        if is_gold:
            return 1.0 / i
    return 0.0
