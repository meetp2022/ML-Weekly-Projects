import numpy as np

def precision_at_k(retrieved_ids, true_id, k):
    return (1 if true_id in retrieved_ids[:k] else 0) / k

def recall_at_k(retrieved_ids, true_id, k):
    return 1.0 if true_id in retrieved_ids[:k] else 0.0

def f1_score_at_k(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def mrr(retrieved_ids, true_id):
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid == true_id:
            return 1.0 / rank
    return 0.0
