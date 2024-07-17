import numpy as np

ROUND_DECIMALS = 2
def get_metrics(fragment_filename, k_candidates, total_relevants):
    coincidences = [1 if filename == fragment_filename else 0 for filename in k_candidates]

    def precision_topk():
        return coincidences.count(1) / len(k_candidates)

    def recall_topk():
        if total_relevants == 0:
            return 0
        return coincidences.count(1) / total_relevants

    def f1_score_topk():
        precision = precision_topk()
        recall = recall_topk()
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

    # def mrr():
    #     for i, candidate in enumerate(k_candidates):
    #         if candidate == fragment_filename:
    #             return 1 / (i + 1)
    #     return 0

    # def map():
    #     def average_precision(fragment_filename, k_candidates):
    #         relevant_count = 0
    #         precision_sum = 0
    #         for i, candidate in enumerate(k_candidates, 1):
    #             if candidate == fragment_filename:
    #                 relevant_count += 1
    #                 precision_sum += relevant_count / i
    #         return precision_sum / (1 if relevant_count == 0 else relevant_count)
    #     ap_sum = 0
    #     for i in range(len(k_candidates)):
    #         ap_sum += average_precision(fragment_filename, k_candidates)
    #     return ap_sum / len(k_candidates)

    # def ndcg():
    #     def dcg(k_candidates):
    #         return sum((1 / np.log2(i + 2) if candidate == fragment_filename else 0)
    #                    for i, candidate in enumerate(k_candidates))
    #     ideal_dcg = 1 / np.log2(1 + 1)
    #     actual_dcg = dcg(k_candidates)
    #     return actual_dcg / ideal_dcg


    metrics = {
        "precision@k": round(precision_topk(), ROUND_DECIMALS),
        "recall@k": round(recall_topk(), ROUND_DECIMALS),
        "F1-score": round(f1_score_topk(), ROUND_DECIMALS)
    }

    return metrics

# # input fragment
# fragment_filename = 'fragment.jpg'
# # input k candidates
# k_candidates = ['fragment.jpg' for _ in range(5)]
# # k_candidates = ['fragment.jpg', 'fragment.jpg', 'fragmen5t.jpg', 'fragment7.jpg', 'fragmen7t.jpg']
# # number of all relevant documents in dataset
# total_relevants = 6
# print(get_metrics(fragment_filename, k_candidates, total_relevants))
