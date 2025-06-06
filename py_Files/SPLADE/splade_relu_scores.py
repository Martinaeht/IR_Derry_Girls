from queries_splade import queries
from splade_relu_log import search
from scipy.sparse import load_npz
import numpy as np
import json

def evaluate(queries, index, metadata, top_k=5):
    results = {}
    total_recall = total_precision = total_mrr = total_map = total_ndcg = total_accuracy = 0

    for i, (query, expected_indices) in enumerate(queries.items()):

        search_results = search(query, index, metadata, top_k=top_k)
        retrieved_ids = [meta["id"] for meta, _ in search_results]

        retrieved_answers = [
            f"{j+1}. ID: {meta.get('id', 'N/A')}, Score: {sim:.4f}, Line: {meta.get('clean_text', '[No clean_text field]')}"
            for j, (meta, sim) in enumerate(search_results)
        ]

        relevant_retrieved = [rid for rid in retrieved_ids if rid in expected_indices]
        recall_at_k = len(relevant_retrieved) / len(expected_indices) if expected_indices else 0
        precision_at_k = len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0
        accuracy = 1 if any(rid in expected_indices for rid in retrieved_ids) else 0

        mrr = 0
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in expected_indices:
                mrr = 1 / rank
                break

        avg_precision = 0
        relevant_count = 0
        precision_sum = 0
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in expected_indices:
                relevant_count += 1
                precision_sum += relevant_count / rank
        avg_precision = precision_sum / len(expected_indices) if expected_indices else 0

        def calculate_dcg(relevance_scores):
            return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores))

        relevance_scores = [1 if rid in expected_indices else 0 for rid in retrieved_ids]
        dcg = calculate_dcg(relevance_scores)
        ideal_relevance = [1] * min(len(expected_indices), top_k)
        idcg = calculate_dcg(ideal_relevance)
        ndcg = dcg / idcg if idcg > 0 else 0

        results[query] = {
            "Recall@k": recall_at_k,
            "Precision@k": precision_at_k,
            "Accuracy": accuracy,
            "MRR": mrr,
            "MAP": avg_precision,
            "NDCG@k": ndcg,
            "Retrieved Answers": retrieved_answers
        }

        total_recall += recall_at_k
        total_precision += precision_at_k
        total_accuracy += accuracy
        total_mrr += mrr
        total_map += avg_precision
        total_ndcg += ndcg


    num_queries = len(queries)
    print(f"\nOverall average results:")
    print(f"Average Recall@{top_k}: {total_recall / num_queries:.4f}")
    print(f"Average Precision@{top_k}: {total_precision / num_queries:.4f}")
    print(f"Average Accuracy: {total_accuracy / num_queries:.4f}")
    print(f"Average MRR: {total_mrr / num_queries:.4f}")
    print(f"Average MAP: {total_map / num_queries:.4f}")
    print(f"Average NDCG@{top_k}: {total_ndcg / num_queries:.4f}")

    return results

index_path = "/home/mlt_ml3/IR_Derry_Girls/py_Files/SPLADE/splade_index_relu.npz"
metadata_path = "/home/mlt_ml3/IR_Derry_Girls/py_Files/SPLADE/metadata_relu.json"

index = load_npz(index_path)
with open(metadata_path, "r") as f:
    metadata = json.load(f)

top_k = 5

evaluation_results = evaluate(queries, index, metadata, top_k=top_k)

print("\nFinal results:")
for query, metrics in evaluation_results.items():
    print(f"Query: {query}")
    print(f"Recall@{top_k}: {metrics['Recall@k']:.4f}")
    print(f"Precision@{top_k}: {metrics['Precision@k']:.4f}")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"MRR: {metrics['MRR']:.4f}")
    print(f"MAP: {metrics['MAP']:.4f}")
    print(f"NDCG@{top_k}: {metrics['NDCG@k']:.4f}")
    print("Retrieved Answers:")
    for answer in metrics["Retrieved Answers"]:
        print(f"  {answer}")
    print("-" * 50)
