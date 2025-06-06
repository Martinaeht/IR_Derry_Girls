import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from queries_splade import queries

# Load SPLADE model and tokenizer
model_name = "naver/splade-cocondenser-ensembledistil"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Normalize text
def normalize_text(text):
    import re
    text = text.lower()
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# SPLADE encoding
def encode_splade(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        relu_log = torch.log(1 + torch.relu(logits))
        weighted_log = relu_log * inputs.attention_mask.unsqueeze(-1)
        sparse_vectors = torch.sum(weighted_log, dim=1)
    return sparse_vectors.cpu().numpy()

# Search function with answer printing
def search(query, index, metadata, top_k=5):
    normalized_query = normalize_text(query)
    print(f"Searching for: '{normalized_query}'")
    
    query_vector = encode_splade([normalized_query])
    print(f"Query vector shape: {query_vector.shape}")
    print(f"Index shape: {index.shape}")
    
    if hasattr(index, 'toarray'):
        index_dense = index.toarray()
    else:
        index_dense = index
    
    if query_vector.shape[1] != index_dense.shape[1]:
        print(f"Dimension mismatch! Query: {query_vector.shape[1]}, Index: {index_dense.shape[1]}")
        min_dim = min(query_vector.shape[1], index_dense.shape[1])
        query_vector = query_vector[:, :min_dim]
        index_dense = index_dense[:, :min_dim]
    
    similarities = cosine_similarity(query_vector, index_dense).flatten()
    print(f"Similarity range: [{similarities.min():.4f}, {similarities.max():.4f}]")
    
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = [(metadata[i], similarities[i]) for i in top_indices]
    
    print("Top results:")
    for i, (meta, sim) in enumerate(results):
        print(f"  {i+1}. ID: {meta.get('id', 'N/A')}, Score: {sim:.4f}")
        print(f"     Answer: {meta.get('answer', '[No answer field]')}")
    
    return results

# Evaluation function with answer printing
def evaluate(queries, index, metadata, top_k=5):
    results = {}
    total_recall = total_precision = total_mrr = total_map = total_ndcg = total_accuracy = 0
    
    for i, (query, expected_indices) in enumerate(queries.items()):
        print(f"\n=== Query {i+1}/{len(queries)} ===")
        print(f"Query: {query}")
        print(f"Expected IDs: {expected_indices}")
        
        search_results = search(query, index, metadata, top_k=top_k)
        retrieved_ids = [meta["id"] for meta, _ in search_results]
        similarities = [sim for _, sim in search_results]
        
        print("Retrieved Answers:")
        for j, (meta, sim) in enumerate(search_results):
            print(f"  {j+1}. ID: {meta.get('id', 'N/A')}, Score: {sim:.4f}")
            print(f"     Answer: {meta.get('answer', '[No answer field]')}")
        
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
            "NDCG@k": ndcg
        }
        
        total_recall += recall_at_k
        total_precision += precision_at_k
        total_accuracy += accuracy
        total_mrr += mrr
        total_map += avg_precision
        total_ndcg += ndcg
        
        print(f"Recall@{top_k}: {recall_at_k:.4f}")
        print(f"Precision@{top_k}: {precision_at_k:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"MRR: {mrr:.4f}")
        print(f"MAP: {avg_precision:.4f}")
        print(f"NDCG@{top_k}: {ndcg:.4f}")
    
    num_queries = len(queries)
    print(f"\n=== OVERALL RESULTS ===")
    print(f"Average Recall@{top_k}: {total_recall / num_queries:.4f}")
    print(f"Average Precision@{top_k}: {total_precision / num_queries:.4f}")
    print(f"Average Accuracy: {total_accuracy / num_queries:.4f}")
    print(f"Average MRR: {total_mrr / num_queries:.4f}")
    print(f"Average MAP: {total_map / num_queries:.4f}")
    print(f"Average NDCG@{top_k}: {total_ndcg / num_queries:.4f}")
    
    return results

# Data verification
def verify_data_types(queries, metadata):
    print("=== DATA VERIFICATION ===")
    print(f"Number of queries: {len(queries)}")
    print(f"Number of documents in metadata: {len(metadata)}")
    
    sample_query = list(queries.items())[0]
    print(f"Sample query: {sample_query[0]}")
    print(f"Expected IDs type: {type(sample_query[1])}")
    print(f"Expected IDs: {sample_query[1]}")
    
    print(f"Sample metadata keys: {list(metadata[0].keys()) if metadata else 'No metadata'}")
    if metadata:
        print(f"Sample document ID: {metadata[0].get('id', 'No ID field')}")
    
    all_doc_ids = set(meta["id"] for meta in metadata if "id" in meta)
    all_expected_ids = set()
    for expected_list in queries.values():
        all_expected_ids.update(expected_list)
    
    overlap = all_doc_ids.intersection(all_expected_ids)
    print(f"Document IDs in metadata: {len(all_doc_ids)}")
    print(f"Expected IDs in queries: {len(all_expected_ids)}")
    print(f"ID overlap: {len(overlap)}")
    
    if len(overlap) == 0:
        print("WARNING: No overlap between expected IDs and document IDs!")
        print(f"Sample document IDs: {list(all_doc_ids)[:5]}")
        print(f"Sample expected IDs: {list(all_expected_ids)[:5]}")

# Load index and metadata
index_path = "/home/mlt_ml3/IR_Derry_Girls/py_Files/SPLADE/splade_index_relu.npz"
metadata_path = "/home/mlt_ml3/IR_Derry_Girls/py_Files/SPLADE/metadata_relu.json"

index = load_npz(index_path)
with open(metadata_path, "r") as f:
    metadata = json.load(f)

'''
# Example queries dictionary (replace with your actual data)
queries = {
    "What is the name of the school?": ["doc1", "doc3"],
    "Who is the headmistress?": ["doc2"]
}
'''

# Run verification and evaluation
verify_data_types(queries, metadata)

# Set top_k for evaluation
top_k = 5

# Run evaluation
evaluation_results = evaluate(queries, index, metadata, top_k=top_k)

# Print final results per query
print("\n=== FINAL RESULTS ===")
for query, metrics in evaluation_results.items():
    print(f"Query: {query}")
    print(f"Recall@{top_k}: {metrics['Recall@k']:.4f}")
    print(f"Precision@{top_k}: {metrics['Precision@k']:.4f}")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"MRR: {metrics['MRR']:.4f}")
    print(f"MAP: {metrics['MAP']:.4f}")
    print(f"NDCG@{top_k}: {metrics['NDCG@k']:.4f}")
    print("-" * 50)
