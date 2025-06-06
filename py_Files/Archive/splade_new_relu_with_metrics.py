import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from splade_queries import queries

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

# Corrected SPLADE encoding function
def encode_splade(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        # Proper SPLADE encoding: ReLU + log transformation
        logits = outputs.logits
        relu_log = torch.log(1 + torch.relu(logits))
        # Mask out padding tokens
        weighted_log = relu_log * inputs.attention_mask.unsqueeze(-1)
        # Sum over sequence length to get sparse document representation
        sparse_vectors = torch.sum(weighted_log, dim=1)
    return sparse_vectors.cpu().numpy()

# Improved search function with debugging
def search(query, index, metadata, top_k=5):
    normalized_query = normalize_text(query)
    print(f"Searching for: '{normalized_query}'")
    
    query_vector = encode_splade([normalized_query])
    print(f"Query vector shape: {query_vector.shape}")
    print(f"Index shape: {index.shape}")
    
    # Convert sparse matrix to dense if needed
    if hasattr(index, 'toarray'):
        index_dense = index.toarray()
    else:
        index_dense = index
    
    # Ensure dimensions match
    if query_vector.shape[1] != index_dense.shape[1]:
        print(f"Dimension mismatch! Query: {query_vector.shape[1]}, Index: {index_dense.shape[1]}")
        # Pad or truncate to match
        min_dim = min(query_vector.shape[1], index_dense.shape[1])
        query_vector = query_vector[:, :min_dim]
        index_dense = index_dense[:, :min_dim]
    
    similarities = cosine_similarity(query_vector, index_dense).flatten()
    print(f"Similarity range: [{similarities.min():.4f}, {similarities.max():.4f}]")
    
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = [(metadata[i], similarities[i]) for i in top_indices]
    
    # Debug: print top results
    print("Top results:")
    for i, (meta, sim) in enumerate(results[:3]):
        print(f"  {i+1}. ID: {meta.get('id', 'N/A')}, Score: {sim:.4f}")
    
    return results

# Enhanced evaluation function with multiple metrics
def evaluate(queries, index, metadata, top_k=5):
    results = {}
    total_recall = 0
    total_precision = 0
    total_mrr = 0
    total_map = 0
    total_ndcg = 0
    total_accuracy = 0
    
    for i, (query, expected_indices) in enumerate(queries.items()):
        print(f"\n=== Query {i+1}/{len(queries)} ===")
        print(f"Query: {query}")
        print(f"Expected IDs: {expected_indices}")
        
        search_results = search(query, index, metadata, top_k=top_k)
        retrieved_ids = [meta["id"] for meta, _ in search_results]
        similarities = [sim for _, sim in search_results]
        
        print(f"Retrieved IDs: {retrieved_ids}")
        
        # Calculate Recall@k
        relevant_retrieved = [rid for rid in retrieved_ids if rid in expected_indices]
        recall_at_k = len(relevant_retrieved) / len(expected_indices) if expected_indices else 0
        
        # Calculate Precision@k
        precision_at_k = len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0
        
        # Calculate Accuracy (binary: found at least one relevant doc)
        accuracy = 1 if any(rid in expected_indices for rid in retrieved_ids) else 0
        
        # Calculate MRR
        mrr = 0
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in expected_indices:
                mrr = 1 / rank
                print(f"Found relevant document at rank {rank}")
                break
        
        # Calculate MAP (Mean Average Precision)
        avg_precision = 0
        relevant_count = 0
        precision_sum = 0
        
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in expected_indices:
                relevant_count += 1
                precision_at_rank = relevant_count / rank
                precision_sum += precision_at_rank
        
        avg_precision = precision_sum / len(expected_indices) if expected_indices else 0
        
        # Calculate NDCG@k
        def calculate_dcg(relevance_scores):
            dcg = 0
            for i, rel in enumerate(relevance_scores):
                if i == 0:
                    dcg += rel
                else:
                    dcg += rel / np.log2(i + 1)
            return dcg
        
        # Create relevance scores (1 if relevant, 0 if not)
        relevance_scores = [1 if rid in expected_indices else 0 for rid in retrieved_ids]
        
        # Calculate DCG
        dcg = calculate_dcg(relevance_scores)
        
        # Calculate IDCG (Ideal DCG)
        ideal_relevance = [1] * min(len(expected_indices), top_k) + [0] * max(0, top_k - len(expected_indices))
        idcg = calculate_dcg(ideal_relevance)
        
        # Calculate NDCG
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
    
    # Print overall statistics
    num_queries = len(queries)
    avg_recall = total_recall / num_queries
    avg_precision = total_precision / num_queries
    avg_accuracy = total_accuracy / num_queries
    avg_mrr = total_mrr / num_queries
    avg_map = total_map / num_queries
    avg_ndcg = total_ndcg / num_queries
    
    print(f"\n=== OVERALL RESULTS ===")
    print(f"Average Recall@{top_k}: {avg_recall:.4f}")
    print(f"Average Precision@{top_k}: {avg_precision:.4f}")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average MRR: {avg_mrr:.4f}")
    print(f"Average MAP: {avg_map:.4f}")
    print(f"Average NDCG@{top_k}: {avg_ndcg:.4f}")
    
    return results

# Data type verification
def verify_data_types(queries, metadata):
    print("=== DATA VERIFICATION ===")
    print(f"Number of queries: {len(queries)}")
    print(f"Number of documents in metadata: {len(metadata)}")
    
    # Check query format
    sample_query = list(queries.items())[0]
    print(f"Sample query: {sample_query[0]}")
    print(f"Expected IDs type: {type(sample_query[1])}")
    print(f"Expected IDs: {sample_query[1]}")
    
    # Check metadata format
    print(f"Sample metadata keys: {list(metadata[0].keys()) if metadata else 'No metadata'}")
    if metadata:
        print(f"Sample document ID: {metadata[0].get('id', 'No ID field')}")
    
    # Check for ID overlap
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
index_path = "/home/mlt_ml3/IR_Derry_Girls/py_Files/SPLADE/splade_index.npz"
metadata_path = "/home/mlt_ml3/IR_Derry_Girls/py_Files/SPLADE/metadata.json"

index = load_npz(index_path)
with open(metadata_path, "r") as f:
    metadata = json.load(f)

# Verify data before evaluation
verify_data_types(queries, metadata)

# Run evaluation
evaluation_results = evaluate(queries, index, metadata, top_k=5)

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