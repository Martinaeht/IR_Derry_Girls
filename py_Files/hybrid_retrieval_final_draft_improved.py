#hybrid retrieval improved
import os
import re
import json
import faiss
import pickle
import unicodedata
import numpy as np
from queries import queries
from rank_bm25 import BM25Okapi
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from sklearn.metrics import label_ranking_average_precision_score

k_faiss = 8
k_bm25 = 8 
k_hybrid = 8

model_name = "all-mpnet-base-v2"
faiss_index_path = "/home/mlt_ml3/IR_Derry_Girls/Dataset/derry_girls_faiss_new.index"
faiss_metadata_path = "/home/mlt_ml3/IR_Derry_Girls/Dataset/derry_girls_metadata_new.json"
bm25_index_path = "/home/mlt_ml3/IR_Derry_Girls/Dataset/bm25_full_index_stemmed.pkl"


def normalize_text(text):
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')  # Remove accents
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def print_results_with_context(results, metadata, query, title="Results"):
    print(f"\n{title} for: '{query}'\n")
    for rank, (score, doc_id) in enumerate(results, 1):
        doc = metadata[doc_id]
        scene = doc.get("scene", "Unknown Scene")
        speaker = doc.get("speaker") or "NARRATION"
        
        # Get context (line before and after)
        context_before = ""
        context_after = ""
        
        # Find lines before and after (if they exist and are in same scene)
        if doc_id > 0 and doc_id - 1 < len(metadata):
            prev_doc = metadata[doc_id - 1]
            if prev_doc.get("scene") == scene and prev_doc.get("clean_text"):
                prev_speaker = prev_doc.get("speaker") or "NARRATION"
                context_before = f"[BEFORE] {prev_speaker}: {prev_doc['clean_text']}"
        
        if doc_id + 1 < len(metadata):
            next_doc = metadata[doc_id + 1]
            if next_doc.get("scene") == scene and next_doc.get("clean_text"):
                next_speaker = next_doc.get("speaker") or "NARRATION"
                context_after = f"[AFTER] {next_speaker}: {next_doc['clean_text']}"
        
        print(f"{rank}. [Score: {score:.4f}] [ID: {doc_id}] Scene: {scene}")
        
        # Print context
        if context_before:
            print(f"   {context_before}")
        
        print(f"   >>> {speaker}: {doc['clean_text']}")
        
        if context_after:
            print(f"   {context_after}")
            
        if doc.get("actions"):
            print(f"   (Actions: {', '.join(doc['actions'])})")
        print("-" * 70)

def bm25_search(query, bm25, docs, metadata, k=k_bm25):
    tokenized_query = normalize_text(query).split()
    scores = bm25.get_scores(tokenized_query)
    top_k = np.argsort(scores)[::-1][:k]
    print(f"[bm25_search] Query: {query}")

    return [(scores[i], i) for i in top_k]

def faiss_search(query, index, model, metadata, k=k_faiss):
    query_embedding = model.encode([normalize_text(query)])
    scores, indices = index.search(np.array(query_embedding).astype('float32'), k)
    
    # Convert L2 distances to similarities (lower distance = higher similarity)
    similarities = [(1.0 / (1.0 + scores[0][i]), int(indices[0][i])) for i in range(k)]
    
    print(f"[faiss_search] Query: {query}")
    print("Raw distances:", scores[0][:3])
    print("Converted similarities:", [s[0] for s in similarities[:3]])
    
    return similarities

def improved_hybrid_search(query, bm25, docs, faiss_index, model, metadata, alpha=0.5, k=8):
    # Get more candidates for better fusion
    bm25_results = bm25_search(query, bm25, docs, metadata, k=k_bm25*2)  # Get more candidates
    faiss_results = faiss_search(query, faiss_index, model, metadata, k=k_faiss*2)

    # Extract scores and doc IDs
    bm25_doc_ids = [doc_id for _, doc_id in bm25_results]
    bm25_scores = np.array([score for score, _ in bm25_results]).reshape(-1, 1)

    faiss_doc_ids = [doc_id for _, doc_id in faiss_results]
    faiss_scores = np.array([score for score, _ in faiss_results]).reshape(-1, 1)

    # Improved normalization - use rank-based scoring for more stable results
    def rank_normalize(scores):
        """Convert scores to rank-based normalization (1.0 for best, decreasing)"""
        ranks = np.argsort(np.argsort(-scores.flatten())) + 1  # Higher score = lower rank number
        return 1.0 / ranks  # Convert to similarity (higher = better)

    # Apply rank normalization
    bm25_norm_scores = rank_normalize(bm25_scores)
    faiss_norm_scores = rank_normalize(faiss_scores)

    # Combine with reciprocal rank fusion approach
    combined = defaultdict(float)
    
    # BM25 contribution
    for doc_id, score in zip(bm25_doc_ids, bm25_norm_scores):
        combined[doc_id] += (1 - alpha) * score
    
    # FAISS contribution  
    for doc_id, score in zip(faiss_doc_ids, faiss_norm_scores):
        combined[doc_id] += alpha * score

    # Final ranking
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
    hybrid_results = [(score, doc_id) for doc_id, score in ranked]

    return hybrid_results, bm25_results[:k_bm25], faiss_results[:k_faiss]

def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    return sum(1 for doc_id in retrieved_k if doc_id in relevant) / k

def reciprocal_rank(retrieved, relevant):
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1 / (i + 1)
    return 0.0

def average_precision(retrieved, relevant):
    hits, score = 0, 0.0
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / len(relevant) if relevant else 0.0

def dcg_at_k(retrieved, relevant, k):
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        rel = 1 if doc_id in relevant else 0
        dcg += rel / np.log2(i + 2)
    return dcg

def ndcg_at_k(retrieved, relevant, k):
    ideal_retrieved = [1] * min(len(relevant), k) + [0] * (k - min(len(relevant), k))
    ideal_dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_retrieved))
    return dcg_at_k(retrieved, relevant, k) / ideal_dcg if ideal_dcg > 0 else 0.0

def enhanced_evaluate_hybrid(queries: dict, bm25, docs, faiss_index, model, metadata, alpha=0.5, k=8):
    total_precision, total_rr, total_ap, total_ndcg = 0, 0, 0, 0
    valid_queries = 0
    
    results_summary = []

    for query, relevant_ids in queries.items():
        if not relevant_ids:
            continue
        
        hybrid_results, bm25_results, faiss_results = improved_hybrid_search(
            query, bm25, docs, faiss_index, model, metadata, alpha, k
        )
        retrieved_ids = [doc_id for _, doc_id in hybrid_results]

        # Calculate metrics
        precision = precision_at_k(retrieved_ids, relevant_ids, k)
        rr = reciprocal_rank(retrieved_ids, relevant_ids)
        ap = average_precision(retrieved_ids, relevant_ids)
        ndcg = ndcg_at_k(retrieved_ids, relevant_ids, k)
        
        # Count hits
        hits = sum(1 for doc_id in retrieved_ids if doc_id in relevant_ids)
        
        total_precision += precision
        total_rr += rr
        total_ap += ap
        total_ndcg += ndcg
        valid_queries += 1

        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"Relevant IDs: {relevant_ids}")
        print(f"Retrieved IDs: {retrieved_ids}")
        print(f"Hits: {hits}/{k} | P@{k}: {precision:.3f} | MRR: {rr:.3f} | AP: {ap:.3f} | nDCG@{k}: {ndcg:.3f}")
        
        results_summary.append({
            'query': query,
            'hits': hits,
            'precision': precision,
            'mrr': rr,
            'ap': ap,
            'ndcg': ndcg
        })

        print_results_with_context(hybrid_results, metadata, query, title="Hybrid Results with Context")
        print("=" * 80)

    if valid_queries > 0:
        print(f"\n{'='*50} FINAL RESULTS {'='*50}")
        print(f"Average Precision@{k}: {total_precision / valid_queries:.4f}")
        print(f"Mean Reciprocal Rank: {total_rr / valid_queries:.4f}")
        print(f"Mean Average Precision: {total_ap / valid_queries:.4f}")
        print(f"Mean nDCG@{k}: {total_ndcg / valid_queries:.4f}")
        
        # Show best and worst performing queries
        results_summary.sort(key=lambda x: x['precision'], reverse=True)
        print(f"\nBest performing query (P@{k}={results_summary[0]['precision']:.3f}): {results_summary[0]['query']}")
        print(f"Worst performing query (P@{k}={results_summary[-1]['precision']:.3f}): {results_summary[-1]['query']}")
    else:
        print("No valid queries found.")

def tune_hybrid_parameters(queries, bm25, docs, faiss_index, model, metadata, k=8):
    """Test different alpha values to find optimal hybrid combination"""
    alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_alpha = 0.5
    best_map = 0.0
    
    print("Tuning hybrid parameters...")
    print(f"{'Alpha':<8} {'MAP':<8} {'P@8':<8} {'MRR':<8} {'nDCG@8':<8}")
    print("-" * 40)
    
    for alpha in alpha_values:
        total_map = total_precision = total_mrr = total_ndcg = 0
        valid_queries = 0
        
        for query, relevant_ids in queries.items():
            if not relevant_ids:
                continue
                
            hybrid_results, _, _ = improved_hybrid_search(
                query, bm25, docs, faiss_index, model, metadata, alpha, k
            )
            retrieved_ids = [doc_id for _, doc_id in hybrid_results]
            
            total_map += average_precision(retrieved_ids, relevant_ids)
            total_precision += precision_at_k(retrieved_ids, relevant_ids, k)
            total_mrr += reciprocal_rank(retrieved_ids, relevant_ids)
            total_ndcg += ndcg_at_k(retrieved_ids, relevant_ids, k)
            valid_queries += 1
        
        if valid_queries > 0:
            avg_map = total_map / valid_queries
            avg_precision = total_precision / valid_queries
            avg_mrr = total_mrr / valid_queries
            avg_ndcg = total_ndcg / valid_queries
            
            print(f"{alpha:<8.1f} {avg_map:<8.3f} {avg_precision:<8.3f} {avg_mrr:<8.3f} {avg_ndcg:<8.3f}")
            
            if avg_map > best_map:
                best_map = avg_map
                best_alpha = alpha
    
    print(f"\nBest alpha: {best_alpha} (MAP: {best_map:.3f})")
    return best_alpha

def main():
    with open(bm25_index_path, "rb") as f:
        bm25_data = pickle.load(f)
        bm25 = bm25_data["bm25"]
        docs = bm25_data["docs"]
        bm25_metadata = bm25_data["metadata"]

    # Lade FAISS
    faiss_index = faiss.read_index(faiss_index_path)
    with open(faiss_metadata_path, "r", encoding="utf-8") as f:
        faiss_metadata = json.load(f)

    model = SentenceTransformer(model_name)

    # Tune parameters
    optimal_alpha = tune_hybrid_parameters(queries, bm25, docs, faiss_index, model, faiss_metadata)
    
    # Run evaluation with optimal parameters
    enhanced_evaluate_hybrid(queries, bm25, docs, faiss_index, model, faiss_metadata, alpha=optimal_alpha)
    
    # Interactive search with context
    while True:
        query = input("\nType your query or 'exit' if you want to quit: ").strip()
        if query.lower() == "exit":
            break
        elif query:
            hybrid_results, bm25_results, faiss_results = improved_hybrid_search(
                query, bm25, docs, faiss_index, model, faiss_metadata, alpha=optimal_alpha
            )
            print_results_with_context(hybrid_results, faiss_metadata, query, title="Hybrid Search Results with Context")

if __name__ == "__main__":
    main()
