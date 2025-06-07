#hybrid retrieval with more printed context lines
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
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn') 
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def print_results_with_context(results, metadata, query, title="Results"):
    print(f"\n{title} for: '{query}'\n")
    for rank, (score, doc_id) in enumerate(results, 1):
        doc = metadata[doc_id]
        scene = doc.get("scene", "Unknown Scene")
        speaker = doc.get("speaker") 
        
        context_before = ""
        context_after = ""
  
        if doc_id > 0 and doc_id - 1 < len(metadata):
            prev_doc = metadata[doc_id - 1]
            if prev_doc.get("scene") == scene and prev_doc.get("clean_text"):
                prev_speaker = prev_doc.get("speaker") 
                context_before = f"[Previous line:] {prev_speaker}: {prev_doc['clean_text']}"
        
        if doc_id + 1 < len(metadata):
            next_doc = metadata[doc_id + 1]
            if next_doc.get("scene") == scene and next_doc.get("clean_text"):
                next_speaker = next_doc.get("speaker")
                context_after = f"[Next line:] {next_speaker}: {next_doc['clean_text']}"
        
        print(f"{rank}. [Score: {score:.4f}] [ID: {doc_id}] Scene: {scene}")
       
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

    similarities = [(1.0 / (1.0 + scores[0][i]), int(indices[0][i])) for i in range(k)]
    
    print(f"[faiss_search] Query: {query}")
    #print("Raw distances:", scores[0][:3])
    #print("Converted similarities:", [s[0] for s in similarities[:3]])
    
    return similarities

def improved_hybrid_search(query, bm25, docs, faiss_index, model, metadata, alpha=0.5, k=8):
    bm25_results = bm25_search(query, bm25, docs, metadata, k=k_bm25*2)
    faiss_results = faiss_search(query, faiss_index, model, metadata, k=k_faiss*2)

    bm25_doc_ids = [doc_id for _, doc_id in bm25_results]
    bm25_scores = np.array([score for score, _ in bm25_results]).reshape(-1, 1)

    faiss_doc_ids = [doc_id for _, doc_id in faiss_results]
    faiss_scores = np.array([score for score, _ in faiss_results]).reshape(-1, 1)

    def rank_normalize(scores):
        ranks = np.argsort(np.argsort(-scores.flatten())) + 1 
        return 1.0 / ranks  

    bm25_norm_scores = rank_normalize(bm25_scores)
    faiss_norm_scores = rank_normalize(faiss_scores)

    combined = defaultdict(float)

    for doc_id, score in zip(bm25_doc_ids, bm25_norm_scores):
        combined[doc_id] += (1 - alpha) * score
 
    for doc_id, score in zip(faiss_doc_ids, faiss_norm_scores):
        combined[doc_id] += alpha * score

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

def evaluate_hybrid(queries: dict, bm25, docs, faiss_index, model, metadata, alpha=0.5, k_vals=[5, 8]):
    precision_scores = {k: [] for k in k_vals}
    ndcg_scores = {k: [] for k in k_vals}
    map_scores = []
    mrr_scores = []

    for query, relevant_ids in queries.items():
        hybrid_results, _, _ = improved_hybrid_search(query, bm25, docs, faiss_index, model, metadata, alpha, max(k_vals))
        retrieved_ids = [doc_id for _, doc_id in hybrid_results]
        
        for k in k_vals:
            precision = precision_at_k(retrieved_ids, relevant_ids, k)
            ndcg = ndcg_at_k(retrieved_ids, relevant_ids, k)
            precision_scores[k].append(precision)
            ndcg_scores[k].append(ndcg)

        map_score = (average_precision(retrieved_ids, relevant_ids))
        mrr_score = (reciprocal_rank(retrieved_ids, relevant_ids))
        map_scores.append(map_score)
        mrr_scores.append(mrr_score)
#
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"Relevant IDs: {relevant_ids}")
        print(f"Retrieved IDs: {retrieved_ids}")
        for k in k_vals:
            hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
            current_precision = precision_scores[k][-1]  
            current_ndcg = ndcg_scores[k][-1] 
            print(f"Hits@{k}: {hits}/{k}")
            print(f"Precision@{k}: {current_precision:.3f}")
            print(f"nDCG@{k}: {current_ndcg:.3f}")
        
        print(f"MRR: {mrr_score:.3f} | MAP: {map_score:.3f}")

    print (f"\nAverage metric scores for queries")
    for k in k_vals:
        print(f"Precision@{k}: {np.mean(precision_scores[k]):.3f}")
        print(f"nDCG@{k}:        {np.mean(ndcg_scores[k]):.3f}")
    print(f"MAP:            {np.mean(map_scores):.3f}")
    print(f"MRR:            {np.mean(mrr_scores):.3f}")

def tune_hybrid_parameters(queries, bm25, docs, faiss_index, model, metadata, k=8):
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

    faiss_index = faiss.read_index(faiss_index_path)
    with open(faiss_metadata_path, "r", encoding="utf-8") as f:
        faiss_metadata = json.load(f)

    model = SentenceTransformer(model_name)

    optimal_alpha = tune_hybrid_parameters(queries, bm25, docs, faiss_index, model, faiss_metadata)

    evaluate_hybrid(queries, bm25, docs, faiss_index, model, faiss_metadata, alpha=optimal_alpha)

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
