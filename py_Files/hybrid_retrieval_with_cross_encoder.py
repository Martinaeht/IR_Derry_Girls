#hybrid_retrieval_with_cross_encoder reranking
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
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics import label_ranking_average_precision_score

k_faiss = 16  
k_bm25 = 16  
k_hybrid = 20 
k_final = 8  

model_name = "all-mpnet-base-v2"
cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"  
faiss_index_path = "/home/mlt_ml3/IR_Derry_Girls/Dataset/derry_girls_faiss_new.index"
faiss_metadata_path = "/home/mlt_ml3/IR_Derry_Girls/Dataset/derry_girls_metadata_new.json"
bm25_index_path = "/home/mlt_ml3/IR_Derry_Girls/Dataset/bm25_full_index_stemmed.pkl"

class CrossEncoderReranker:
    def __init__(self, model_name=cross_encoder_model):
        print(f"Loading cross-encoder model: {model_name}")
        self.model = CrossEncoder(model_name)
        print("Cross-encoder model loaded successfully!")
    
    def rerank(self, query, candidates, metadata, top_k=k_final):
        if not candidates:
            return []
    
        query_doc_pairs = []
        doc_ids = []
        
        for score, doc_id in candidates:
            doc = metadata[doc_id]
            doc_text = doc.get('clean_text', '')

            if doc.get('speaker'):
                enhanced_text = f"{doc['speaker']}: {doc_text}"
            else:
                enhanced_text = doc_text
            
            query_doc_pairs.append([query, enhanced_text])
            doc_ids.append(doc_id)
        
        # Get cross-encoder scores
        print(f"Cross-encoder reranking {len(candidates)} candidates...")
        cross_scores = self.model.predict(query_doc_pairs)
        
        # Combine with doc_ids and sort by cross-encoder score
        reranked = list(zip(cross_scores, doc_ids))
        reranked.sort(key=lambda x: x[0], reverse=True)
        
        return reranked[:top_k]

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
        speaker = doc.get("speaker") or "NARRATION"
        
        context_before = ""
        context_after = ""
  
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
        
        print(f"{rank}. [Cross-Encoder Score: {score:.4f}] [ID: {doc_id}] Scene: {scene}")
       
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
    return [(scores[i], i) for i in top_k]

def faiss_search(query, index, model, metadata, k=k_faiss):
    query_embedding = model.encode([normalize_text(query)])
    scores, indices = index.search(np.array(query_embedding).astype('float32'), k)
    similarities = [(1.0 / (1.0 + scores[0][i]), int(indices[0][i])) for i in range(k)]
    return similarities

def hybrid_search_with_reranking(query, bm25, docs, faiss_index, model, cross_encoder, 
                                metadata, alpha=0.5, k_candidates=k_hybrid, k_final=k_final):
    """
    Perform hybrid search followed by cross-encoder reranking
    
    Steps:
    1. Get candidates from BM25 and FAISS
    2. Combine using hybrid scoring
    3. Rerank top candidates with cross-encoder
    """
    
    # Step 1: Get more candidates than usual for reranking
    bm25_results = bm25_search(query, bm25, docs, metadata, k=k_bm25)
    faiss_results = faiss_search(query, faiss_index, model, metadata, k=k_faiss)

    # Step 2: Hybrid combination (same as before)
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

    # Get top candidates for reranking
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k_candidates]
    hybrid_candidates = [(score, doc_id) for doc_id, score in ranked]

    # Step 3: Cross-encoder reranking
    reranked_results = cross_encoder.rerank(query, hybrid_candidates, metadata, k_final)

    return reranked_results, hybrid_candidates, bm25_results, faiss_results

# Evaluation functions (same as before)
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

def ndcg_at_k(retrieved, relevant, k):
    def dcg_at_k(retrieved, relevant, k):
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k]):
            rel = 1 if doc_id in relevant else 0
            dcg += rel / np.log2(i + 2)
        return dcg
    
    ideal_retrieved = [1] * min(len(relevant), k) + [0] * (k - min(len(relevant), k))
    ideal_dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_retrieved))
    return dcg_at_k(retrieved, relevant, k) / ideal_dcg if ideal_dcg > 0 else 0.0

def evaluate_with_reranking(queries: dict, bm25, docs, faiss_index, model, cross_encoder, 
                           metadata, alpha=0.5, k=k_final):
    """Evaluate the hybrid + cross-encoder system"""
    precision_scores = []
    ndcg_scores = []
    map_scores = []
    mrr_scores = []
    valid_queries = 0

    for query, relevant_ids in queries.items():
        if not relevant_ids:
            continue
            
        reranked_results, _, _, _ = hybrid_search_with_reranking(
            query, bm25, docs, faiss_index, model, cross_encoder, metadata, alpha
        )
        retrieved_ids = [doc_id for _, doc_id in reranked_results]

        precision = precision_at_k(retrieved_ids, relevant_ids, k)
        ndcg = ndcg_at_k(retrieved_ids, relevant_ids, k)
        ap = average_precision(retrieved_ids, relevant_ids)
        rr = reciprocal_rank(retrieved_ids, relevant_ids)
        
        precision_scores.append(precision)
        ndcg_scores.append(ndcg)
        map_scores.append(ap)
        mrr_scores.append(rr)
        valid_queries += 1

        hits = sum(1 for doc_id in retrieved_ids if doc_id in relevant_ids)
        print(f"Query: '{query}' | Hits: {hits}/{k} | P@{k}: {precision:.3f} | nDCG: {ndcg:.3f}")

    print(f"\n{'='*50} FINAL RESULTS WITH CROSS-ENCODER {'='*50}")
    print(f"Average Precision@{k}: {np.mean(precision_scores):.4f}")
    print(f"Mean nDCG@{k}: {np.mean(ndcg_scores):.4f}")
    print(f"Mean Average Precision: {np.mean(map_scores):.4f}")
    print(f"Mean Reciprocal Rank: {np.mean(mrr_scores):.4f}")

def compare_systems(queries: dict, bm25, docs, faiss_index, model, cross_encoder, metadata):
    """Compare hybrid-only vs hybrid+cross-encoder"""
    print("Comparing Hybrid vs Hybrid+CrossEncoder...")
    
    # Test hybrid only
    def simple_hybrid_search(query, alpha=0.5, k=k_final):
        bm25_results = bm25_search(query, bm25, docs, metadata, k=k_bm25)
        faiss_results = faiss_search(query, faiss_index, model, metadata, k=k_faiss)
        
        combined = defaultdict(float)
        for score, doc_id in bm25_results:
            combined[doc_id] += (1 - alpha) * score
        for score, doc_id in faiss_results:
            combined[doc_id] += alpha * score
            
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
        return [doc_id for doc_id, _ in ranked]
    
    hybrid_scores = []
    reranked_scores = []
    
    for query, relevant_ids in queries.items():
        if not relevant_ids:
            continue
            
        # Hybrid only
        hybrid_retrieved = simple_hybrid_search(query)
        hybrid_precision = precision_at_k(hybrid_retrieved, relevant_ids, k_final)
        hybrid_scores.append(hybrid_precision)
        
        # Hybrid + Cross-encoder
        reranked_results, _, _, _ = hybrid_search_with_reranking(
            query, bm25, docs, faiss_index, model, cross_encoder, metadata
        )
        reranked_retrieved = [doc_id for _, doc_id in reranked_results]
        reranked_precision = precision_at_k(reranked_retrieved, relevant_ids, k_final)
        reranked_scores.append(reranked_precision)
    
    print(f"\nHybrid Only - Average P@{k_final}: {np.mean(hybrid_scores):.4f}")
    print(f"Hybrid + Cross-Encoder - Average P@{k_final}: {np.mean(reranked_scores):.4f}")
    print(f"Improvement: {np.mean(reranked_scores) - np.mean(hybrid_scores):.4f}")

def main():
    # Load data
    with open(bm25_index_path, "rb") as f:
        bm25_data = pickle.load(f)
        bm25 = bm25_data["bm25"]
        docs = bm25_data["docs"]

    faiss_index = faiss.read_index(faiss_index_path)
    with open(faiss_metadata_path, "r", encoding="utf-8") as f:
        faiss_metadata = json.load(f)

    model = SentenceTransformer(model_name)
    cross_encoder = CrossEncoderReranker()

    # Evaluate
    print("Evaluating Hybrid + Cross-Encoder system...")
    evaluate_with_reranking(queries, bm25, docs, faiss_index, model, cross_encoder, faiss_metadata)
    
    # Compare systems
    compare_systems(queries, bm25, docs, faiss_index, model, cross_encoder, faiss_metadata)

    # Interactive search
    while True:
        query = input("\nType your query or 'exit' if you want to quit: ").strip()
        if query.lower() == "exit":
            break
        elif query:
            reranked_results, hybrid_candidates, bm25_results, faiss_results = hybrid_search_with_reranking(
                query, bm25, docs, faiss_index, model, cross_encoder, faiss_metadata
            )
            
            print(f"\n--- BEFORE RERANKING (Top {len(hybrid_candidates)} Hybrid Results) ---")
            for i, (score, doc_id) in enumerate(hybrid_candidates[:5], 1):
                doc = faiss_metadata[doc_id]
                speaker = doc.get("speaker") or "NARRATION"
                print(f"{i}. [Hybrid Score: {score:.4f}] {speaker}: {doc['clean_text'][:100]}...")
            
            print_results_with_context(reranked_results, faiss_metadata, query, 
                                     title="CROSS-ENCODER RERANKED Results")

if __name__ == "__main__":
    main()