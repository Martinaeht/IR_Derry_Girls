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

#To make sure that my indices are aligned, I compared a few documents from both indices.
"""with open(bm25_index_path, "rb") as f:
    bm25_data = pickle.load(f)
    bm25_metadata = bm25_data["metadata"]

with open(faiss_metadata_path, "r", encoding="utf-8") as f:
    faiss_metadata = json.load(f)

test_ids = [0, 10, 50, 100]  # Beispiel-IDs

for doc_id in test_ids:
    bm25_doc = bm25_metadata[doc_id] if isinstance(bm25_metadata, list) else bm25_metadata[str(doc_id)]
    faiss_doc = faiss_metadata[doc_id] if isinstance(faiss_metadata, list) else faiss_metadata[str(doc_id)]
    
    print(f"DocID {doc_id}:")
    print("BM25:", bm25_doc.get("clean_text")[:100])
    print("FAISS:", faiss_doc.get("clean_text")[:100])
    print("---")
"""

def normalize_text(text):
    """Normalize text by removing accents, converting to lowercase, and removing special characters."""
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')  # Remove accents
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def print_results(results, metadata, query, title="Results"):
    print(f"\n{title} for: '{query}'\n")
    for rank, (score, doc_id) in enumerate(results, 1):
        doc = metadata[doc_id]
        scene = doc.get("scene", "Unknown Scene")
        speaker = doc.get("speaker") or "NARRATION"
        print(f"{rank}. [Score: {score:.4f}] Scene: {scene}")
        print(f"{speaker}: {doc['clean_text']}")
        if doc.get("actions"):
            print(f"   (Actions: {', '.join(doc['actions'])})")
        print("-" * 50)

def bm25_search(query, bm25, docs, metadata, k=k_bm25):
    tokenized_query = normalize_text(query).split()
    scores = bm25.get_scores(tokenized_query)
    top_k = np.argsort(scores)[::-1][:k]
    print(f"[bm25_search] Query: {query}")

    return [(scores[i], i) for i in top_k]

# === FAISS-Suche ===
"""def faiss_search(query, index, model, metadata, k=k_faiss):
    query_embedding = model.encode([normalize_text(query)])
    scores, indices = index.search(np.array(query_embedding).astype('float32'), k)
    print(f"[faiss_search] Query: {query}")
    print("Query embedding:", query_embedding[:5])  # first 5 dims
    print("Scores:", scores)
    #return [(1 - (scores[0][i] / 100), int(indices[0][i])) for i in range(k)]
    return [(scores[0][i], int(indices[0][i])) for i in range(k)]
"""
def faiss_search(query, index, model, metadata, k=k_faiss):
    query_embedding = model.encode([normalize_text(query)])
    scores, indices = index.search(np.array(query_embedding).astype('float32'), k)
    
    # Convert L2 distances to similarities (lower distance = higher similarity)
    similarities = [(1.0 / (1.0 + scores[0][i]), int(indices[0][i])) for i in range(k)]
    
    print(f"[faiss_search] Query: {query}")
    print("Raw distances:", scores[0][:3])
    print("Converted similarities:", [s[0] for s in similarities[:3]])
    
    return similarities

def hybrid_search(query, bm25, docs, faiss_index, model, metadata, alpha=0.7, k=k_hybrid): #alpha was 0.5 at first meanng euqal weight for bm25 and faiss, but we want to give more weight to faiss, so I set it to 0.7
    total_precision, total_rr, total_ap, total_ndcg = 0, 0, 0, 0
    bm25_results = bm25_search(query, bm25, docs, metadata, k=k_bm25)
    faiss_results = faiss_search(query, faiss_index, model, metadata, k=k_faiss)

    # --- 1. Extrahiere Scores und DocIDs
    bm25_doc_ids = [doc_id for _, doc_id in bm25_results]
    bm25_scores = np.array([score for score, _ in bm25_results]).reshape(-1, 1)

    faiss_doc_ids = [doc_id for _, doc_id in faiss_results]
    faiss_scores = np.array([score for score, _ in faiss_results]).reshape(-1, 1)

    # --- 2. Normiere Scores
    if len(bm25_scores) > 1:
        bm25_scores = MinMaxScaler().fit_transform(bm25_scores).flatten()
    else:
        bm25_scores = bm25_scores.flatten()

    if len(faiss_scores) > 1:
        faiss_scores = MinMaxScaler().fit_transform(faiss_scores).flatten()
    else:
        faiss_scores = faiss_scores.flatten()

    # --- 3. Kombinieren
    combined = defaultdict(float)
    for doc_id, score in zip(bm25_doc_ids, bm25_scores):
        combined[doc_id] += (1 - alpha) * score
    for doc_id, score in zip(faiss_doc_ids, faiss_scores):
        combined[doc_id] += alpha * score

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
    hybrid_results = [(score, doc_id) for doc_id, score in ranked]

    # Debug-Ausgabe
    print(f"[Hybrid Search] Query: '{query}'")
    print("Top Hybrid DocIDs:", [doc_id for _, doc_id in hybrid_results])

    return hybrid_results, bm25_results, faiss_results

"""def hybrid_search(query, bm25, docs, faiss_index, model, metadata, alpha=0.5, k=k_hybrid):
    bm25_results = bm25_search(query, bm25, docs, metadata, k=k_bm25)
    faiss_results = faiss_search(query, faiss_index, model, metadata, k=k_faiss)

    combined = defaultdict(float)
    for score, doc_id in bm25_results:
        combined[doc_id] += (1 - alpha) * score
    for score, doc_id in faiss_results:
        combined[doc_id] += alpha * score

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
    #return [(score, doc_id) for doc_id, score in ranked]
    hybrid_results = [(score, doc_id) for doc_id, score in ranked]

    #debugprint
    print("Returning from hybrid_search:")
    print("Hybrid:", type(hybrid_results), len(hybrid_results))
    print("BM25:", type(bm25_results), len(bm25_results))
    print("FAISS:", type(faiss_results), len(faiss_results))
    return hybrid_results, bm25_results, faiss_results
"""

def print_hybrid_comparison(query, hybrid_results, bm25_results, faiss_results, metadata):
    print(f"\nHybrid Search Results for: '{query}'\n")
    print(f"{'Rank':<5}{'Score':<10}{'DocID':<6} | {'BM25 Score':<10}{'Faiss Score':<10} | Text Snippet")
    print("-" * 80)
    
    # Hilfsdict für schnellen Score-Zugriff
    bm25_dict = dict(bm25_results)
    faiss_dict = dict(faiss_results)
    
    for rank, (score, doc_id) in enumerate(hybrid_results, 1):
        bm25_score = bm25_dict.get(doc_id, 0)
        faiss_score = faiss_dict.get(doc_id, 0)
        #doc = metadata[str(doc_id)] if isinstance(doc_id, int) else metadata[doc_id]
        doc = metadata[doc_id]
        snippet = doc["clean_text"][:60].replace("\n", " ") + "..."
        print(f"{rank:<5}{score:<10.4f}{doc_id:<6} | {bm25_score:<10.4f}{faiss_score:<10.4f} | {snippet}")

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

def evaluate_hybrid(queries: dict, bm25, docs, faiss_index, model, metadata, alpha=0.7,  k=8): #alpha was 0.5 at first meanng euqal weight for bm25 and faiss, but we want to give more weight to faiss, so I set it to 0.7
    total_precision, total_rr, total_ap, total_ndcg = 0, 0, 0, 0
    valid_queries = 0

    for query, relevant_ids in queries.items():
        if not relevant_ids:
            continue
        
        #normalized_query = normalize_text(query)
        normalized_query = normalize_text(query)
        print(f"Original query: '{query}'")
        print(f"Normalized query: '{normalized_query}'")
        #results = hybrid_search(normalized_query, bm25, docs, faiss_index, model, metadata, alpha)
        hybrid_results, _, _ = hybrid_search(normalized_query, bm25, docs, faiss_index, model, metadata, alpha)
        retrieved_ids = [doc_id for _, doc_id in hybrid_results]

        #if not set(retrieved_ids).intersection(relevant_ids):
         #   continue

        print(f"Query: {query}")
        print(f"Relevant IDs: {relevant_ids}")
        print(f"Retrieved IDs: {retrieved_ids}")

        total_precision += precision_at_k(retrieved_ids, relevant_ids, k)
        total_rr += reciprocal_rank(retrieved_ids, relevant_ids)
        total_ap += average_precision(retrieved_ids, relevant_ids)
        total_ndcg += ndcg_at_k(retrieved_ids, relevant_ids, k)
        valid_queries += 1

        print_results(hybrid_results, metadata, query, title="Hybrid Results")
        print(f"Precision@{k}: {total_precision/valid_queries:.4f}, MRR: {total_rr/valid_queries:.4f}, AP: {total_ap/valid_queries:.4f}, nDCG@{k}: {total_ndcg/valid_queries:.4f}")
        print("=" * 60)

    if valid_queries > 0:
        print("\nEvaluation Results:")
        print(f"Average Precision@{k}: {total_precision / valid_queries:.4f}")
        print(f"Mean Reciprocal Rank: {total_rr / valid_queries:.4f}")
        print(f"Mean Average Precision: {total_ap / valid_queries:.4f}")
        print(f"Mean nDCG@{k}: {total_ndcg / valid_queries:.4f}")
    else:
        print("No valid queries found.")

"""def evaluate_hybrid(queries, relevance_data, bm25, docs, faiss_index, model, metadata, alpha=0.5):
    mrr_total, map_total, ndcg_total = 0, 0, 0

    for query in queries:
        relevant_ids = relevance_data.get(query, [])
        if not relevant_ids:
            continue

        results = hybrid_search(query, bm25, docs, faiss_index, model, metadata, alpha)
        retrieved_ids = [doc_id for _, doc_id in results]

        y_true = [1 if doc_id in relevant_ids else 0 for doc_id in retrieved_ids]
        y_scores = [score for score, _ in results]

        if not any(y_true):
            continue

        # Metrics
        mrr = 1.0 / (y_true.index(1) + 1) if 1 in y_true else 0
        map_ = label_ranking_average_precision_score([y_true], [y_scores])
        dcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(y_true)])
        idcg = sum([1.0 / np.log2(idx + 2) for idx in range(min(len(relevant_ids), len(y_true)))])
        ndcg = dcg / idcg if idcg > 0 else 0

        mrr_total += mrr
        map_total += map_
        ndcg_total += ndcg

        print_results(results, metadata, query, title="Hybrid Results")
        print(f"MRR: {mrr:.4f}, MAP: {map_:.4f}, nDCG: {ndcg:.4f}")
        print("=" * 60)

    count = len(queries)
    print("\nEvaluation Results:")
    print(f"MRR: {mrr_total / count:.4f}")
    print(f"MAP: {map_total / count:.4f}")
    print(f"nDCG: {ndcg_total / count:.4f}")
"""
# === Hauptfunktion ===
def main():
    # Lade BM25
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

    evaluate_hybrid(queries, bm25, docs, faiss_index, model, faiss_metadata)

    # Benutzerabfrage
    while True:
        query = input("\nType your query or 'exit' if you want to quit: ").strip()
        if query.lower() == "exit":
            break
        elif query:
            #results = hybrid_search(query, bm25, docs, faiss_index, model, faiss_metadata)
            #print_results(results, faiss_metadata, query, title="Hybrid Search Results")
            hybrid_results, bm25_results, faiss_results = hybrid_search(query, bm25, docs, faiss_index, model, faiss_metadata)
            print_hybrid_comparison(query, hybrid_results, bm25_results, faiss_results, faiss_metadata)

if __name__ == "__main__":
    main()
