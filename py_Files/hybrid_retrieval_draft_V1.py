#noch nicht fertig!!!!!
import pickle
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --------- Load BM25 index and metadata ---------
def load_bm25_index(pickle_path):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    bm25 = data["bm25"]
    docs = data["docs"]       # list of document texts, order matters
    metadata = data["metadata"]  # list of dicts with at least 'id' field
    # Build id->metadata dict
    meta_by_id = {item["id"]: item for item in metadata}
    return bm25, docs, meta_by_id

# --------- Load FAISS index and metadata ---------
def load_faiss_index(index_path, metadata_path):
    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    meta_by_id = {item["id"]: item for item in metadata}
    return index, meta_by_id

# --------- BM25 retrieval ---------
def bm25_search(bm25, query, docs, meta_by_id, top_n=50):
    tokens = query.lower().split()  # adapt tokenization if needed
    scores = bm25.get_scores(tokens)  # returns numpy array with scores per doc index
    ranked_idx_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results = []
    for idx, score in ranked_idx_scores[:top_n]:
        doc_id = meta_by_id[docs[idx]]["id"] if docs[idx] in meta_by_id else idx
        results.append({
            "id": doc_id,
            "score": score,
            "source": "bm25",
            "metadata": meta_by_id.get(doc_id, {}),
            "text": docs[idx]
        })
    return results

# --------- FAISS retrieval ---------
def faiss_search(faiss_index, model, query, meta_by_id, top_n=50):
    q_emb = model.encode([query])
    D, I = faiss_index.search(np.array(q_emb).astype(np.float32), top_n)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx == -1:  # FAISS returns -1 if less than top_n results
            continue
        meta = meta_by_id.get(idx, {})
        results.append({
            "id": idx,
            "score": -dist,  # For L2 distances, smaller is better, invert sign
            "source": "faiss",
            "metadata": meta,
            "text": meta.get("clean_text", "")
        })
    return results

# --------- Normalize scores to [0,1] ---------
def normalize_scores(results):
    if not results:
        return results
    scores = np.array([r["score"] for r in results])
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s > 0:
        for r in results:
            r["score"] = (r["score"] - min_s) / (max_s - min_s)
    else:
        for r in results:
            r["score"] = 1.0  # all equal
    return results

# --------- Hybrid retrieval and rerank ---------
def hybrid_search(bm25, bm25_docs, bm25_meta, faiss_index, faiss_meta, model, query,
                  top_bm25=50, top_faiss=50, final_top=10, bm25_weight=0.5, faiss_weight=0.5):
    bm25_results = bm25_search(bm25, query, bm25_docs, bm25_meta, top_bm25)
    faiss_results = faiss_search(faiss_index, model, query, faiss_meta, top_faiss)
    
    bm25_results = normalize_scores(bm25_results)
    faiss_results = normalize_scores(faiss_results)

    combined = {}
    # Insert BM25 results
    for r in bm25_results:
        combined[r["id"]] = {
            "id": r["id"],
            "score": r["score"] * bm25_weight,
            "sources": {"bm25": True},
            "metadata": r["metadata"],
            "text": r["text"]
        }
    # Merge FAISS results
    for r in faiss_results:
        if r["id"] in combined:
            combined[r["id"]]["score"] += r["score"] * faiss_weight
            combined[r["id"]]["sources"]["faiss"] = True
        else:
            combined[r["id"]] = {
                "id": r["id"],
                "score": r["score"] * faiss_weight,
                "sources": {"faiss": True},
                "metadata": r["metadata"],
                "text": r["text"]
            }

    # Sort by combined score desc
    sorted_results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)

    # Return top final results
    return sorted_results[:final_top]

# ------------------ Main --------------------
if __name__ == "__main__":
    # Paths
    bm25_pickle_path = "bm25_full_index.pkl"  # update to your path
    faiss_index_path = "derry_girls_faiss.index"  # update to your path
    faiss_metadata_path = "derry_girls_metadata.json"  # update to your path

    # Load indices
    bm25, bm25_docs, bm25_meta = load_bm25_index(bm25_pickle_path)
    faiss_index, faiss_meta = load_faiss_index(faiss_index_path, faiss_metadata_path)

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Example query
    query = "Protestants and Catholics arguments"

    # Search
    results = hybrid_search(bm25, bm25_docs, bm25_meta, faiss_index, faiss_meta, model, query)

    # Display results
    for rank, res in enumerate(results, 1):
        sources = "+".join(res["sources"].keys())
        meta = res["metadata"]
        print(f"Rank {rank} (score={res['score']:.3f}, from {sources}):")
        print(f"  Season: {meta.get('season')} Episode: {meta.get('episode')}")
        print(f"  Speaker: {meta.get('speaker')}")
        print(f"  Scene: {meta.get('scene')}")
        print(f"  Text: {res['text']}")
        print("-" * 60)
