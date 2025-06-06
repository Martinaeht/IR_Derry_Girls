#hybrid retrieval improved with BERT
import os
import re
import json
import faiss
import torch
import pickle
import unicodedata
import numpy as np
from queries import queries
from rank_bm25 import BM25Okapi
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from sklearn.metrics import label_ranking_average_precision_score
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

k_faiss = 8
k_bm25 = 8 
k_bert = 8
k_hybrid = 8

model_name = "all-mpnet-base-v2"
bert_model_name = "bert-base-uncased"
faiss_index_path = "/home/mlt_ml3/IR_Derry_Girls/Dataset/derry_girls_faiss_new.index"
faiss_metadata_path = "/home/mlt_ml3/IR_Derry_Girls/Dataset/derry_girls_metadata_new.json"
bm25_index_path = "/home/mlt_ml3/IR_Derry_Girls/Dataset/bm25_full_index_stemmed.pkl"
bert_embeddings_path = "/home/mlt_ml3/IR_Derry_Girls/Dataset/bert_embeddings_new.pkl"

class BERTRetriever:
    def __init__(self, model_name=bert_model_name):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.embeddings = None
        
    def encode_text(self, text, max_length=512):
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            padding=True, 
            max_length=max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings
    
    def create_embeddings(self, documents, metadata, batch_size=32):
        print("Creating BERT embeddings...")
        embeddings = []
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_embeddings = []
            
            for doc in batch_docs:
                text = doc if isinstance(doc, str) else str(doc)
                embedding = self.encode_text(text)
                batch_embeddings.append(embedding[0])  
            
            embeddings.extend(batch_embeddings)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {i + len(batch_docs)}/{len(documents)} documents")
        
        self.embeddings = np.array(embeddings)
        print(f"Created BERT embeddings with shape: {self.embeddings.shape}")
        return self.embeddings
    
    def save_embeddings(self, path):
        if self.embeddings is not None:
            with open(path, 'wb') as f:
                pickle.dump(self.embeddings, f)
            print(f"BERT embeddings saved to {path}")
    
    def load_embeddings(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.embeddings = pickle.load(f)
            print(f"BERT embeddings loaded from {path}")
            return True
        return False
    
    def search(self, query, k=k_bert):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Please create or load embeddings first.")
        
        query_embedding = self.encode_text(([normalize_text(query)]))
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top k indices
        top_k_indices = np.argsort(similarities)[::-1][:k]
        results = [(similarities[i], i) for i in top_k_indices]
        
        return results

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
                context_before = f"[BEFORE:] {prev_speaker}: {prev_doc['clean_text']}"
        
        if doc_id + 1 < len(metadata):
            next_doc = metadata[doc_id + 1]
            if next_doc.get("scene") == scene and next_doc.get("clean_text"):
                next_speaker = next_doc.get("speaker")
                context_after = f"[AFTER:] {next_speaker}: {next_doc['clean_text']}"
        
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
    print("Raw distances:", scores[0][:3])
    print("Converted similarities:", [s[0] for s in similarities[:3]])
    
    return similarities

def bert_search(query, bert_retriever, metadata, k=k_bert):
    """Search using BERT embeddings"""
    results = bert_retriever.search(query, k)
    print(f"[bert_search] Query: {query}")
    print("Top 3 BERT similarities:", [score for score, _ in results[:3]])
    return results

def improved_hybrid_search_with_bert(query, bm25, docs, faiss_index, model, bert_retriever, metadata, 
                                   alpha_bm25=0.33, alpha_faiss=0.33, alpha_bert=0.34, k=8):

    bm25_results = bm25_search(query, bm25, docs, metadata, k=k_bm25*2)
    faiss_results = faiss_search(query, faiss_index, model, metadata, k=k_faiss*2)
    bert_results = bert_search(query, bert_retriever, metadata, k=k_bert*2)

    bm25_doc_ids = [doc_id for _, doc_id in bm25_results]
    bm25_scores = np.array([score for score, _ in bm25_results]).reshape(-1, 1)

    faiss_doc_ids = [doc_id for _, doc_id in faiss_results]
    faiss_scores = np.array([score for score, _ in faiss_results]).reshape(-1, 1)
    
    bert_doc_ids = [doc_id for _, doc_id in bert_results]
    bert_scores = np.array([score for score, _ in bert_results]).reshape(-1, 1)

    def rank_normalize(scores):
        """Convert scores to rank-based normalization (1.0 for best, decreasing)"""
        ranks = np.argsort(np.argsort(-scores.flatten())) + 1  
        return 1.0 / ranks 

    bm25_norm_scores = rank_normalize(bm25_scores)
    faiss_norm_scores = rank_normalize(faiss_scores)
    bert_norm_scores = rank_normalize(bert_scores)

    combined = defaultdict(float)

    for doc_id, score in zip(bm25_doc_ids, bm25_norm_scores):
        combined[doc_id] += alpha_bm25 * score

    for doc_id, score in zip(faiss_doc_ids, faiss_norm_scores):
        combined[doc_id] += alpha_faiss * score

    for doc_id, score in zip(bert_doc_ids, bert_norm_scores):
        combined[doc_id] += alpha_bert * score

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
    hybrid_results = [(score, doc_id) for doc_id, score in ranked]

    return hybrid_results, bm25_results[:k_bm25], faiss_results[:k_faiss], bert_results[:k_bert]

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

def enhanced_evaluate_hybrid_with_bert(queries: dict, bm25, docs, faiss_index, model, bert_retriever, metadata, 
                                     alpha_bm25=0.33, alpha_faiss=0.33, alpha_bert=0.34, k=8):
    total_precision, total_rr, total_ap, total_ndcg = 0, 0, 0, 0
    valid_queries = 0
    
    results_summary = []

    for query, relevant_ids in queries.items():
        if not relevant_ids:
            continue
        
        hybrid_results, bm25_results, faiss_results, bert_results = improved_hybrid_search_with_bert(
            query, bm25, docs, faiss_index, model, bert_retriever, metadata, 
            alpha_bm25, alpha_faiss, alpha_bert, k
        )
        retrieved_ids = [doc_id for _, doc_id in hybrid_results]

        precision = precision_at_k(retrieved_ids, relevant_ids, k)
        rr = reciprocal_rank(retrieved_ids, relevant_ids)
        ap = average_precision(retrieved_ids, relevant_ids)
        ndcg = ndcg_at_k(retrieved_ids, relevant_ids, k)

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

        print_results_with_context(hybrid_results, metadata, query, title="Hybrid Results with BERT")
        print("=" * 80)

    if valid_queries > 0:
        print(f"\n{'='*50} FINAL RESULTS {'='*50}")
        print(f"Average Precision@{k}: {total_precision / valid_queries:.4f}")
        print(f"Mean Reciprocal Rank: {total_rr / valid_queries:.4f}")
        print(f"Mean Average Precision: {total_ap / valid_queries:.4f}")
        print(f"Mean nDCG@{k}: {total_ndcg / valid_queries:.4f}")

        results_summary.sort(key=lambda x: x['precision'], reverse=True)
        print(f"\nBest performing query (P@{k}={results_summary[0]['precision']:.3f}): {results_summary[0]['query']}")
        print(f"Worst performing query (P@{k}={results_summary[-1]['precision']:.3f}): {results_summary[-1]['query']}")
    else:
        print("No valid queries found.")

def tune_hybrid_parameters_with_bert(queries, bm25, docs, faiss_index, model, bert_retriever, metadata, k=8):
    """Test different weight combinations to find optimal hybrid parameters"""
    weight_combinations = [
        (0.33, 0.33, 0.34),  # Equal weights
        (0.4, 0.3, 0.3),     # BM25 emphasis
        (0.3, 0.4, 0.3),     # FAISS emphasis
        (0.3, 0.3, 0.4),     # BERT emphasis
        (0.5, 0.25, 0.25),   # Strong BM25
        (0.25, 0.5, 0.25),   # Strong FAISS
        (0.25, 0.25, 0.5),   # Strong BERT
        (0.2, 0.4, 0.4),     # Semantic emphasis
        (0.6, 0.2, 0.2),     # Lexical emphasis
    ]
    
    best_weights = (0.33, 0.33, 0.34)
    best_map = 0.0
    
    print("Tuning hybrid parameters with BERT...")
    print(f"{'BM25':<6} {'FAISS':<6} {'BERT':<6} {'MAP':<8} {'P@8':<8} {'MRR':<8} {'nDCG@8':<8}")
    print("-" * 50)
    
    for alpha_bm25, alpha_faiss, alpha_bert in weight_combinations:
        total_map = total_precision = total_mrr = total_ndcg = 0
        valid_queries = 0
        
        for query, relevant_ids in queries.items():
            if not relevant_ids:
                continue
                
            hybrid_results, _, _, _ = improved_hybrid_search_with_bert(
                query, bm25, docs, faiss_index, model, bert_retriever, metadata, 
                alpha_bm25, alpha_faiss, alpha_bert, k
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
            
            print(f"{alpha_bm25:<6.2f} {alpha_faiss:<6.2f} {alpha_bert:<6.2f} {avg_map:<8.3f} {avg_precision:<8.3f} {avg_mrr:<8.3f} {avg_ndcg:<8.3f}")
            
            if avg_map > best_map:
                best_map = avg_map
                best_weights = (alpha_bm25, alpha_faiss, alpha_bert)
    
    print(f"\nBest weights - BM25: {best_weights[0]:.2f}, FAISS: {best_weights[1]:.2f}, BERT: {best_weights[2]:.2f} (MAP: {best_map:.3f})")
    return best_weights

def setup_bert_embeddings(bert_retriever, docs, metadata):
    if bert_retriever.load_embeddings(bert_embeddings_path):
        print("BERT embeddings loaded successfully!")
        return
    
    print("BERT embeddings not found. Creating new embeddings...")
    documents = []
    for doc in metadata:
        clean_text = doc.get('clean_text', '')
        if clean_text:
            documents.append(clean_text)
        else:
            documents.append('')
    
    bert_retriever.create_embeddings(documents, metadata)
    bert_retriever.save_embeddings(bert_embeddings_path)

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
    bert_retriever = BERTRetriever(bert_model_name)

    setup_bert_embeddings(bert_retriever, docs, faiss_metadata)

    optimal_weights = tune_hybrid_parameters_with_bert(queries, bm25, docs, faiss_index, model, bert_retriever, faiss_metadata)

    enhanced_evaluate_hybrid_with_bert(
        queries, bm25, docs, faiss_index, model, bert_retriever, faiss_metadata, 
        alpha_bm25=optimal_weights[0], alpha_faiss=optimal_weights[1], alpha_bert=optimal_weights[2]
    )

    while True:
        query = input("\nType your query or 'exit' if you want to quit: ").strip()
        if query.lower() == "exit":
            break
        elif query:
            hybrid_results, bm25_results, faiss_results, bert_results = improved_hybrid_search_with_bert(
                query, bm25, docs, faiss_index, model, bert_retriever, faiss_metadata, 
                alpha_bm25=optimal_weights[0], alpha_faiss=optimal_weights[1], alpha_bert=optimal_weights[2]
            )
            print_results_with_context(hybrid_results, faiss_metadata, query, title="Hybrid Search Results with BERT")

if __name__ == "__main__":
    main()