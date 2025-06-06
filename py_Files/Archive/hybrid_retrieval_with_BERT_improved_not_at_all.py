# Enhanced hybrid retrieval with improvements for better performance
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
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

k_faiss = 12  # Increased for better candidate pool
k_bm25 = 12 
k_bert = 12
k_hybrid = 8

model_name = "all-mpnet-base-v2"
bert_model_name = "bert-base-uncased"
faiss_index_path = "/home/mlt_ml3/IR_Derry_Girls/Dataset/derry_girls_faiss_new.index"
faiss_metadata_path = "/home/mlt_ml3/IR_Derry_Girls/Dataset/derry_girls_metadata_new.json"
bm25_index_path = "/home/mlt_ml3/IR_Derry_Girls/Dataset/bm25_full_index_stemmed.pkl"
bert_embeddings_path = "/home/mlt_ml3/IR_Derry_Girls/Dataset/bert_embeddings.pkl"

class EnhancedBERTRetriever:
    def __init__(self, model_name=bert_model_name):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.embeddings = None
        
    def encode_text(self, text, max_length=512):
        """Enhanced text encoding with mean pooling"""
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
            
            # Use mean pooling instead of just [CLS] token for better representation
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            
            # Mean pooling with attention mask
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
        return embeddings.cpu().numpy()
    
    def create_embeddings(self, documents, metadata, batch_size=16):  # Reduced batch size for stability
        """Create BERT embeddings for all documents with better text preprocessing"""
        print("Creating enhanced BERT embeddings...")
        embeddings = []
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_embeddings = []
            
            for j, doc in enumerate(batch_docs):
                # Enhanced text preprocessing
                if isinstance(doc, dict):
                    text = doc.get('clean_text', '')
                elif isinstance(doc, str):
                    text = doc
                else:
                    text = str(doc)
                
                # Add speaker context if available
                doc_idx = i + j
                if doc_idx < len(metadata) and metadata[doc_idx].get('speaker'):
                    speaker = metadata[doc_idx]['speaker']
                    text = f"{speaker}: {text}"
                
                # Ensure minimum text length
                if len(text.strip()) < 10:
                    text = text + " " + metadata[doc_idx].get('scene', '')
                
                embedding = self.encode_text(text)
                batch_embeddings.append(embedding[0])
            
            embeddings.extend(batch_embeddings)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {i + len(batch_docs)}/{len(documents)} documents")
        
        self.embeddings = np.array(embeddings)
        print(f"Created enhanced BERT embeddings with shape: {self.embeddings.shape}")
        return self.embeddings
    
    def save_embeddings(self, path):
        if self.embeddings is not None:
            with open(path, 'wb') as f:
                pickle.dump(self.embeddings, f)
            print(f"Enhanced BERT embeddings saved to {path}")
    
    def load_embeddings(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.embeddings = pickle.load(f)
            print(f"Enhanced BERT embeddings loaded from {path}")
            return True
        return False
    
    def search(self, query, k=k_bert):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Please create or load embeddings first.")
        
        query_embedding = self.encode_text(query)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        top_k_indices = np.argsort(similarities)[::-1][:k]
        results = [(similarities[i], i) for i in top_k_indices]
        
        return results

def enhanced_normalize_text(text):
    """Enhanced text normalization"""
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    
    # More sophisticated cleaning
    text = re.sub(r'[^\w\s\-\']', ' ', text)  # Keep hyphens and apostrophes
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
    
    return text.strip()

def advanced_hybrid_search(query, bm25, docs, faiss_index, model, bert_retriever, metadata, 
                          alpha_bm25=0.3, alpha_faiss=0.35, alpha_bert=0.35, k=8):
    """Advanced hybrid search with better fusion (query expansion removed)"""
    
    # Get candidates with original query
    bm25_results = bm25_search(query, bm25, docs, metadata, k=k_bm25)
    faiss_results = faiss_search(query, faiss_index, model, metadata, k=k_faiss)
    bert_results = bert_search(query, bert_retriever, metadata, k=k_bert)

    # Advanced score fusion using RRF (Reciprocal Rank Fusion)
    def reciprocal_rank_fusion(results_list, k_param=60):
        """Reciprocal Rank Fusion for combining rankings"""
        rrf_scores = defaultdict(float)
        
        for method_results in results_list:
            for rank, (score, doc_id) in enumerate(method_results):
                rrf_scores[doc_id] += 1 / (k_param + rank + 1)
        
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Apply RRF
    rrf_combined = reciprocal_rank_fusion([bm25_results, faiss_results, bert_results])
    
    # Also apply weighted combination as before
    def weighted_combination():
        bm25_doc_ids = [doc_id for _, doc_id in bm25_results]
        bm25_scores = np.array([score for score, _ in bm25_results])
        
        faiss_doc_ids = [doc_id for _, doc_id in faiss_results]
        faiss_scores = np.array([score for score, _ in faiss_results])
        
        bert_doc_ids = [doc_id for _, doc_id in bert_results]
        bert_scores = np.array([score for score, _ in bert_results])
        
        # Min-max normalization for better score alignment
        def min_max_normalize(scores):
            if len(scores) == 0:
                return scores
            min_score, max_score = scores.min(), scores.max()
            if max_score == min_score:
                return np.ones_like(scores)
            return (scores - min_score) / (max_score - min_score)
        
        bm25_norm = min_max_normalize(bm25_scores)
        faiss_norm = min_max_normalize(faiss_scores)
        bert_norm = min_max_normalize(bert_scores)
        
        combined = defaultdict(float)
        
        for doc_id, score in zip(bm25_doc_ids, bm25_norm):
            combined[doc_id] += alpha_bm25 * score
        
        for doc_id, score in zip(faiss_doc_ids, faiss_norm):
            combined[doc_id] += alpha_faiss * score
            
        for doc_id, score in zip(bert_doc_ids, bert_norm):
            combined[doc_id] += alpha_bert * score
        
        return sorted(combined.items(), key=lambda x: x[1], reverse=True)
    
    weighted_combined = weighted_combination()
    
    # Ensemble of RRF and weighted combination
    final_combined = defaultdict(float)
    
    # RRF contribution (60%)
    for doc_id, score in rrf_combined:
        final_combined[doc_id] += 0.6 * score
    
    # Weighted contribution (40%)
    max_weighted_score = max([score for _, score in weighted_combined]) if weighted_combined else 1
    for doc_id, score in weighted_combined:
        final_combined[doc_id] += 0.4 * (score / max_weighted_score)
    
    # Final ranking
    final_results = sorted(final_combined.items(), key=lambda x: x[1], reverse=True)[:k]
    hybrid_results = [(score, doc_id) for doc_id, score in final_results]
    
    return hybrid_results, bm25_results[:k_bm25], faiss_results[:k_faiss], bert_results[:k_bert]

def bm25_search(query, bm25, docs, metadata, k=k_bm25):
    tokenized_query = enhanced_normalize_text(query).split()
    scores = bm25.get_scores(tokenized_query)
    top_k = np.argsort(scores)[::-1][:k]
    return [(scores[i], i) for i in top_k]

def faiss_search(query, index, model, metadata, k=k_faiss):
    query_embedding = model.encode([enhanced_normalize_text(query)])
    scores, indices = index.search(np.array(query_embedding).astype('float32'), k)
    similarities = [(1.0 / (1.0 + scores[0][i]), int(indices[0][i])) for i in range(k)]
    return similarities

def bert_search(query, bert_retriever, metadata, k=k_bert):
    results = bert_retriever.search(query, k)
    return results

# Keep all the evaluation functions the same...
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

def enhanced_evaluate_system(queries: dict, bm25, docs, faiss_index, model, bert_retriever, metadata, 
                           alpha_bm25=0.3, alpha_faiss=0.35, alpha_bert=0.35, k=8):
    total_precision, total_rr, total_ap, total_ndcg = 0, 0, 0, 0
    valid_queries = 0
    results_summary = []

    for query, relevant_ids in queries.items():
        if not relevant_ids:
            continue
        
        hybrid_results, bm25_results, faiss_results, bert_results = advanced_hybrid_search(
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

        print(f"\nQuery: {query}")
        print(f"Hits: {hits}/{k} | P@{k}: {precision:.3f} | MRR: {rr:.3f} | AP: {ap:.3f} | nDCG@{k}: {ndcg:.3f}")
        
        results_summary.append({
            'query': query,
            'hits': hits,
            'precision': precision,
            'mrr': rr,
            'ap': ap,
            'ndcg': ndcg
        })

    if valid_queries > 0:
        avg_precision = total_precision / valid_queries
        avg_mrr = total_rr / valid_queries
        avg_map = total_ap / valid_queries
        avg_ndcg = total_ndcg / valid_queries
        
        print(f"\n{'='*60}")
        print(f"ENHANCED RESULTS:")
        print(f"Average Precision@{k}: {avg_precision:.4f}")
        print(f"Mean Reciprocal Rank: {avg_mrr:.4f}")
        print(f"Mean Average Precision: {avg_map:.4f}")
        print(f"Mean nDCG@{k}: {avg_ndcg:.4f}")
        print(f"{'='*60}")
        
        return avg_precision, avg_mrr, avg_map, avg_ndcg

def setup_enhanced_bert_embeddings(bert_retriever, docs, metadata):
    """Setup enhanced BERT embeddings"""
    embeddings_exist = bert_retriever.load_embeddings(bert_embeddings_path.replace('.pkl', '_enhanced.pkl'))
    
    if not embeddings_exist:
        print("Creating enhanced BERT embeddings...")
        documents = []
        for doc in metadata:
            clean_text = doc.get('clean_text', '')
            documents.append(clean_text if clean_text else '')
        
        bert_retriever.create_embeddings(documents, metadata)
        bert_retriever.save_embeddings(bert_embeddings_path.replace('.pkl', '_enhanced.pkl'))

def main():
    # Load data
    with open(bm25_index_path, "rb") as f:
        bm25_data = pickle.load(f)
        bm25 = bm25_data["bm25"]
        docs = bm25_data["docs"]
        bm25_metadata = bm25_data["metadata"]

    faiss_index = faiss.read_index(faiss_index_path)
    with open(faiss_metadata_path, "r", encoding="utf-8") as f:
        faiss_metadata = json.load(f)

    # Load enhanced models
    model = SentenceTransformer(model_name)
    bert_retriever = EnhancedBERTRetriever(bert_model_name)
    
    # Setup enhanced embeddings
    setup_enhanced_bert_embeddings(bert_retriever, docs, faiss_metadata)

    # Test enhanced system
    print("Testing Enhanced Hybrid System...")
    enhanced_evaluate_system(queries, bm25, docs, faiss_index, model, bert_retriever, faiss_metadata)

if __name__ == "__main__":
    main()