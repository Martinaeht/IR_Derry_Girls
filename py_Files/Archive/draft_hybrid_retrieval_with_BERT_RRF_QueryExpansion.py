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

# Try to import NLTK for WordNet expansion
try:
    import nltk
    from nltk.corpus import wordnet
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not available. WordNet expansion will be disabled.")

k_faiss = 8
k_bm25 = 8 
k_hybrid = 8

model_name = "all-mpnet-base-v2"
faiss_index_path = "/home/mlt_ml3/IR_Derry_Girls/Dataset/derry_girls_faiss_new.index"
faiss_metadata_path = "/home/mlt_ml3/IR_Derry_Girls/Dataset/derry_girls_metadata_new.json"
bm25_index_path = "/home/mlt_ml3/IR_Derry_Girls/Dataset/bm25_full_index_stemmed.pkl"

# Enhanced parameters
CANDIDATE_MULTIPLIER = 2  # Retrieve 2x more candidates initially
RRF_K = 60  # Reciprocal Rank Fusion parameter
BERT_RERANK_TOP_K = 12  # Number of candidates to re-rank with BERT
BERT_ALPHA = 0.7  # Weight for BERT scores in final combination

def normalize_text(text):
    """Normalize text by removing accents, converting to lowercase, and removing special characters."""
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')  # Remove accents
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def get_wordnet_synonyms(word, pos=None):
    """Get synonyms for a word using WordNet."""
    if not NLTK_AVAILABLE:
        return []
    
    synonyms = set()
    try:
        for syn in wordnet.synsets(word, pos=pos):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ').lower()
                if synonym != word.lower() and len(synonym) > 2:
                    synonyms.add(synonym)
    except:
        pass
    
    return list(synonyms)[:3]  # Limit to 3 synonyms per word

def expand_query_wordnet(query):
    """Expand query using WordNet synonyms."""
    if not NLTK_AVAILABLE:
        return query
    
    words = query.lower().split()
    expanded_terms = []
    
    for word in words:
        # Skip very short words and common stop words
        if len(word) <= 2 or word in ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']:
            continue
            
        synonyms = get_wordnet_synonyms(word)
        if synonyms:
            expanded_terms.extend(synonyms[:2])  # Add up to 2 synonyms per word
    
    if expanded_terms:
        expanded_query = query + " " + " ".join(expanded_terms)
        print(f"WordNet expansion: '{query}' → '{expanded_query}'")
        return expanded_query
    
    return query

def expand_query_semantic(query, model, metadata, top_docs=5):
    """Expand query using semantic similarity with top documents."""
    # Get initial search results
    query_embedding = model.encode([normalize_text(query)])
    
    # Simple semantic expansion using document terms
    expanded_terms = []
    query_words = set(normalize_text(query).split())
    
    # Look through some documents to find related terms
    for doc_id, doc in enumerate(metadata[:100]):  # Sample first 100 docs
        doc_text = normalize_text(doc.get('clean_text', ''))
        doc_words = set(doc_text.split())
        
        # Find words that appear with query terms
        if query_words.intersection(doc_words):
            # Add some context words (simple heuristic)
            for word in doc_words:
                if (len(word) > 3 and 
                    word not in query_words and 
                    len(expanded_terms) < 5):
                    expanded_terms.append(word)
    
    if expanded_terms:
        # Take most common expansion terms
        from collections import Counter
        term_counts = Counter(expanded_terms)
        top_terms = [term for term, _ in term_counts.most_common(3)]
        
        expanded_query = query + " " + " ".join(top_terms)
        print(f"Semantic expansion: '{query}' → '{expanded_query}'")
        return expanded_query
    
    return query

def expand_query(query, method='both', model=None, metadata=None):
    """Expand query using specified method(s)."""
    if method == 'wordnet':
        return expand_query_wordnet(query)
    elif method == 'semantic' and model and metadata:
        return expand_query_semantic(query, model, metadata)
    elif method == 'both':
        # Combine both methods
        wordnet_expanded = expand_query_wordnet(query)
        if model and metadata:
            semantic_expanded = expand_query_semantic(wordnet_expanded, model, metadata)
            return semantic_expanded
        return wordnet_expanded
    else:
        return query

def reciprocal_rank_fusion(rankings_list, k=RRF_K):
    """Combine multiple rankings using Reciprocal Rank Fusion."""
    combined_scores = defaultdict(float)
    
    for rankings in rankings_list:
        for rank, (score, doc_id) in enumerate(rankings, 1):
            combined_scores[doc_id] += 1.0 / (k + rank)
    
    # Sort by combined score (higher is better)
    return sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

def bert_rerank(query, candidates, metadata, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Re-rank candidates using BERT cross-encoder."""
    if not candidates:
        return []
    
    try:
        # Initialize BERT re-ranker
        reranker = CrossEncoder(model_name)
        
        # Prepare query-document pairs for BERT
        pairs = []
        doc_ids = []
        
        for score, doc_id in candidates:
            doc = metadata[doc_id]
            doc_text = doc.get('clean_text', '')[:512]  # Limit text length for BERT
            pairs.append([query, doc_text])
            doc_ids.append((score, doc_id))
        
        # Get BERT relevance scores
        bert_scores = reranker.predict(pairs)
        
        # Combine original scores with BERT scores
        combined_results = []
        for i, (original_score, doc_id) in enumerate(doc_ids):
            bert_score = float(bert_scores[i])
            # Combine scores: 70% BERT, 30% original
            final_score = BERT_ALPHA * bert_score + (1 - BERT_ALPHA) * original_score
            combined_results.append((final_score, doc_id))
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x[0], reverse=True)
        
        print(f"BERT re-ranking applied to {len(candidates)} candidates")
        return combined_results
        
    except Exception as e:
        print(f"BERT re-ranking failed: {e}")
        return candidates

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

def faiss_search(query, index, model, metadata, k=k_faiss):
    query_embedding = model.encode([normalize_text(query)])
    scores, indices = index.search(np.array(query_embedding).astype('float32'), k)
    
    # Convert L2 distances to similarities (lower distance = higher similarity)
    similarities = [(1.0 / (1.0 + scores[0][i]), int(indices[0][i])) for i in range(k)]
    
    print(f"[faiss_search] Query: {query}")
    print("Raw distances:", scores[0][:3])
    print("Converted similarities:", [s[0] for s in similarities[:3]])
    
    return similarities

def enhanced_hybrid_search(query, bm25, docs, faiss_index, model, metadata, 
                          expand_query_flag=True, expansion_method='both',
                          use_bert_rerank=True, alpha=0.7, k=k_hybrid):
    """Enhanced hybrid search with query expansion and BERT re-ranking."""
    
    # Step 1: Query Expansion
    original_query = query
    if expand_query_flag:
        query = expand_query(query, method=expansion_method, model=model, metadata=metadata)
    
    # Step 2: Retrieve more candidates initially
    expanded_k = k * CANDIDATE_MULTIPLIER
    
    bm25_results = bm25_search(query, bm25, docs, metadata, k=expanded_k)
    faiss_results = faiss_search(query, faiss_index, model, metadata, k=expanded_k)
    
    # Step 3: Reciprocal Rank Fusion
    rrf_results = reciprocal_rank_fusion([bm25_results, faiss_results])
    
    # Convert RRF results back to (score, doc_id) format and take top candidates
    top_candidates = [(score, doc_id) for doc_id, score in rrf_results[:BERT_RERANK_TOP_K]]
    
    # Step 4: BERT Re-ranking (optional)
    if use_bert_rerank and len(top_candidates) > 1:
        final_results = bert_rerank(original_query, top_candidates, metadata)
    else:
        final_results = top_candidates
    
    # Return top k results
    final_results = final_results[:k]
    
    print(f"[Enhanced Hybrid Search] Original query: '{original_query}'")
    if expand_query_flag and query != original_query:
        print(f"[Enhanced Hybrid Search] Expanded query: '{query}'")
    print("Top Enhanced Hybrid DocIDs:", [doc_id for _, doc_id in final_results])
    
    return final_results, bm25_results, faiss_results

def hybrid_search(query, bm25, docs, faiss_index, model, metadata, alpha=0.7, k=k_hybrid):
    """Original hybrid search function (maintained for compatibility)."""
    bm25_results = bm25_search(query, bm25, docs, metadata, k=k_bm25)
    faiss_results = faiss_search(query, faiss_index, model, metadata, k=k_faiss)

    # Extract scores and DocIDs
    bm25_doc_ids = [doc_id for _, doc_id in bm25_results]
    bm25_scores = np.array([score for score, _ in bm25_results]).reshape(-1, 1)

    faiss_doc_ids = [doc_id for _, doc_id in faiss_results]
    faiss_scores = np.array([score for score, _ in faiss_results]).reshape(-1, 1)

    # Normalize scores
    if len(bm25_scores) > 1:
        bm25_scores = MinMaxScaler().fit_transform(bm25_scores).flatten()
    else:
        bm25_scores = bm25_scores.flatten()

    if len(faiss_scores) > 1:
        faiss_scores = MinMaxScaler().fit_transform(faiss_scores).flatten()
    else:
        faiss_scores = faiss_scores.flatten()

    # Combine
    combined = defaultdict(float)
    for doc_id, score in zip(bm25_doc_ids, bm25_scores):
        combined[doc_id] += (1 - alpha) * score
    for doc_id, score in zip(faiss_doc_ids, faiss_scores):
        combined[doc_id] += alpha * score

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
    hybrid_results = [(score, doc_id) for doc_id, score in ranked]

    print(f"[Hybrid Search] Query: '{query}'")
    print("Top Hybrid DocIDs:", [doc_id for _, doc_id in hybrid_results])

    return hybrid_results, bm25_results, faiss_results

def print_hybrid_comparison(query, hybrid_results, bm25_results, faiss_results, metadata):
    print(f"\nHybrid Search Results for: '{query}'\n")
    print(f"{'Rank':<5}{'Score':<10}{'DocID':<6} | {'BM25 Score':<10}{'Faiss Score':<10} | Text Snippet")
    print("-" * 80)
    
    # Helper dict for quick score access
    bm25_dict = dict(bm25_results)
    faiss_dict = dict(faiss_results)
    
    for rank, (score, doc_id) in enumerate(hybrid_results, 1):
        bm25_score = bm25_dict.get(doc_id, 0)
        faiss_score = faiss_dict.get(doc_id, 0)
        doc = metadata[doc_id]
        snippet = doc["clean_text"][:60].replace("\n", " ") + "..."
        print(f"{rank:<5}{score:<10.4f}{doc_id:<6} | {bm25_score:<10.4f}{faiss_score:<10.4f} | {snippet}")

def print_enhanced_comparison(query, enhanced_results, bm25_results, faiss_results, metadata):
    """Print enhanced results with detailed comparison."""
    print(f"\nEnhanced Hybrid Search Results for: '{query}'\n")
    print(f"{'Rank':<5}{'Enhanced Score':<15}{'DocID':<6} | {'BM25':<10}{'FAISS':<10} | Scene | Text Snippet")
    print("-" * 100)
    
    # Helper dicts for quick score access
    bm25_dict = dict(bm25_results)
    faiss_dict = dict(faiss_results)
    
    for rank, (score, doc_id) in enumerate(enhanced_results, 1):
        bm25_score = bm25_dict.get(doc_id, 0)
        faiss_score = faiss_dict.get(doc_id, 0)
        doc = metadata[doc_id]
        scene = doc.get("scene", "Unknown")
        snippet = doc["clean_text"][:40].replace("\n", " ") + "..."
        print(f"{rank:<5}{score:<15.4f}{doc_id:<6} | {bm25_score:<10.4f}{faiss_score:<10.4f} | {scene:<10} | {snippet}")

# Evaluation functions (unchanged)
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

def evaluate_hybrid(queries: dict, bm25, docs, faiss_index, model, metadata, alpha=0.7, k=8):
    """Original evaluation function (unchanged)."""
    total_precision, total_rr, total_ap, total_ndcg = 0, 0, 0, 0
    valid_queries = 0

    for query, relevant_ids in queries.items():
        if not relevant_ids:
            continue
        
        normalized_query = normalize_text(query)
        print(f"Original query: '{query}'")
        print(f"Normalized query: '{normalized_query}'")
        
        hybrid_results, _, _ = hybrid_search(normalized_query, bm25, docs, faiss_index, model, metadata, alpha)
        retrieved_ids = [doc_id for _, doc_id in hybrid_results]

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

def evaluate_enhanced_hybrid(queries: dict, bm25, docs, faiss_index, model, metadata, 
                           expand_query_flag=True, use_bert_rerank=True, k=8):
    """Evaluate enhanced hybrid search system."""
    total_precision, total_rr, total_ap, total_ndcg = 0, 0, 0, 0
    valid_queries = 0

    print("\n" + "="*60)
    print("EVALUATING ENHANCED HYBRID SEARCH SYSTEM")
    print("="*60)

    for query, relevant_ids in queries.items():
        if not relevant_ids:
            continue
        
        normalized_query = normalize_text(query)
        print(f"\nOriginal query: '{query}'")
        print(f"Normalized query: '{normalized_query}'")
        
        enhanced_results, bm25_results, faiss_results = enhanced_hybrid_search(
            normalized_query, bm25, docs, faiss_index, model, metadata,
            expand_query_flag=expand_query_flag, use_bert_rerank=use_bert_rerank, k=k
        )
        retrieved_ids = [doc_id for _, doc_id in enhanced_results]

        print(f"Query: {query}")
        print(f"Relevant IDs: {relevant_ids}")
        print(f"Retrieved IDs: {retrieved_ids}")

        total_precision += precision_at_k(retrieved_ids, relevant_ids, k)
        total_rr += reciprocal_rank(retrieved_ids, relevant_ids)
        total_ap += average_precision(retrieved_ids, relevant_ids)
        total_ndcg += ndcg_at_k(retrieved_ids, relevant_ids, k)
        valid_queries += 1

        print_enhanced_comparison(query, enhanced_results, bm25_results, faiss_results, metadata)
        print(f"Precision@{k}: {total_precision/valid_queries:.4f}, MRR: {total_rr/valid_queries:.4f}, AP: {total_ap/valid_queries:.4f}, nDCG@{k}: {total_ndcg/valid_queries:.4f}")
        print("=" * 60)

    if valid_queries > 0:
        print("\nENHANCED EVALUATION RESULTS:")
        print(f"Average Precision@{k}: {total_precision / valid_queries:.4f}")
        print(f"Mean Reciprocal Rank: {total_rr / valid_queries:.4f}")
        print(f"Mean Average Precision: {total_ap / valid_queries:.4f}")
        print(f"Mean nDCG@{k}: {total_ndcg / valid_queries:.4f}")
    else:
        print("No valid queries found.")

def main():
    # Load BM25
    with open(bm25_index_path, "rb") as f:
        bm25_data = pickle.load(f)
        bm25 = bm25_data["bm25"]
        docs = bm25_data["docs"]
        bm25_metadata = bm25_data["metadata"]

    # Load FAISS
    faiss_index = faiss.read_index(faiss_index_path)
    with open(faiss_metadata_path, "r", encoding="utf-8") as f:
        faiss_metadata = json.load(f)

    model = SentenceTransformer(model_name)

    # Download NLTK data if available
    if NLTK_AVAILABLE:
        try:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        except:
            pass

    print("Choose evaluation mode:")
    print("1. Original Hybrid Search")
    print("2. Enhanced Hybrid Search (with query expansion and BERT re-ranking)")
    print("3. Both (comparison)")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1":
        evaluate_hybrid(queries, bm25, docs, faiss_index, model, faiss_metadata)
    elif choice == "2":
        evaluate_enhanced_hybrid(queries, bm25, docs, faiss_index, model, faiss_metadata)
    elif choice == "3":
        print("\n" + "="*80)
        print("ORIGINAL HYBRID SEARCH EVALUATION")
        print("="*80)
        evaluate_hybrid(queries, bm25, docs, faiss_index, model, faiss_metadata)
        
        print("\n" + "="*80)
        print("ENHANCED HYBRID SEARCH EVALUATION")
        print("="*80)
        evaluate_enhanced_hybrid(queries, bm25, docs, faiss_index, model, faiss_metadata)
    else:
        print("Invalid choice. Running original evaluation.")
        evaluate_hybrid(queries, bm25, docs, faiss_index, model, faiss_metadata)

    # Interactive query loop
    print("\n" + "="*60)
    print("INTERACTIVE SEARCH MODE")
    print("="*60)
    print("Commands:")
    print("- Type your query for enhanced search")
    print("- Type 'original: <query>' for original hybrid search")
    print("- Type 'exit' to quit")
    
    while True:
        user_input = input("\nType your query or 'exit' if you want to quit: ").strip()
        if user_input.lower() == "exit":
            break
        elif user_input:
            if user_input.startswith("original:"):
                # Use original hybrid search
                query = user_input[9:].strip()
                hybrid_results, bm25_results, faiss_results = hybrid_search(
                    query, bm25, docs, faiss_index, model, faiss_metadata
                )
                print_hybrid_comparison(query, hybrid_results, bm25_results, faiss_results, faiss_metadata)
            else:
                # Use enhanced hybrid search
                query = user_input
                enhanced_results, bm25_results, faiss_results = enhanced_hybrid_search(
                    query, bm25, docs, faiss_index, model, faiss_metadata,
                    expand_query_flag=True, use_bert_rerank=True
                )
                print_enhanced_comparison(query, enhanced_results, bm25_results, faiss_results, faiss_metadata)

if __name__ == "__main__":
    main()