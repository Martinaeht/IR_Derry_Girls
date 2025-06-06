#evaluation_function_from_hybrid_final_draft_improved
"""
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

        print_results_with_context(hybrid_results, metadata, query, title="Hybrid Results with Context")
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
"""