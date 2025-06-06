import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from py_Files.SPLADE.queries_splade import queries

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

# Encode text into sparse vectors using SPLADE
def encode_splade(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        sparse_vectors = torch.max(outputs.logits, dim=1).values
    return sparse_vectors.cpu().numpy()


queries = {normalize_text(q): v for q, v in queries.items()}


# Search function without biasing
def search(query, index, metadata, top_k=5):
    query_vector = encode_splade([normalize_text(query)])
    similarities = cosine_similarity(query_vector, index).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [(metadata[i], similarities[i]) for i in top_indices]

# Evaluation function
def evaluate(queries, index, metadata, top_k=5):
    results = {}
    for query, expected_indices in queries.items():
        search_results = search(query, index, metadata, top_k=top_k)
        retrieved_ids = [meta["id"] for meta, _ in search_results]

        recall_at_k = 1 if any(idx in retrieved_ids for idx in expected_indices) else 0
        mrr = 0
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in expected_indices:
                mrr = 1 / rank
                break

        results[query] = {
            "Recall@5": recall_at_k,
            "MRR": mrr
        }

    return results

# Load index and metadata
index_path = "/home/mlt_ml3/IR_Derry_Girls/py_Files/SPLADE/splade_index.npz"
metadata_path = "/home/mlt_ml3/IR_Derry_Girls/py_Files/SPLADE/metadata.json"

index = load_npz(index_path)
with open(metadata_path, "r") as f:
    metadata = json.load(f)

# Run evaluation
evaluation_results = evaluate(queries, index, metadata, top_k=5)
for query, metrics in evaluation_results.items():
    print(f"Query: {query}")
    print(f"Recall@5: {metrics['Recall@5']:.4f}")
    print(f"MRR: {metrics['MRR']:.4f}")
    print("-" * 50)

