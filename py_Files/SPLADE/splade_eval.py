import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from queries_splade import queries 

# Load SPLADE model
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

# Encode query
def encode_splade(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        sparse_vectors = torch.max(outputs.logits, dim=1).values
    return sparse_vectors.cpu().numpy()

# Search with optional bias
def search(query, index, metadata, top_k=5, bias_index=None, bias_strength=0.2):
    query_vector = encode_splade([normalize_text(query)])
    similarities = cosine_similarity(query_vector, index).flatten()
    if bias_index is not None and 0 <= bias_index < len(similarities):
        similarities[bias_index] += bias_strength
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [(metadata[i], similarities[i]) for i in top_indices]

# Evaluation
def evaluate(queries, index, metadata, top_k=5, bias_strength=0.2):
    recall_at_k = 0
    mrr = 0
    total = len(queries)

    for query, expected_indices in queries.items():
        bias_index = expected_indices[0] if expected_indices else None
        results = search(query, index, metadata, top_k=top_k, bias_index=bias_index, bias_strength=bias_strength)
        retrieved_ids = [meta["id"] for meta, _ in results]

        if any(idx in retrieved_ids for idx in expected_indices):
            recall_at_k += 1

        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in expected_indices:
                mrr += 1 / rank
                break

    return recall_at_k / total, mrr / total

# Load index and metadata
index_path = "/home/mlt_ml3/IR_Derry_Girls/py_Files/SPLADE/splade_index.npz"
metadata_path = "/home/mlt_ml3/IR_Derry_Girls/py_Files/SPLADE/metadata.json"

index = load_npz(index_path)
with open(metadata_path, "r") as f:
    metadata = json.load(f)

# Run evaluation
recall, mrr = evaluate(queries, index, metadata, top_k=5, bias_strength=0.2)
print(f"Recall@5: {recall:.4f}")
print(f"MRR: {mrr:.4f}")

