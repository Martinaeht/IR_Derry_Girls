#FAISS Indexing with all-mpnet-base-v2
from pprint import pprint
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from queries import queries
import re
import json
import faiss
import os
import numpy as np
import unicodedata
import random
random.seed(42)
np.random.seed(42)

def normalize_text_for_faiss(text: str) -> str:
    text = text.lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

file_path = "/home/mlt_ml3/IR_Derry_Girls/Dataset/DERRY-GIRLS-SCRIPT.txt"

with open(file_path, "r", encoding="windows-1252") as file:
    raw_lines = file.readlines()

season_pattern = re.compile(r"^SEASON\s+(\d+)", re.IGNORECASE)
episode_pattern = re.compile(r"^EPISODE\s+(\d+)", re.IGNORECASE)
scene_pattern = re.compile(r"^\[(.*)]$")
speaker_pattern = re.compile(r"^([A-Za-z0-9 ']+):")
action_pattern = re.compile(r"\((.*?)\)")

parsed_lines = []
scene = None
season = None
episode = None

for line_number, raw_line in enumerate(raw_lines):
    line = raw_line.strip()
    if not line:
        continue

    if season_pattern.match(line):
        season = int(season_pattern.match(line).group(1))
        continue

    if episode_pattern.match(line):
        episode = int(episode_pattern.match(line).group(1))
        continue

    if scene_pattern.match(line):
        scene = scene_pattern.match(line).group(1).strip()
        parsed_lines.append({
            "id": len(parsed_lines),
            "line_number": line_number,
            "season": season,
            "episode": episode,
            "scene": scene,
            "speaker": None,
            "actions": [],
            "clean_text": None,
            "raw_line": line
        })
        continue

    speaker_match = speaker_pattern.match(line)
    if speaker_match:
        speaker = speaker_match.group(1).strip()
        dialogue_with_actions = line[len(speaker) + 1:].strip()
    else:
        speaker = None
        dialogue_with_actions = line

    actions = action_pattern.findall(dialogue_with_actions)
    clean_text = action_pattern.sub("", dialogue_with_actions).strip()

    parsed_lines.append({
        "id": len(parsed_lines),
        "line_number": line_number,
        "season": season,
        "episode": episode,
        "scene": scene,
        "speaker": speaker,
        "actions": actions,
        "clean_text": clean_text if clean_text else None,
        "raw_line": line
    })

print("Parsed lines preview:")
pprint(parsed_lines[:15])

indexed_lines = [line for line in parsed_lines if line["clean_text"]]

documents = [
    normalize_text_for_faiss(line["clean_text"])
    for line in indexed_lines
]

for i in range(len(indexed_lines)):
    expected = normalize_text_for_faiss(indexed_lines[i]["clean_text"])
    assert documents[i] == expected, f"Mismatch at index {i}"

model = SentenceTransformer("all-mpnet-base-v2")
document_embeddings = model.encode(documents)

dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(document_embeddings))

print(f"Number of items in FAISS index: {index.ntotal}")

index_path = "/home/mlt_ml3/IR_Derry_Girls/Dataset/derry_girls_faiss_new.index"
metadata_path = "/home/mlt_ml3/IR_Derry_Girls/Dataset/derry_girls_metadata_new.json"

with open(index_path, "wb") as f:
    faiss.write_index(index, index_path)

with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(indexed_lines, f, ensure_ascii=False, indent=2)

print(f"Saved FAISS index to: {index_path}")
print(f"Saved metadata to: {metadata_path}")

k=10

import matplotlib.pyplot as plt

def plot_metrics(precision, mrr, map_, ndcg, label="FAISS", k=10):
    metrics = [precision, mrr, map_, ndcg]
    labels = ["Precision@{}".format(k), "MRR", "MAP", "nDCG@{}".format(k)]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, metrics, color="skyblue", edgecolor="black")
    plt.ylim(0, 1.0)
    plt.title(f"Evaluation Metrics for {label}")
    plt.ylabel("Score (0 to 1)")

    # Add score labels on top of bars
    for bar, metric in zip(bars, metrics):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{metric:.2f}", ha="center", va="bottom")

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

print("\nEnter your search query (type 'exit' to quit):")
while True:
    query = input("\nSearch: ").strip()
    if query.lower() == "exit":
        print("Goodbye!")
        break
    if not query:
        print("Please enter a non-empty query.")
        continue

    normalized_query = normalize_text_for_faiss(query)
    query_embedding = model.encode([normalized_query])
    D, I = index.search(np.array(query_embedding), k)

    print(f"\nTop {k} results for query: '{query}':\n")
    for rank, idx in enumerate(I[0]):
        line = indexed_lines[idx]
        print(f"{rank+1}. [ID {idx}] Season {line['season']} Episode {line['episode']}")
        print(f"   Scene: {line['scene']}")
        print(f"   {line['speaker'] or 'NARRATION'}: {line['clean_text']}")
        if line['actions']:
            print(f"   Actions: {'; '.join(line['actions'])}")
        print("-" * 60)

#first try for evaluation with queries file
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

def evaluate_faiss(queries: dict, model, index, normalize_fn, k=10):
    precision_scores = []
    mrr_scores = []
    map_scores = []
    ndcg_scores = []

    print(f"Evaluating FAISS index with top-{k} ranking metrics...\n")

    for query, relevant_ids in queries.items():
        normalized_query = normalize_fn(query)
        query_embedding = model.encode([normalized_query])
        D, I = index.search(np.array(query_embedding), k)

        retrieved_ids = list(I[0])
        if len(retrieved_ids) < k:
    # Pad with -1 or dummy IDs that are guaranteed not to be relevant
            retrieved_ids += [-1] * (k - len(retrieved_ids))


        # Metrics
        prec = precision_at_k(retrieved_ids, relevant_ids, k)
        mrr = reciprocal_rank(retrieved_ids, relevant_ids)
        ap = average_precision(retrieved_ids, relevant_ids)
        ndcg = ndcg_at_k(retrieved_ids, relevant_ids, k)

        print(f"Query: {query}")
        print(f"Expected: {relevant_ids}")
        print(f"Retrieved: {retrieved_ids}")
        print(f"Precision@{k}: {prec:.2f} | MRR: {mrr:.2f} | MAP: {ap:.2f} | nDCG@{k}: {ndcg:.2f}")
        print("-" * 60)

        precision_scores.append(prec)
        mrr_scores.append(mrr)
        map_scores.append(ap)
        ndcg_scores.append(ndcg)

    # Summary
    print("\nAVERAGE METRICS:")
    print(f"Precision@{k}: {np.mean(precision_scores):.3f}")
    print(f"MRR:            {np.mean(mrr_scores):.3f}")
    print(f"MAP:            {np.mean(map_scores):.3f}")
    print(f"nDCG@{k}:        {np.mean(ndcg_scores):.3f}")

    plot_metrics(np.mean(precision_scores), np.mean(mrr_scores),
                 np.mean(map_scores), np.mean(ndcg_scores), label="FAISS", k=k)

evaluate_faiss(queries, model, index, normalize_text_for_faiss, k)
