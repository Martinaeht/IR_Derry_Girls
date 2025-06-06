#BM25 with stemming and evaluation
import re
import pickle
import unicodedata
import nltk
import numpy as np
from pprint import pprint
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from queries import queries  
nltk.download('punkt')
stemmer = PorterStemmer()

def normalize_string(text: str) -> str:
    text = text.lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)


def load_script(file_path):
    with open(file_path, "r", encoding="windows-1252") as file:
        return file.readlines()


def parse_script(raw_lines):
    scene_pattern = re.compile(r"^\[(.*)\]$")
    speaker_pattern = re.compile(r"^([A-Za-z0-9 ']+):")
    action_pattern = re.compile(r"\((.*?)\)")

    parsed_lines = []
    scene = None
    season = episode = None

    for line_number, raw_line in enumerate(raw_lines):
        line = raw_line.strip()
        if not line:
            continue

        if line.upper().startswith("SEASON"):  
            season_match = re.search(r"\d+", line)
            if season_match:
                season = int(season_match.group())
                print(f"Found Season {season} at line {line_number}")
            else:
                print(f"Warning: SEASON line without number at line {line_number}: '{line}'")
            continue

        if line.upper().startswith("EPISODE"):
            episode_match = re.search(r"\d+", line)
            if episode_match:
                episode = int(episode_match.group())
                print(f"Found Episode {episode} at line {line_number}")
            else:
                print(f"Warning: EPISODE line without number at line {line_number}: '{line}'")
            continue

        scene_match = scene_pattern.match(line)
        if scene_match:
            scene = scene_match.group(1).strip()
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
    return parsed_lines


def build_bm25_index(parsed_lines):
    docs = [line["clean_text"] for line in parsed_lines if line["clean_text"]]
    tokenized = [normalize_string(doc).split() for doc in docs]
    bm25 = BM25Okapi(tokenized)
    return bm25, docs


def save_index_and_metadata(bm25, docs, metadata, index_path):
    with open(index_path, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "docs": docs,
            "metadata": metadata,
        }, f)
    print(f"BM25 index and metadata saved to {index_path}")


def load_index_and_metadata(index_path):
    with open(index_path, "rb") as f:
        data = pickle.load(f)
    print(f"BM25 index and metadata loaded from {index_path}")
    return data["bm25"], data["docs"], data["metadata"]


def precision_at_k(retrieved, relevant, k):
    """Calculate precision@k"""
    if k == 0:
        return 0.0
    retrieved_at_k = retrieved[:k]
    relevant_retrieved = len(set(retrieved_at_k) & set(relevant))
    return relevant_retrieved / k


def dcg_at_k(retrieved, relevant, k):
    """Calculate DCG@k"""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
    return dcg


def ndcg_at_k(retrieved, relevant, k):
    """Calculate NDCG@k"""
    dcg = dcg_at_k(retrieved, relevant, k)
    # IDCG: best possible DCG if we had perfect ranking
    idcg = dcg_at_k(relevant, relevant, min(k, len(relevant)))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(retrieved, relevant):
    if not relevant:
        return 0.0
    
    relevant_set = set(relevant)
    ap = 0.0
    relevant_count = 0
    
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant_set:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            ap += precision_at_i
    
    return ap / len(relevant) if relevant else 0.0


def reciprocal_rank(retrieved, relevant):
    relevant_set = set(relevant)
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_queries(bm25, docs, metadata, queries, k_values=[5, 8]):
    all_precisions = {k: [] for k in k_values}
    all_ndcg = {k: [] for k in k_values}
    all_ap = []
    all_rr = []
    
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    for query_text, relevant_indices in queries.items():
        print(f"\nQuery: '{query_text}'")
        print(f"Relevant documents: {relevant_indices}")
        
        tokenized_query = normalize_string(query_text).split()
        scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:8]
        
        print(f"Top 8 retrieved: {top_indices[:8]}")
        
        for k in k_values:
            prec_k = precision_at_k(top_indices, relevant_indices, k)
            ndcg_k = ndcg_at_k(top_indices, relevant_indices, k)
            all_precisions[k].append(prec_k)
            all_ndcg[k].append(ndcg_k)
            print(f"  Precision@{k}: {prec_k:.3f}")
            print(f"  NDCG@{k}: {ndcg_k:.3f}")
        
        ap = average_precision(top_indices, relevant_indices)
        rr = reciprocal_rank(top_indices, relevant_indices)
        all_ap.append(ap)
        all_rr.append(rr)
        
        print(f"  Average Precision: {ap:.3f}")
        print(f"  Reciprocal Rank: {rr:.3f}")
        print("-" * 50)
   
    print("\n" + "=" * 80)
    print("OVERALL METRICS")
    print("=" * 80)
    
    for k in k_values:
        mean_prec_k = np.mean(all_precisions[k])
        mean_ndcg_k = np.mean(all_ndcg[k])
        print(f"Mean Precision@{k}: {mean_prec_k:.3f}")
        print(f"Mean NDCG@{k}: {mean_ndcg_k:.3f}")
    
    map_score = np.mean(all_ap)
    mrr_score = np.mean(all_rr)
    
    print(f"Mean Average Precision (MAP): {map_score:.3f}")
    print(f"Mean Reciprocal Rank (MRR): {mrr_score:.3f}")
    
    return {
        'precision_at_k': {k: np.mean(all_precisions[k]) for k in k_values},
        'ndcg_at_k': {k: np.mean(all_ndcg[k]) for k in k_values},
        'map': map_score,
        'mrr': mrr_score
    }


def search_loop(bm25, docs, metadata, top_n=8):
    """Interactive search loop"""
    print("Enter your search query or type 'exit' to quit.")
    while True:
        query = input("\nSearch: ").strip()
        if query.lower() == "exit":
            break
        if not query:
            print("Please enter a non-empty query.")
            continue

        tokenized_query = normalize_string(query).split()
        scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]

        print(f"\nTop {top_n} results for query: '{query}'\n")
        for rank, idx in enumerate(top_indices, start=1):
            score = scores[idx]
            doc_text = docs[idx]
            meta = metadata[idx]

            speaker = meta["speaker"]
            print(f"{rank}. [Index: {idx}][Score: {score:.2f}] Season {meta['season']} Episode {meta['episode']}")
            print(f"Scene: {meta['scene']}")
            print(f"{speaker}: {doc_text}")
            if meta['actions']:
                print(f"Actions: {'; '.join(meta['actions'])}")
            print("-" * 50)


if __name__ == "__main__":
    file_path = "/home/mlt_ml3/IR_Derry_Girls/Dataset/DERRY-GIRLS-SCRIPT.txt" 
    index_file = "/home/mlt_ml3/IR_Derry_Girls/Dataset/bm25_full_index_stemmed.pkl" 

    raw_lines = load_script(file_path)
    parsed_lines = parse_script(raw_lines)

    metadata = [line for line in parsed_lines if line["clean_text"]]

    print("Building BM25 index from scratch...")
    bm25, docs = build_bm25_index(parsed_lines)
    save_index_and_metadata(bm25, docs, metadata, index_file)

    print("Running evaluation on predefined queries...")
    metrics = evaluate_queries(bm25, docs, metadata, queries, k_values=[5, 8])

    print("\n" + "=" * 80)
    user_input = input("Do you want to run interactive search? (y/n): ").strip().lower()
    if user_input == 'y':
        search_loop(bm25, docs, metadata)