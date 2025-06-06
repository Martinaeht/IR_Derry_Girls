import os
import re
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.metrics.pairwise import cosine_similarity

# Load SPLADE model and tokenizer
model_name = "naver/splade-cocondenser-ensembledistil"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Normalize text
def normalize_text(text: str) -> str:
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

# Load and parse the script
def load_and_parse_script(file_path):
    season_pattern = re.compile(r"^SEASON\s+(\d+)", re.IGNORECASE)
    episode_pattern = re.compile(r"^EPISODE\s+(\d+)", re.IGNORECASE)
    scene_pattern = re.compile(r"^\[(.*)]$")
    speaker_pattern = re.compile(r"^([A-Za-z0-9 ']+):")
    action_pattern = re.compile(r"\((.*?)\)")

    with open(file_path, "r", encoding="windows-1252") as file:
        raw_lines = file.readlines()

    parsed_lines = []
    scene = season = episode = None

    for line_number, raw_line in enumerate(raw_lines):
        line = raw_line.strip()
        if not line:
            continue

        season_match = season_pattern.match(line)
        if season_match:
            season = int(season_match.group(1))
            continue

        episode_match = episode_pattern.match(line)
        if episode_match:
            episode = int(episode_match.group(1))
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
            dialogue = line[len(speaker) + 1:].strip()
        else:
            speaker = None
            dialogue = line

        actions = action_pattern.findall(dialogue)
        clean_text = action_pattern.sub("", dialogue).strip()

        if clean_text:
            parsed_lines.append({
                "id": len(parsed_lines),
                "line_number": line_number,
                "season": season,
                "episode": episode,
                "scene": scene,
                "speaker": speaker,
                "actions": actions,
                "clean_text": clean_text,
                "normalized_text": normalize_text(clean_text)
            })

    return parsed_lines

# Build sparse matrix index
def build_index(sparse_vectors):
    return csr_matrix(sparse_vectors)

# Search function using cosine similarity on normalized_text
def search(query, index, metadata, top_k=5):
    query_vector = encode_splade([normalize_text(query)])
    similarities = cosine_similarity(query_vector, index).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [(metadata[i], similarities[i]) for i in top_indices]

# Main CLI loop
def main():
    script_path = "/home/mlt_ml3/IR_Derry_Girls/Dataset/DERRY-GIRLS-SCRIPT.txt"
    index_path = "/home/mlt_ml3/IR_Derry_Girls/py_Files/SPLADE/splade_index.npz"
    metadata_path = "/home/mlt_ml3/IR_Derry_Girls/py_Files/SPLADE/metadata.json"

    if os.path.exists(index_path) and os.path.exists(metadata_path):
        print("Loading existing index and metadata...")
        index = load_npz(index_path)
        with open(metadata_path, 'r') as f:
            parsed_lines = json.load(f)
    else:
        print("Parsing script and building new index...")
        parsed_lines = load_and_parse_script(script_path)
        documents = [line["normalized_text"] for line in parsed_lines if line.get("normalized_text")]
        sparse_vectors = encode_splade(documents)
        index = build_index(sparse_vectors)
        save_npz(index_path, index)
        with open(metadata_path, 'w') as f:
            json.dump(parsed_lines, f)

    print("SPLADE IR system ready. Type your query or 'exit' to quit.")
    while True:
        query = input("\nSearch: ").strip()
        if query.lower() == "exit":
            break
        results = search(query, index, [line for line in parsed_lines if line.get("normalized_text")])
        for rank, (meta, score) in enumerate(results, start=1):
            print(f"\n{rank}. [Score: {score:.4f}] Season {meta['season']} Episode {meta['episode']}")
            print(f"Scene: {meta['scene']}")
            print(f"{meta['speaker'] or 'NARRATION'}: {meta['clean_text']}")
            if meta['actions']:
                print(f"Actions: {'; '.join(meta['actions'])}")
            print("-" * 50)

if __name__ == "__main__":
    main()
