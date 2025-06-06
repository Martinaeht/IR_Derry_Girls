import os
import re
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.metrics.pairwise import cosine_similarity

model_name = "naver/splade-cocondenser-ensembledistil"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

def normalize_text(text: str) -> str:
    text = text.lower()
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def encode_splade(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        relu_log = torch.log(1 + torch.relu(logits))
        weighted_log = relu_log * inputs.attention_mask.unsqueeze(-1)
        sparse_vectors = torch.sum(weighted_log, dim=1)
    return sparse_vectors.cpu().numpy()

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

def build_index(sparse_vectors):
    return csr_matrix(sparse_vectors)

def search(query, index, metadata, top_k=5):
    query_vector = encode_splade([normalize_text(query)])
    similarities = cosine_similarity(query_vector, index).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [(metadata[i], similarities[i]) for i in top_indices]

def create_evaluation_metadata(parsed_lines):
    filtered_lines = [line for line in parsed_lines if line.get("normalized_text")]
    
    evaluation_metadata = []
    for i, line in enumerate(filtered_lines):
        eval_line = line.copy()
        eval_line["eval_index"] = i  
        evaluation_metadata.append(eval_line)
    
    return evaluation_metadata


def main():
    script_path = "/home/mlt_ml3/IR_Derry_Girls/Dataset/DERRY-GIRLS-SCRIPT.txt"
    index_path = "/home/mlt_ml3/IR_Derry_Girls/py_Files/SPLADE/splade_index_relu.npz"
    metadata_path = "/home/mlt_ml3/IR_Derry_Girls/py_Files/SPLADE/metadata_relu.json"

    if os.path.exists(index_path) and os.path.exists(metadata_path):
        print("Loading existing index and metadata...")
        index = load_npz(index_path)
        with open(metadata_path, 'r') as f:
            evaluation_metadata = json.load(f)
    else:
        print("Parsing script and building new index with SPLADE encoding...")
        parsed_lines = load_and_parse_script(script_path)
        
        evaluation_metadata = create_evaluation_metadata(parsed_lines)
        
        documents = [line["normalized_text"] for line in evaluation_metadata]
        
        print(f"Encoding {len(documents)} documents...")
        
        batch_size = 32
        all_vectors = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            batch_vectors = encode_splade(batch)
            all_vectors.append(batch_vectors)
        
        sparse_vectors = np.vstack(all_vectors)
        print(f"Created sparse vectors with shape: {sparse_vectors.shape}")
        
        index = build_index(sparse_vectors)
        save_npz(index_path, index)
        with open(metadata_path, 'w') as f:
            json.dump(evaluation_metadata, f, indent=2)
        
        print(f"Index saved with {index.shape[0]} documents and {index.shape[1]} features")

    print("SPLADE IR system ready. Type your query or 'exit' to quit.")
    while True:
        query = input("\nSearch: ").strip()
        if query.lower() == "exit":
            break
            
        print(f"\nSearching for: '{query}'")
        results = search(query, index, evaluation_metadata)
        
        for rank, (meta, score) in enumerate(results, start=1):
            print(f"\n{rank}. [Score: {score:.4f}] ID: {meta['id']} Season {meta['season']} Episode {meta['episode']}")
            print(f"Scene: {meta['scene']}")
            print(f"{meta['speaker']}: {meta['clean_text']}")
            if meta['actions']:
                print(f"Actions: {'; '.join(meta['actions'])}")
            print("-" * 50)

if __name__ == "__main__":
    main()

