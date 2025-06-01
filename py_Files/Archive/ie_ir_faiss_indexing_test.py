#FAISS Indexing

from pprint import pprint
import re
import json
import faiss
import os
import numpy as np
from sentence_transformers import SentenceTransformer

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

    # Check for season
    season_match = season_pattern.match(line)
    if season_match:
        season = int(season_match.group(1))
        continue  # Usually these lines are not dialogue, so we skip adding them to parsed_lines

    # Check for episode
    episode_match = episode_pattern.match(line)
    if episode_match:
        episode = int(episode_match.group(1))
        continue

    # Check for scene
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

    # Check for speaker
    speaker_match = speaker_pattern.match(line)
    if speaker_match:
        speaker = speaker_match.group(1).strip()
        dialogue_with_actions = line[len(speaker)+1:].strip()
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

# 3. Prepare documents for embedding
documents = [line["clean_text"] for line in parsed_lines if line["clean_text"]]

# 4. Generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
document_embeddings = model.encode(documents)

# 5. Create FAISS index and add embeddings
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(document_embeddings))

print(f"Number of items in FAISS index: {index.ntotal}")

# 6. Map indexed documents back to parsed_lines
indexed_lines = [line for line in parsed_lines if line["clean_text"]]

# 7. Save FAISS index and metadata
index_path = "/home/mlt_ml3/IR_Derry_Girls/Dataset/derry_girls_faiss.index"
metadata_path = "/home/mlt_ml3/IR_Derry_Girls/Dataset/derry_girls_metadata.json"

with open(index_path, "wb") as f:
    faiss.write_index(index, f)

with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(indexed_lines, f, ensure_ascii=False, indent=2)

print(f"Saved FAISS index to: {index_path}")
print(f"Saved metadata to: {metadata_path}")

#Load FAISS index and metadata
#index = faiss.read_index("index_path")
#with open("derry_girls_metadata.json", "r", encoding="utf-8") as f:
 #    indexed_lines = json.load(f)

print("10 First Indexed lines with FAISS IDs:")
for idx, line in enumerate(parsed_lines[:10]):
    print(f"[ID {idx}] Season: {line['season']}, Episode: {line['episode']}, Scene: {line['scene']}, Speaker: {line['speaker']}, Actions: {line['actions']}")
    print(f"   Text: {line['clean_text']}\n")

# 10. Example similarity search
query = "Protestants and Catholics arguments"
query_embedding = model.encode([query])
k = 5
D, I = index.search(np.array(query_embedding), k)

print(f"\nTop {k} similar lines for query: '{query}'")
for rank, idx in enumerate(I[0]):
    line = indexed_lines[idx]
    print(f"Rank {rank+1}: [ID {idx}] Season: {line['season']}, Episode: {line['episode']}, Scene: {line['scene']}, Speaker: {line['speaker']}")
    print(f"   Text: {line['clean_text']}\n")