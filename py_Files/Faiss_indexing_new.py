#FAISS Indexing new
from pprint import pprint
import re
import json
import faiss
import os
import numpy as np
import unicodedata
from sentence_transformers import SentenceTransformer

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

model = SentenceTransformer("all-MiniLM-L6-v2")
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

k = 7

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

