import re
import pickle
import unicodedata
from pprint import pprint
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')


def normalize_string(text: str) -> str:
    # Lowercase
    text = text.lower()
    # Remove accents
    text = ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')
    # Remove punctuation (keep letters, digits, spaces)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


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

        if line.upper().startswith("SEASON"):  # ← Changed from "SEASON" in line.upper()
            season_match = re.search(r"\d+", line)  # ← Store the match result
            if season_match:  # ← Check if match exists before calling .group()
                season = int(season_match.group())
                print(f"Found Season {season} at line {line_number}")  # ← Added debug
            else:
                print(f"Warning: SEASON line without number at line {line_number}: '{line}'")  # ← Added warning
            continue

        if line.upper().startswith("EPISODE"):  # ← Changed from "EPISODE" in line.upper()
            episode_match = re.search(r"\d+", line)  # ← Store the match result
            if episode_match:  # ← Check if match exists before calling .group()
                episode = int(episode_match.group())
                print(f"Found Episode {episode} at line {line_number}")  # ← Added debug
            else:
                print(f"Warning: EPISODE line without number at line {line_number}: '{line}'")  # ← Added warning
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
    # Only index lines with clean_text (dialogue lines)
    docs = [line["clean_text"] for line in parsed_lines if line["clean_text"]]
    # Normalize and tokenize using professor's normalize_string
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


def search_loop(bm25, docs, metadata, top_n=5):
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

            speaker = meta["speaker"] or "NARRATION"
            print(f"{rank}. [Score: {score:.2f}] Season {meta['season']} Episode {meta['episode']}")
            print(f"Scene: {meta['scene']}")
            print(f"{speaker}: {doc_text}")
            if meta['actions']:
                print(f"Actions: {'; '.join(meta['actions'])}")
            print("-" * 50)


if __name__ == "__main__":
    file_path = "/home/mlt_ml3/IR_Derry_Girls/Dataset/DERRY-GIRLS-SCRIPT.txt"  # your script filename
    index_file = "/home/mlt_ml3/IR_Derry_Girls/Dataset/bm25_full_index.pkl"     # where to save/load the index

    raw_lines = load_script(file_path)
    parsed_lines = parse_script(raw_lines)

    # Filter metadata to only lines indexed (those with clean_text)
    metadata = [line for line in parsed_lines if line["clean_text"]]

    try:
        bm25, docs, metadata = load_index_and_metadata(index_file)
    except FileNotFoundError:
        print("Index file not found, building BM25 index from scratch...")
        bm25, docs = build_bm25_index(parsed_lines)
        save_index_and_metadata(bm25, docs, metadata, index_file)

    search_loop(bm25, docs, metadata)
