from splade_relu_log import search
from scipy.sparse import load_npz
import json

def interactive_query():
    index_path = "/home/mlt_ml3/IR_Derry_Girls/py_Files/SPLADE/splade_index_relu.npz"
    metadata_path = "/home/mlt_ml3/IR_Derry_Girls/py_Files/SPLADE/metadata_relu.json"

    index = load_npz(index_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print("SPLADE IR system ready. Type your query or 'exit' to quit.")
    while True:
        query = input("\nSearch: ").strip()
        if query.lower() == "exit":
            break

        print(f"\nSearching for: '{query}'")
        results = search(query, index, metadata)

        for rank, (meta, score) in enumerate(results, start=1):
            print(f"\n{rank}. [Score: {score:.4f}] ID: {meta['id']} Season {meta['season']} Episode {meta['episode']}")
            print(f"Scene: {meta['scene']}")
            print(f"{meta['speaker']}: {meta['clean_text']}")
            if meta['actions']:
                print(f"Actions: {'; '.join(meta['actions'])}")
            print("-" * 50)

if __name__ == "__main__":
    interactive_query()
