from utils.text_utils import Text

def query_sentence_index(query, index):
    query_tokens = tuple(Text([query.strip()]).tokenize())
    query_tokens_set = set(query_tokens)

    results = []
    for token_tuple, occurrences in index.items():
        if query_tokens_set.issubset(token_tuple):
            for entry in occurrences:
                results.append({
                    "sentence_id": (entry["season"], entry["episode"], entry["line_idx"]),
                    "sentence": entry["original_sentence"],
                    "speaker": entry["speaker"]
                })
    return results
