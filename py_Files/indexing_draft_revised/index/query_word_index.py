from utils.text_utils import Text

def query_word_index(query, index):
    """Search the token index for any matching words in the query."""
    query_tokens = Text([query]).tokenize()
    if not query_tokens:
        return []

    query_tokens_set = set(token.lower() for token in query_tokens)
    results = []

    for token, occurrences in index.items():
        if token.lower() in query_tokens_set:
            results.extend(format_occurrences(occurrences))

    return results


def format_occurrences(occurrences):
    """Helper function to format index entries into result dictionaries."""
    return [
        {
            "sentence_id": f'S{entry["season"]},E{entry["episode"]}, LineNr.{entry["line_idx"]}',
            "sentence": entry.get("text", ""),
            "speaker": entry.get("speaker")
        }
        for entry in occurrences
    ]


