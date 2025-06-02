from collections import defaultdict
from utils.text_utils import Text

def build_token_index(episodes):
    token_index = defaultdict(list)
    for episode in episodes:
        for idx, line in enumerate(episode.lines):
            if not line["text"]:
                continue
            tokens = Text([line["text"]]).tokenize()
            for token in tokens:
                token_index[token].append({
                    "season": episode.season,
                    "episode": episode.episode,
                    "line_idx": idx,
                    "speaker": line.get("speaker"),
                    "text": line["text"]
                })
    return token_index

def build_sentence_index(episodes_data):
    sentence_index = defaultdict(list)
    for episode in episodes_data:
        for idx, line in enumerate(episode["lines"]):
            if not line["text"]:
                continue
            tokens = tuple(Text([line["text"]]).tokenize())
            sentence_index[tokens].append({
                "season": episode["season"],
                "episode": episode["episode"],
                "line_idx": idx,
                "speaker": line.get("speaker"),
                "original_sentence": line["text"]
            })
    return sentence_index
