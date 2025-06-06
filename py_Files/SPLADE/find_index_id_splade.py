#from py_Files.SPLADE.splade_relu_log import normalize_text, encode_splade, load_and_parse_script
from splade_relu_log import normalize_text, encode_splade, load_and_parse_script
import os
import json 

metadata_path = "/home/mlt_ml3/IR_Derry_Girls/py_Files/SPLADE/metadata_relu.json"
with open(metadata_path, 'r') as f:
    parsed_lines = json.load(f)


def find_ids_by_substring(sentences, parsed_lines):
    results = {}
    for sentence in sentences:
        norm = normalize_text(sentence)
        matches = []
        for entry in parsed_lines:
            if "normalized_text" in entry and norm in entry["normalized_text"]:
                matches.append((entry["id"], entry["normalized_text"]))
        results[sentence] = matches
    return results


'''
def find_index_by_sentence(sentence, parsed_lines):
    norm = normalize_text(sentence)
    for entry in parsed_lines:
        if entry.get("normalized_text") == norm:
            return entry["id"]
    return None

'''

sentences = ["Do you mean when you shacked up with a slutty hairdresser, but then she dumped you?", 
             "Not all because of us, Orla. I mean, a bit because of us. But mostly because it turns out he had a connection with one of the colourists in Hair and Flair, who does our Sarah's forwards, by the way. And apparently she's a dirty tramp.", 
             "And you know, it became more challenging, when I met this amazing girl."
]

matches = find_ids_by_substring(sentences, parsed_lines)
for sentence, ids in matches.items():
    print(f"\nSentence: '{sentence}'")
    if ids:
        for id_, text in ids:
            print(f"ID: {id_}")
    else:
        print("No match found.")


# print(find_index_by_substring("How long does it take to defuse a fecking bomb?", parsed_lines))