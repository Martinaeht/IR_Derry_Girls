from _splade import normalize_text, encode_splade, load_and_parse_script
import os
import json 

metadata_path = "/home/mlt_ml3/IR_Derry_Girls/py_Files/SPLADE/metadata.json"
with open(metadata_path, 'r') as f:
    parsed_lines = json.load(f)

def find_index_by_sentence(sentence, parsed_lines):
    norm = normalize_text(sentence)
    for entry in parsed_lines:
        if entry.get("normalized_text") == norm:
            return entry["id"]
    return None

print(find_index_by_sentence("The mural on our house, the spray paint. It was Emmett, and I can prove it.", parsed_lines))