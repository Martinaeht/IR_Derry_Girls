import os
import pickle
from pprint import pprint

from parser.script_parser import parse_episodes
from index.builder import build_token_index, build_sentence_index
from index.query_sentence_index import query_sentence_index
from index.query_word_index import query_word_index  
from utils.text_utils import Text
from episode import Episode  

# Configuration 
script_path = "/home/mlt_ml3/IR_Derry_Girls/py_Files/indexing_draft_revised/data/DERRY-GIRLS-SCRIPT.txt"
output_dir = "/home/mlt_ml3/IR_Derry_Girls/py_Files/indexing_draft_revised/index"
os.makedirs(output_dir, exist_ok=True)

# Load and parse script 
with open(script_path, "r", encoding="windows-1252") as file:
    file_lines = file.readlines()

print("Preview of the script content:")
pprint(file_lines[:5])

episodes_data = parse_episodes(file_lines)
episodes = [Episode(ep) for ep in episodes_data]

# Build and save indexes
sentence_index = build_sentence_index(episodes_data)
token_index = build_token_index(episodes)

with open(os.path.join(output_dir, "sentence_index.pkl"), "wb") as f:
    pickle.dump(sentence_index, f)

with open(os.path.join(output_dir, "token_index.pkl"), "wb") as f:
    pickle.dump(token_index, f)

print("Indexes saved successfully.")


while True:
    user_query = input("\nEnter your search query (or type 'exit' to quit): ").strip()
    if user_query.lower() == "exit":
        print("Exiting search.")
        break
    if not user_query:
        print("Please enter a non-empty query.")
        continue

    print(f"\nResults for sentence query: '{user_query}'")
    sentence_results = query_sentence_index(user_query, sentence_index)
    pprint(sentence_results[:5])

    print(f"\nResults for word query: '{user_query}'")
    word_results = query_word_index(user_query, token_index)
    pprint(word_results[:5])


'''
# Example queries 
query_1 = "chip bags"
query_2 = "What happened?"

print(f"\nResults for sentence query: '{query_1}'")
sentence_results = query_sentence_index(query_1, sentence_index)
pprint(sentence_results[:5])

print(f"\nResults for sentence query: '{query_2}'")
sentence_results = query_sentence_index(query_2, sentence_index)
pprint(sentence_results[:5])

print(f"\nResults for word query: '{query_1}'")
word_results = query_word_index(query_1, token_index)
pprint(word_results[:5])

print(f"\nResults for word query: '{query_2}'")
word_results = query_word_index(query_2, token_index)
pprint(word_results[:5])
'''