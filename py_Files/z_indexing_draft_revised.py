
import os
import pickle
import requests
import re
from collections import defaultdict
from pprint import pprint


# Work with Dataset

file_path = "/home/mlt_ml3/IR_Derry_Girls/Dataset/DERRY-GIRLS-SCRIPT.txt"

with open(file_path, "r", encoding="windows-1252") as file:
    file_lines = file.readlines()  
    #file_lines = [line.lower() for line in file_lines]  #added

# Print a preview of the script content
print("Preview of the script content:")
pprint(file_lines[:5])  

#text processing class for tokenizing and working with text

class Text:
    def __init__(self, lines: list[str]):
        self.lines = lines

    def tokenize(self) -> list[str]:
        tokens = []
        for line in self.lines:
          clean_line = re.sub(r"[^\w\s]", "", line.lower()) # Remove punctuation and convert to lowercase
          words = clean_line.split()  # Split into words
          tokens.extend(words)
        return tokens


#still open
#Remove metadata or non-dialogue text, e.g. sound description

# Episode class representing a single episode

class Episode(Text):
    def __init__(self, episode_data):
        self.season = episode_data["season"]
        self.episode = episode_data["episode"]
        self.id = f"S{self.season}E{self.episode}"  # Unique identifier for the episode
        self.title = f"Season {self.season}, Episode {self.episode}" # Human-readable title
        self.lines = episode_data["lines"]
        # Flatten all text lines
        #lines = [entry["text"] for entry in episode_data["lines"] if entry["text"]]
        #super().__init__(lines)

    def __str__(self):
        return f"{self.title}\n" + "\n".join(self.lines)

    def __repr__(self):
        return f"{self.title}"

# Parse the script content into structured episodes.

def parse_episodes(file_lines):
    episodes = []
    current_lines = []
    season = episode = None

    for line in file_lines:
        line = line.strip() # Remove leading/trailing whitespace

        if not line:  # Skip empty lines
            continue

        # Detect season header
        if line.upper().startswith("SEASON"):
            match = re.search(r"\d+", line)
            if match:
                season = int(match.group())

        # Detect episode header and save previous episode's data
        elif line.upper().startswith("EPISODE"):
            # Save previous episode
            if current_lines: # If there are lines from the previous episode, save them
                episodes.append({
                    "season": season,
                    "episode": episode,
                    "lines": current_lines,
                })
                current_lines = []  # Reset for the new episode

            match = re.search(r"\d+", line)
            if match:
                episode = int(match.group())

        # Detect speaker line
        elif ":" in line and not line.startswith("(") and not line.startswith("["):
            parts = line.split(":", 1)
            speaker = parts[0].strip()
            text = parts[1].strip()
            current_lines.append({
                "speaker": speaker,
                "text": text
            })

        # Fallback (narration, noise, untagged text)
        else:
            current_lines.append({
                "speaker": None,
                "text": line
            })

    # Don't forget to save the last episode
    if current_lines:
        episodes.append({
            "season": season,
            "episode": episode,
            "lines": current_lines
        })

    return episodes

# Parse the script into structured episodes

episodes_data = parse_episodes(file_lines)
episodes = [Episode(ep) for ep in episodes_data]  # Convert dicts to Episode objects

# added tokenization and speaker here:

def build_sentence_index(episodes_data):
    sentence_index = defaultdict(list)

    for episode in episodes_data:
        season_num = episode["season"]
        episode_num = episode["episode"]

        for line_idx, line in enumerate(episode["lines"]):
          if line["text"]:
                sentence = line["text"].strip()
                tokens = tuple(Text([sentence]).tokenize())

                speaker = line.get("speaker", None)

                sentence_index[tokens].append({
                    "season": season_num,
                    "episode": episode_num,
                    "line_idx": line_idx,
                    "speaker": speaker,
                    "original_sentence": sentence
                })

    return sentence_index

# Token-based index
def build_token_index(episodes_data):
    token_index = defaultdict(list)
    for episode in episodes:
        for idx, line in enumerate(episode.lines):
            if line["text"]:
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

sentence_index = build_sentence_index(episodes_data)

# Save the sentence-level index to a pickle file
with open("/content/drive/MyDrive/IR_Project/sentence_index.pkl", "wb") as f:
    pickle.dump(sentence_index, f)

print("Sentence-level index has been saved to 'sentence_index.pkl'.")

token_index = build_token_index(episodes_data)
with open("/content/drive/MyDrive/IR_Project/token_index.pkl", "wb") as f:
    pickle.dump(token_index, f)

with open("/content/drive/MyDrive/IR_Project/sentence_index.pkl", "rb") as f:
    sentence_index = pickle.load(f)

def query_sentence_index(query, index):
    """Search for partial matches of a sentence or phrase in the index."""
    query = query.strip()
    query_tokens = tuple(Text([query]).tokenize())  # Tokenize the query just like the indexed sentences
    query_tokens_set = set(query_tokens)  # Use a set for partial matching

    results = []

    # Loop through the sentence index to check for partial matches
    for token_tuple, occurrences in index.items():
        # Check if all query tokens are in the sentence token tuple
        if query_tokens_set.issubset(token_tuple):
            for entry in occurrences:
                results.append({
                    "sentence_id": (entry["season"], entry["episode"], entry["line_idx"]),
                    "sentence": entry["original_sentence"],
                    "speaker": entry["speaker"]
                })

    return results

#attempt
def query_word_index(query, index):
    # List of common question words
    question_words = ["what", "who", "when", "where", "how", "why", "which", "is", "are", "did", "does", "can", "will"]

    # Clean and tokenize the query
    query_tokens = Text([query]).tokenize()
    first_word = query_tokens[0].lower() if query_tokens else ""

    # Refined check for a question (check if the first token is a common question word)
    is_question = first_word in question_words or query.endswith("?")

    results = []
    results_counter = 0 # added

    if is_question:
        # Remove question words from the query tokens
        #filtered_query_tokens = [token for token in query_tokens if token not in question_words] #add?

        # If it's a question, search for all tokens in the query
        query_tokens_set = set(query_tokens)  # Convert query to a set of tokens
        #query_tokens_set = set(filtered_query_tokens)  # add?

        # Loop through the token index to check for matches with the query tokens
        for token, occurrences in index.items():
            if token in query_tokens_set:  # If token from index matches a query token
                for entry in occurrences:
                    # Format the sentence_id as requested
                    sentence_id = f'S{entry["season"]},E{entry["episode"]}, LineNr.{entry["line_idx"]}'
                    sentence_text = entry.get("text", "")  # Get the text or default to empty string if it doesn't exist
                    results.append({
                        "sentence_id": sentence_id,
                        "sentence": sentence_text,
                        "speaker": entry["speaker"]
                    })
                    #results_counter += 1 # add?
    else:
        # If it's not a question, assume it's a single word or phrase
        word = query.strip().lower()

        # Debug: Check for word-based search
        print(f"Searching for word: {word}")

        for token_tuple, occurrences in index.items():
            if word in token_tuple:  # Search for the exact word in the index
                for entry in occurrences:
                    sentence_id = f'S{entry["season"]},E{entry["episode"]}, LineNr.:{entry["line_idx"]}'
                    sentence_text = entry.get("text", "")
                    results.append({
                        "sentence_id": sentence_id,
                        "sentence": sentence_text,
                        "speaker": entry["speaker"]
                    })
    #print(f"Total Results Found: {results_counter}")  # add?
    return results  # add results_counter?

#example word query using the sentence index

query_word = "Mammy"
results = query_sentence_index(query_word, sentence_index)

if results:
    print(f"Occurrences of the word '{query_word}':")
    for result in results:
        sentence_id = result["sentence_id"]
        speaker = result["speaker"]
        sentence = result["sentence"]
        print(f"ID: Season {sentence_id[0]}, Episode {sentence_id[1]}, Line {sentence_id[2]}")
        print(f"Speaker: {speaker}")
        print(f"Sentence: {sentence}")
else:
    print(f"The word '{query_word}' was not found in any sentence.")

#query example using a partial sentence (sentence_index)
query = "Orla is reading "
#query = "What's the boy in Africa called?"
results = query_sentence_index(query, sentence_index)

if results:
    print(f"Occurrences of the partial sentence '{query}':")
    for occurrence in results:
        season, episode, line_idx = occurrence["sentence_id"]
        print(f"Season {season}, Episode {episode}, Line {line_idx}")
        print(f"Sentence: {occurrence['sentence']}")
        print(f"Speaker: {occurrence['speaker']}")
else:
    print(f"The partial sentence '{query}' was not found in the index.")

word = "college"
token_hits = query_word_index(word, token_index)
for r in token_hits:
    print(f"Token Match: {r['sentence']}, Speaker: {r['speaker']}, Sentence_ID: {r['sentence_id']}")

#word = "bomb"
#word = "detonate"
#word = "Troubles"
#word = "Troubles political"
#word = "class trip"
#word = "dick"
word = "fuck"


token_hits = query_word_index(word, token_index)

sorted_token_hits = sorted(token_hits, key=lambda r: r['sentence_id'])

# Print the sorted results
for r in sorted_token_hits:
    #print(f"Sentence_ID: {r['sentence_id']}, Sentence: {r['sentence']}")
    print(f"{r['sentence_id']}")

#query1 = "What's the name of the college?"
#query1 = "What show does Erin never miss?"
#query1 = "troubles, political"
#query1 = "show, Erin"
query1 = "bags of chips"

results1 = query_sentence_index(query1, sentence_index)

if results1:
    print(f"Occurrences of the query '{query1}':")
    for occurrence in results1:
        season, episode, line_idx = occurrence["sentence_id"]
        print(f"Season {season}, Episode {episode}, Line {line_idx}")
        print(f"Sentence: {occurrence['sentence']}")
        print(f"Speaker: {occurrence['speaker']}")
else:
    print(f"The partial sentence '{query1}' was not found in the index.")

# query_word_index (without question_words filter) --> Total Results Found: 2623

#word = "What's the name of the college?"
#word = "What show does Erin never miss?"

results2 = query_word_index(word, token_index) # add results_counter?
for r in results2:
    print(f"Speaker: {r['speaker']}, Token Match: {r['sentence']}, Sentence_ID: {r['sentence_id']}")

"""# query_word_index with question_words filter --> Total Results Found: 2623

word = "What's the name of the college?"
results2 = query_word_index(word, token_index) # add resulst_counter?
for r in results2:
    print(f"Speaker: {r['speaker']}, Token Match: {r['sentence']}, Sentence_ID: {r['sentence_id']}")"""

pprint(token_index)

for token, occurrences in token_index.items():
    print(f"Token: {token}")
    for entry in occurrences:
        print(entry)

"""#**Previous implementations we chose not to use**

# initially created these classes that we learnt about in Object Oriented Programming
#but we don't need them anymore, because of the functions
class Query(Text):
    def __init__(self, query: str):
        super().__init__([query])


class Index(dict[str, set[int]]):
    def __init__(self, scripts: list[Text]):
        super().__init__()
        self.scripts = scripts
        for script in scripts:
            self.add(script)

    def add(self, script: Text):
        for token in script.tokenize():
            if token not in self:
                self[token] = set()
            self[token].add(script.id)

    def search(self, query: Query) -> list[Episode]:
        result = None
        for token in query.tokenize():
            if token in self:
                token_docs = self[token]
            else:
                token_docs = set()
            if result is None:
                result = token_docs
            else:
                result = result.intersection(token_docs)

        if result:
            return [doc for doc in self.scripts if doc.id in result]

        else:
            return []

# Build an index where each sentence maps to its occurrences.

def build_sentence_index(data):
    sentence_index = defaultdict(list)

    for episode in data:
        season_num = episode["season"]
        episode_num = episode["episode"]

        for line_idx, line in enumerate(episode["lines"]):
            if line["text"]:
                sentence = line["text"].strip()
                #speaker = line.get("speaker", "Narrator/Unknown")  # Get speaker, default to "Narrator/Unknown"
                #sentence_index[sentence].append((season_num, episode_num, line_idx, speaker))
                sentence_index[sentence].append((season_num, episode_num, line_idx))


    return sentence_index


#token index v1

def build_token_index(episodes):
    token_index = defaultdict(list)

    for episode in episodes:
        season_num = episode["season"]
        episode_num = episode["episode"]

        for line_idx, line in enumerate(episode["lines"]):
            if line["text"]:
                sentence = line["text"].strip()
                tokens = Text([sentence]).tokenize()

                speaker = line.get("speaker", None)

                for token in tokens:
                    token_index[token].append({
                        "season": season_num,
                        "episode": episode_num,
                        "line_idx": line_idx,
                        "speaker": speaker,
                        "original_sentence": sentence
                    })

    return token_index

"""
"""
# Function to search for a word within sentences

# with open("/content/drive/MyDrive/IR_Project/sentence_index.pkl", "rb") as f:
    # sentence_index = pickle.load(f)

def search_word_in_sentences(word, index):
    """
    Search for a word in the indexed sentences and return sentences containing that word
    along with their IDs (season, episode, line index).
    """
    word = word.lower()  # Normalize the word for case-insensitive search
    results = []

    for sentence, occurrences in index.items():
        # Check if the word is present in the sentence (case-insensitive)
        if word in sentence.lower():
            for occurrence in occurrences:
                season, episode, line_idx = occurrence
                #season, episode, line_idx, speaker = occurrence

                speaker = None ### added from here
                for line in episodes_data:
                    if line["season"] == season and line["episode"] == episode:
                        speaker = (
                            line["lines"][line_idx]["speaker"]
                            if "speaker" in line["lines"][line_idx]
                            else None
                        )
                        break ### to here --> not needed if we use speaker directly in def build_sentence_index
                results.append({
                    "sentence_id": (season, episode, line_idx),
                    "sentence": sentence,
                    "speaker": speaker # added
                })
    return results


# Sentence-level search
# Searches for an exact match of the user's query as a sentence or phrase.

# Load the serialized sentence index
with open("/content/drive/MyDrive/IR_Project/sentence_index.pkl", "rb") as f:
    sentence_index = pickle.load(f)

def query_sentence_index(query, index):
    """Search for a sentence or phrase in the index."""
    query = query.strip()
    if query in index:
        return index[query]
    else:
        return []

query = "In Erin's bedroom. Orla is reading Erin's diary."
results = query_sentence_index(query, sentence_index)

if results:
    print(f"Occurrences of the sentence '{query}':")
    for occurrence in results:
        season, episode, line_idx = occurrence
        print(f"Season {season}, Episode {episode}, Line {line_idx}")
        #print(f"Season {season}, Episode {episode}, Speaker {speaker}, Line {line_idx}") #I added speaker here
else:
    print(f"The sentence '{query}' was not found in the index.")


def query_word_index(word, index):
    word = word.lower()
    results = []

    for token_tuple, occurrences in index.items():
        if word in token_tuple:  # Search within the tuple of tokens
          for entry in occurrences:
                sentence_text = entry.get("text", "")  # Get the text or default to empty string if it doesn't exist
                sentence_id = f'S{entry["season"]},E{entry["episode"]}, LineNr.{entry["line_idx"]}'
                results.append({
                    "sentence_id": sentence_id,
                    "sentence": sentence_text,
                    "speaker": entry["speaker"]
                })

    return results

#token based search with speaker identification
class TokenIndex:
    def __init__(self, episodes):
        self.index = defaultdict(list)
        self.build_index(episodes)

    def build_index(self, episodes):
        for episode in episodes:
            for line_idx, line in enumerate(episode.lines):
                tokens = line.split()
                for token in tokens:
                    self.index[token.lower()].append((episode.season, episode.episode, line_idx, line))

    def search_tokens(self, query):
        query_tokens = query.lower().split()  # Split query into tokens
        results = []

        for token in query_tokens:
            if token in self.index:
                results.extend(self.index[token])

        return results

# Example usage:
query = "Where is the school?"
results = token_index.search_tokens(query)

if results:
    print(f"Occurrences of the query '{query}':")
    for result in results:
        season, episode, line_idx, line = result
        print(f"Season {season}, Episode {episode}, Line {line_idx}: {line}")
else:
    print(f"No results found for the query '{query}'")

# Token based search
# Searches the token-level inverted index for episodes that contain all the tokens in the query.

index = Index(episodes)
query = Query("What's the college called?")
results = index.search(query)

#for result in results:
 # print(result)

for result in results:
    print(f"\n{result.title}")
    print("-" * 40)


    #for line in result.lines[:5]:  # Preview the first few lines
        # Dynamically fetch the speaker for each line
    #    speaker = (
    #        result.lines[result.lines.index(line)]["speaker"]
    #        if "speaker" in result.lines[result.lines.index(line)]
    #        else "Narrator/Unknown"
    #    )
    #    print(f"{speaker}: {line}")

    for line in result.lines[:5]:  # preview first few lines
        print(f" Line: {line}") #I'm afraid because it is a token based search, the speaker can't be retrieved

#i modified the code above because the speaker was printed even when the lines didn't have a speaker and were descriptions
query = Query("What's the college called?")
results = index.search(query)

for result in results:
    print(f"\n{result.title}")
    print("-" * 40)

    for line in result.lines[:5]:  # Preview the first few lines
        if isinstance(line, dict):  # Check if the line is a dictionary
            # Fetch the speaker if available, otherwise "Narrator/Unknown"
            speaker = line.get("speaker", "Narrator/Unknown")
            print(f"Line: {line['text']}, Speaker: {speaker}")
        else:
            # If the line is a string (description), set the speaker to "Narrator/Unknown"
            print(f"Line: {line}, Speaker: Narrator/Unknown")

# Token based search PLUS contextual snippets of matching lines from episodes

query_text = "college"
query = Query(query_text)
results = index.search(query)

if not results:
    print("No episodes found for the query.")
else:
    print(f"Search results for: '{query_text}'")
    print("=" * 50)

    query_tokens = set(Query(query_text).tokenize())

    for result in results:
        print(f"\n {result.title}")
        print("-" * 40)

        # Find and print a snippet that contains a token from the query
        found_snippet = False
        for line in result.lines:
            line_lower = line.lower()
            if any(token in line_lower for token in query_tokens):
                print(f" Snippet: {line}, Speaker: {speaker}") # want to add speaker-->added
                found_snippet = True
                break

        if not found_snippet:
            print("No matching snippet found in this episode.")

with open("/content/drive/MyDrive/IR_Project/inverted_index.pkl", "wb") as f:
    pickle.dump(index, f)

with open("/content/drive/MyDrive/IR_Project/inverted_index.pkl", mode="rb") as f:
    data = pickle.load(f)

pprint(data)
"""