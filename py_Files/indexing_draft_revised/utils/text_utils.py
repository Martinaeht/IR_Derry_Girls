import re

class Text:
    def __init__(self, lines):
        self.lines = lines

    def tokenize(self):
        tokens = []
        for line in self.lines:
            clean_line = re.sub(r"\([^)]*\)", "", line)  
            clean_line = re.sub(r"[^\w\s]", "", clean_line.lower())
            tokens.extend(clean_line.split())
        return tokens
