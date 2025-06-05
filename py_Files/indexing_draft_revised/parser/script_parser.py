import re

def clean_text(text):
    return re.sub(r"\([^)]*\)", "", text).strip() # Removes anything in parentheses

def parse_episodes(file_lines):
    episodes = []
    current_lines = []
    season = episode = None

    for line in file_lines:
        line = line.strip()
        if not line:
            continue

        if line.upper().startswith("SEASON"):
            match = re.search(r"\d+", line)
            if match:
                season = int(match.group())

        elif line.upper().startswith("EPISODE"):
            if current_lines:
                episodes.append({
                    "season": season,
                    "episode": episode,
                    "lines": current_lines,
                })
                current_lines = []

            match = re.search(r"\d+", line)
            if match:
                episode = int(match.group())

        elif ":" in line and not line.startswith(("(", "[")):
            speaker, text = map(str.strip, line.split(":", 1))
            cleaned_text = clean_text(text)
            current_lines.append({"speaker": speaker, "text": cleaned_text})

        else:
            cleaned_text = clean_text(line)
            current_lines.append({"speaker": None, "text": cleaned_text})

    if current_lines:
        episodes.append({
            "season": season,
            "episode": episode,
            "lines": current_lines
        })

    return episodes
