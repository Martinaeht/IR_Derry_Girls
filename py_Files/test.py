import re



def parse_script(raw_lines):
    scene_pattern = re.compile(r"^\[(.*)\]$")
    speaker_pattern = re.compile(r"^([A-Za-z0-9 ']+):")
    action_pattern = re.compile(r"\((.*?)\)")

    parsed_lines = []
    scene = None
    season = episode = None

    for line_number, raw_line in enumerate(raw_lines):
        line = raw_line.strip()
        if not line:
            continue

        if line.upper().startswith("SEASON"):  
            season_match = re.search(r"\d+", line) 
            if season_match: 
                season = int(season_match.group())
                print(f"Found Season {season} at line {line_number}")  
            else:
                print(f"Warning: SEASON line without number at line {line_number}: '{line}'") 
            continue

        if line.upper().startswith("EPISODE"):  
            episode_match = re.search(r"\d+", line)  
            if episode_match:  
                episode = int(episode_match.group())
                print(f"Found Episode {episode} at line {line_number}") 
            else:
                print(f"Warning: EPISODE line without number at line {line_number}: '{line}'") 
            continue

        scene_match = scene_pattern.match(line)
        if scene_match:
            scene = scene_match.group(1).strip()
            parsed_lines.append({
                "id": len(parsed_lines),
                "line_number": line_number,
                "season": season,
                "episode": episode,
                "scene": scene,
                "speaker": None,
                "actions": [],
                "clean_text": None,
                "raw_line": line
            })
            continue

        speaker_match = speaker_pattern.match(line)
        if speaker_match:
            speaker = speaker_match.group(1).strip()
            dialogue_with_actions = line[len(speaker) + 1:].strip()
        else:
            speaker = None
            dialogue_with_actions = line

        actions = action_pattern.findall(dialogue_with_actions)
        clean_text = action_pattern.sub("", dialogue_with_actions).strip()

        parsed_lines.append({
            "id": len(parsed_lines),
            "line_number": line_number,
            "season": season,
            "episode": episode,
            "scene": scene,
            "speaker": speaker,
            "actions": actions,
            "clean_text": clean_text if clean_text else None,
            "raw_line": line
        })
    return parsed_lines
