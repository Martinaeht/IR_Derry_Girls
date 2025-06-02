# episode.py

class Episode:
    def __init__(self, episode_data):
        self.season = episode_data["season"]
        self.episode = episode_data["episode"]
        self.id = f"S{self.season}E{self.episode}"  # Unique identifier
        self.title = f"Season {self.season}, Episode {self.episode}"
        self.lines = episode_data["lines"]

    def __str__(self):
        return f"{self.title}\n" + "\n".join(
            f"{line.get('speaker', '')}: {line['text']}" for line in self.lines if line["text"]
        )

    def __repr__(self):
        return f"{self.title}"
