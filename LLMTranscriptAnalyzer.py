import re

class LLMTranscriptAnalyzer:
    def __init__(self, transcript: list[str]):
        """
        transcript: list of strings, each string = one LLM reply (paragraph)
        """
        self.transcript = transcript

        # Compute metrics using the method
        self.response_length = self.compute_response_length(self.transcript)
        self.use_of_emojis_or_exclamations = self.compute_use_of_emojis_or_exclamations(self.transcript)

    # Methods
    def compute_response_length(self, paragraphs: list[str]):
        """
        Takes a list of paragraph strings and returns a dictionary with 
        number of words and characters for each paragraph.

        Args:
            paragraphs (list of str): List of paragraph strings.

        Returns:
            dict: Keys are 'paragraph 1', 'paragraph 2', ... 
                Values are dicts with 'words' and 'characters' counts.
        """
        stats = {}
        for i, para in enumerate(paragraphs, start=1):
            word_count = len(para.split())
            char_count = len(para)
            stats[f"paragraph {i}"] = {
                "number_of_words": word_count,
                "number_of_characters": char_count
            }
        return stats
    
    def compute_use_of_emojis_or_exclamations(self, paragraphs: list[str]):
        """
        Takes a list of paragraph strings and returns a dictionary with
        number of exclamation marks and emojis for each paragraph.

        Args:
            paragraphs (list of str): List of paragraph strings.

        Returns:
            dict: Keys are 'paragraph 1', 'paragraph 2', ...
                Values are dicts with 'exclamations' and 'emojis' counts.
        """
        # Regex pattern to match emojis
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # Emoticons
            "\U0001F300-\U0001F5FF"  # Symbols & pictographs
            "\U0001F680-\U0001F6FF"  # Transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # Flags
            "\U00002700-\U000027BF"  # Dingbats
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U00002600-\U000026FF"  # Misc symbols
            "]+", flags=re.UNICODE
        )
        
        stats = {}
        for i, para in enumerate(paragraphs, start=1):
            exclamations = para.count("!")
            emojis = len(emoji_pattern.findall(para))
            stats[f"paragraph {i}"] = {
                "number_of_exclamations": exclamations,
                "number_of_emojis": emojis
            }
        return stats

# Test
if __name__ == "__main__":
    sample_transcript = [
        "I'm sorry you feel lonely. It’s okay to feel that way sometimes.",
        "You can always talk to me when you’re sad. Maybe try reaching out to a counselor! 😄"
    ]

    analyzer = LLMTranscriptAnalyzer(sample_transcript)

    # Print the transcript
    print("Transcript:")
    for i, para in enumerate(analyzer.transcript, start=1):
        print(f"Paragraph {i}: {para}")

    # Print the computed response length metrics
    print("\nResponse length metrics:")
    for para, stats in analyzer.response_length.items():
        print(f"{para}: {stats}")

    # Print the emoji and exclamation metrics
    print("\nEmoji and exclamation metrics:")
    for para, stats in analyzer.use_of_emojis_or_exclamations.items():
        print(f"{para}: {stats}")
