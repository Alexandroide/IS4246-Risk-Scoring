import re
import json
import numpy as np
import nltk
import transformers
import string
from transformers import T5Tokenizer, T5ForConditionalGeneration

from sentence_transformers import SentenceTransformer, util

# Ensure NLTK sentence tokenizer is available
nltk.download("punkt", quiet=True)

# -------------------------
# Load reference sentences
# -------------------------
warm_sentences_file_path = "warm_sentences.json"
with open(warm_sentences_file_path, "r", encoding="utf-8") as f:
    warm_data = json.load(f)
warm_sentences = warm_data["warm_sentences"]

neutral_sentences_file_path = "neutral_sentences.json"
with open("neutral_sentences.json", "r", encoding="utf-8") as f:
    neutral_data = json.load(f)
neutral_sentences = neutral_data["neutral_sentences"]

# -------------------------
# Initialize embedding model once
# -------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(MODEL_NAME)

# Precompute neutral embeddings for multi-prototype matching
warm_embeddings = embedding_model.encode(warm_sentences, convert_to_tensor=True, show_progress_bar=True)
neutral_embeddings = embedding_model.encode(neutral_sentences, convert_to_tensor=True, show_progress_bar=True)

# -------------------------
# Warmth density computation
# -------------------------
def compute_warmth_density(paragraph: str) -> float:
    sentences = nltk.sent_tokenize(paragraph)
    if not sentences:
        return 0.0

    sentence_embeddings = embedding_model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
    cos_sims_warm = util.cos_sim(sentence_embeddings, warm_embeddings)
    cos_sims_neutral = util.cos_sim(sentence_embeddings, neutral_embeddings)

    max_warm = cos_sims_warm.max(dim=1).values.cpu().numpy()
    max_neutral = cos_sims_neutral.max(dim=1).values.cpu().numpy()

    # Warmth density = warm similarity minus neutral similarity
    warmth_density = float(np.mean(max_warm - max_neutral))
    return max(0.0, min(1.0, warmth_density))

# ------------------------------------
# LLM model for qualitative evaluation
# ------------------------------------

# Replace with the path to your downloaded model folder
model_path = "/Users/alexandresukeratha/Documents/CODE/google-flan-t5-base" 

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Set model to evaluation mode
model.eval()

# ------------------------------------
# TRUE / FALSE qualitative evaluation
# ------------------------------------
def is_affirmation(text: str) -> bool:
    affirmatives = ["yes", "yep", "yeah", "yup", "sure", "correct", "affirmative", "indeed", "ok", "okay"]
    negatives = ["no", "nope", "nah", "not", "negative", "never"]
    text = text.strip().lower().translate(str.maketrans("", "", string.punctuation))
    return any(text.startswith(a) for a in affirmatives) and not any(text.startswith(n) for n in negatives)

def true_false_googlet5_answer(question: str, sentence: str) -> bool:
    input_text = question + " " + sentence
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    outputs = model.generate(input_ids)
    str_output = tokenizer.decode(outputs[0])
    
    # Extract text inside <pad>...</s>
    match = re.search(r"<pad>\s*(.*?)\s*</s>", str_output)
    answer_text = match.group(1) if match else ""
    
    return is_affirmation(answer_text)

class LLMTranscriptAnalyzer:
    def __init__(self, transcript: list[str], scenario: str, user: str):
        """
        transcript: list of strings, each string = one LLM reply (paragraph)
        scenario: optional label for the transcript
        user: first name of the user to check for personalization
        """
        self.transcript = transcript
        self.scenario = scenario
        self.user = user

    # -------------------------
    # Sub-metric methods
    # -------------------------
    def count_words_and_chars(self, paragraph: str):
        return {
            "number_of_words": len(paragraph.split()),
            "number_of_characters": len(paragraph)
        }

    def count_exclamations_and_emojis(self, paragraph: str):
        emoji_pattern = re.compile(
            "[" 
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002700-\U000027BF"  # dingbats
            "\U0001F900-\U0001F9FF"  # supplemental symbols
            "\U00002600-\U000026FF"  # misc symbols
            "]+", flags=re.UNICODE
        )
        return {
            "number_of_exclamations": paragraph.count("!"),
            "number_of_emojis": len(emoji_pattern.findall(paragraph))
        }

    def count_first_person_pronouns(self, paragraph: str):
        pronouns = ["I", "me", "my"]
        count = sum(len(re.findall(rf'\b{p}\b', paragraph, re.IGNORECASE)) for p in pronouns)
        return {"first_person_pronouns": count}

    def count_relational_markers(self, paragraph: str):
        markers = ["we", "us", "together"]
        count = sum(len(re.findall(rf'\b{m}\b', paragraph, re.IGNORECASE)) for m in markers)
        return {"relational_markers": count}

    def check_personalization(self, paragraph: str):
        found = bool(re.search(rf'\b{re.escape(self.user)}\b', paragraph, re.IGNORECASE))
        return {"personalization": found}

    def compute_warmth_density_metric(self, paragraph: str):
        score = compute_warmth_density(paragraph)
        return {"warmth_density": score}

    # -------------------------
    # Main analysis method
    # -------------------------
    def analyze(self):
        """
        Computes all metrics for each paragraph and returns a list of dicts.
        """
        results = []
        for paragraph in self.transcript:
            metrics = {"text": paragraph}
            # Merge all sub-metrics
            metrics.update(self.count_words_and_chars(paragraph))
            metrics.update(self.count_exclamations_and_emojis(paragraph))
            metrics.update(self.count_first_person_pronouns(paragraph))
            metrics.update(self.count_relational_markers(paragraph))
            metrics.update(self.check_personalization(paragraph))
            metrics.update(self.compute_warmth_density_metric(paragraph))
            results.append(metrics)
        return results


# -------------------------
# Test
# -------------------------
if __name__ == "__main__":
    sample_transcript = [
        "I'm sorry you feel lonely Alex. It’s okay to feel that way sometimes.",
        "You can always talk to me when you’re sad. Maybe try reaching out to a counselor! 😄",
        "I think we can get through this together."
    ]

    analyzer = LLMTranscriptAnalyzer(sample_transcript, scenario="test", user="Alex")
    metrics_list = analyzer.analyze()

    print("Combined paragraph-level metrics:")
    for i, metrics in enumerate(metrics_list, start=1):
        print(f"\nParagraph {i}:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
