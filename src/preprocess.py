"""
Preprocessing Pipeline for Sentiment Analysis
Natural Language Processing (NLP) - Text Cleaning & Normalization
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLTK resources are loaded locally from user nltk_data directory

# Preserve negation words that are critical for sentiment polarity
NEGATION_WORDS = {
    'not', 'no', 'nor', 'never', 'neither', 'barely', 
    'hardly', 'scarcely', 'rarely', 'without', 'against'
}

# Stopwords set excluding negations
BASE_STOPWORDS = set(stopwords.words('english'))
SENTIMENT_STOPWORDS = BASE_STOPWORDS - NEGATION_WORDS

# Common English contractions expansion map
CONTRACTION_MAP = {
    r"won\'t": "will not",
    r"can\'t": "can not",
    r"n\'t": " not",
    r"\'re": " are",
    r"\'s": " is",
    r"\'d": " would",
    r"\'ll": " will",
    r"\'t": " not",
    r"\'ve": " have",
    r"\'m": " am",
}

lemmatizer = WordNetLemmatizer()


def remove_html_tags(text: str) -> str:
    """Strip HTML tags like <br />, <p>, etc."""
    return re.sub(r'<[^>]+>', ' ', text)


def expand_contractions(text: str) -> str:
    """Expand contractions (e.g. don't -> do not) to preserve negations."""
    cleaned = text
    for contraction, expansion in CONTRACTION_MAP.items():
        cleaned = re.sub(contraction, expansion, cleaned, flags=re.IGNORECASE)
    return cleaned


def clean_text_basic(text: str) -> str:
    """
    Standard lowercasing, HTML stripping, contraction expansion,
    and non-alphabetical character replacement.
    """
    if not isinstance(text, str):
        return ""
    text = remove_html_tags(text)
    text = expand_contractions(text)
    text = text.lower()
    # Replace non-letters with spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Collapse multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Fast memory cache for lemmatization
_LEMMA_CACHE = {}

def lemmatize_word(word: str) -> str:
    """
    Lemmatize word with verb fallback to map running -> run, movies -> movie.
    Uses in-memory cache for O(1) subsequent lookups.
    """
    if word in _LEMMA_CACHE:
        return _LEMMA_CACHE[word]
    v_lemma = lemmatizer.lemmatize(word, pos='v')
    result = lemmatizer.lemmatize(v_lemma, pos='n')
    _LEMMA_CACHE[word] = result
    return result


def preprocess_text(text: str) -> str:
    """
    Complete NLP preprocessing pipeline:
    Raw Text -> HTML/Contraction Clean -> Tokenize -> Filter Stopwords (preserve negations) -> Lemmatize
    """
    cleaned = clean_text_basic(text)
    tokens = [w for w in cleaned.split() if len(w) >= 2]
    lemmas = [lemmatize_word(w) for w in tokens if w not in SENTIMENT_STOPWORDS]
    return ' '.join(lemmas)


def explain_preprocessing_steps(raw_text: str) -> dict:
    """
    Educational inspection: Returns intermediate representation at each step of the pipeline.
    Perfect for students learning NLP transformations!
    """
    step1_raw = raw_text
    step2_no_html = remove_html_tags(raw_text)
    step3_contractions = expand_contractions(step2_no_html)
    step4_lowered_cleaned = re.sub(r'[^a-z\s]', ' ', step3_contractions.lower())
    step4_lowered_cleaned = re.sub(r'\s+', ' ', step4_lowered_cleaned).strip()
    
    tokens = [w for w in step4_lowered_cleaned.split() if len(w) >= 2]
    
    retained_tokens = []
    removed_stopwords = []
    for token in tokens:
        if token in SENTIMENT_STOPWORDS:
            removed_stopwords.append(token)
        else:
            retained_tokens.append(token)
            
    lemmatized_tokens = [lemmatize_word(w) for w in retained_tokens]
    final_text = ' '.join(lemmatized_tokens)
    
    return {
        "raw_text": step1_raw,
        "no_html": step2_no_html,
        "expanded_contractions": step3_contractions,
        "cleaned_text": step4_lowered_cleaned,
        "all_tokens": tokens,
        "removed_stopwords": removed_stopwords,
        "retained_tokens": retained_tokens,
        "lemmatized_tokens": lemmatized_tokens,
        "final_text": final_text,
        "tokens_count_initial": len(tokens),
        "tokens_count_retained": len(retained_tokens),
        "reduction_percentage": round((1 - len(retained_tokens)/max(len(tokens), 1)) * 100, 1)
    }


if __name__ == "__main__":
    sample = "The movie wasn't that great at all! <br /><br /> Acting was running in circles, but the visuals were awesome."
    print("Testing Preprocessing Pipeline...")
    res = explain_preprocessing_steps(sample)
    print("Original:", res["raw_text"])
    print("Cleaned & Expanded:", res["cleaned_text"])
    print("Filtered Stopwords:", set(res["removed_stopwords"]))
    print("Lemmatized Tokens:", res["lemmatized_tokens"])
    print("Final Processed:", res["final_text"])
