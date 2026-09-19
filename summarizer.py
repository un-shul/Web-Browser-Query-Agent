import contextlib
import os
import sys
import re
import threading
import warnings
import logging

import config

# Suppress transformers warnings and logging
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

_summarizer = None
_summarizer_lock = threading.Lock()


def get_summarizer():
    """Build the summarization pipeline on first use.

    This used to run at import time, which pulled ~1.2GB of model weights into
    memory before the app could serve a single request.
    """
    global _summarizer
    if _summarizer is not None:
        return _summarizer
    with _summarizer_lock:
        if _summarizer is None:  # re-check under the lock
            from transformers import pipeline

            _summarizer = pipeline(
                "summarization", model=config.LOCAL_SUMMARIZER_MODEL
            )
    return _summarizer

# Silence ALL output (both stdout and stderr) including Hugging Face warnings
@contextlib.contextmanager
def suppress_warnings():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def clean_text(text):
    """Targeted cleaning to remove trending topics, news widgets, and junk"""
    # Remove trending topics and news widgets (the main problem)
    text = re.sub(r"\b(Hot Picks|Trending|Popular|Latest News|Breaking News|Also Read|Related Articles)\b.*?(?=\.|$)", "", text, flags=re.I)
    
    # Remove specific junk patterns from your example
    text = re.sub(r"\b(Bigg Boss|Weekly Horoscope|AAP Candidates|Luigi Mangione|RBI Governor|Harmeet Dhillon|Delhi Bomb|Mumbai Bus|Israel Strikes|Winter Weight Loss|Hair Fall|Microsoft 365|Narayana Murthy|Elon Musk)\b.*?(?=\.|$)", "", text, flags=re.I)
    
    # Remove obvious login/registration prompts
    text = re.sub(r"\b(You are already registered|Please login|Dont have an account|Sign up|Create account|Login|Register)\b.*?(?=\.|$)", "", text, flags=re.I)
    
    # Remove obvious ads and promotional content
    text = re.sub(r"\b(Advertisement|Sponsored|Click here|Buy now|Shop now|Subscribe|Newsletter)\b.*?(?=\.|$)", "", text, flags=re.I)
    
    # Remove cookie notices
    text = re.sub(r"\b(This website uses cookies|We use cookies|Accept cookies|Cookie policy)\b.*?(?=\.|$)", "", text, flags=re.I)
    
    # Remove sentences that are just lists of random topics (like your example)
    text = re.sub(r"\b\d+\s+\d+\s+\d+\s+Hot Picks.*?(?=\.|$)", "", text, flags=re.I)
    
    # Remove sentences with too many capitalized words (likely headlines/topics)
    sentences = text.split('.')
    filtered_sentences = []
    for sentence in sentences:
        # Count capitalized words
        words = sentence.split()
        if len(words) > 3:
            caps_count = sum(1 for word in words if word and word[0].isupper())
            # If more than 40% of words are capitalized, likely a list of topics
            if caps_count / len(words) > 0.4:
                continue
        filtered_sentences.append(sentence)
    
    text = '.'.join(filtered_sentences)
    
    # Basic cleanup
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[^\w\s,.!?']", "", text)  # Keep apostrophes
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()

def fix_punctuation(text):
    """Basic punctuation fixes"""
    # Remove spaces before punctuation
    text = re.sub(r'\s+([.!?,:;])', r'\1', text)
    
    # Fix capitalization after periods
    text = re.sub(r'(\.)(\s*)([a-z])', lambda m: m.group(1) + ' ' + m.group(3).upper(), text)
    
    # Ensure single space after punctuation
    text = re.sub(r'([.!?])\s*', r'\1 ', text)
    text = re.sub(r'([,:;])\s*', r'\1 ', text)
    
    # Fix multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Ensure proper capitalization at start
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    
    return text

def split_text(text, max_words=400):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])

def trim_to_sentence_boundary(text, max_chars=3500):
    """Trim to sentence boundary"""
    text = fix_punctuation(text)
    
    if len(text) <= max_chars:
        return text

    sentences = re.split(r'(?<=[.!?]) +', text)
    trimmed = ''
    for sentence in sentences:
        if len(trimmed) + len(sentence) <= max_chars:
            trimmed += sentence + ' '
        else:
            break
    
    return fix_punctuation(trimmed.strip())

def summarize_text(text, query=None, max_chunk_words=380):
    """Simple summarization with targeted junk removal"""
    text = clean_text(text)
    
    if len(text.split()) < 30:
        return "Content too short to summarize."
    
    chunks = list(split_text(text, max_chunk_words))

    summaries = []
    for chunk in chunks:
        try:
            with suppress_warnings():
                summary = get_summarizer()(
                    chunk, max_length=160, min_length=50, do_sample=False
                )
            summaries.append(summary[0]['summary_text'])
        except Exception as e:
            print(f"⚠️ Skipping chunk due to error: {e}")
            continue

    if not summaries:
        return "Unable to generate summary."
    
    final_summary = "\n\n".join(summaries)
    final_summary = fix_punctuation(final_summary)
    
    return trim_to_sentence_boundary(final_summary)