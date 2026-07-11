"""Shared text preprocessing used by both training and inference."""

import re


def clean_text(text: str) -> str:
    """Normalize English resume text for TF-IDF feature extraction."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s.,;:!?\'\"()-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
