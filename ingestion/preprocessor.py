"""
Text preprocessing and normalization for Vietnamese admissions documents.
"""
import re
import unicodedata
from typing import Optional

from src.core.logger import get_logger

logger = get_logger(__name__)


def normalize_unicode(text: str) -> str:
    """Normalize Vietnamese Unicode text to NFC form."""
    return unicodedata.normalize("NFC", text)


def clean_whitespace(text: str) -> str:
    """Collapse multiple whitespace/newlines into single spaces."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_html_tags(text: str) -> str:
    """Strip HTML/XML tags."""
    return re.sub(r"<[^>]+>", "", text)


def remove_special_chars(text: str) -> str:
    """Remove control characters but keep Vietnamese diacritics."""
    # Keep: letters (including Vietnamese), digits, punctuation, whitespace
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)


def normalize_numbers(text: str) -> str:
    """Normalize number formats (e.g., '1.000.000' → '1000000')."""
    # Vietnamese uses dots as thousands separator
    def _replace_vn_number(match):
        return match.group(0).replace(".", "")

    return re.sub(r"\d{1,3}(?:\.\d{3})+", _replace_vn_number, text)


def preprocess_text(
    text: str,
    normalize_nums: bool = False,
    strip_html: bool = True,
) -> str:
    """
    Full preprocessing pipeline for a text string.

    1. Unicode normalization (NFC)
    2. HTML tag removal (optional)
    3. Special character removal
    4. Whitespace normalization
    5. Number normalization (optional)
    """
    text = normalize_unicode(text)
    if strip_html:
        text = remove_html_tags(text)
    text = remove_special_chars(text)
    text = clean_whitespace(text)
    if normalize_nums:
        text = normalize_numbers(text)
    return text


def split_sentences_vietnamese(text: str) -> list[str]:
    """
    Split Vietnamese text into sentences.
    Handles common Vietnamese punctuation patterns.
    """
    # Split on sentence-ending punctuation followed by space or newline
    sentences = re.split(r'(?<=[.!?;])\s+(?=[A-ZÀ-Ỹa-zà-ỹ\d])', text)
    # Also split on double newlines (paragraph boundaries)
    result = []
    for sent in sentences:
        parts = sent.split("\n\n")
        result.extend(p.strip() for p in parts if p.strip())
    return result
