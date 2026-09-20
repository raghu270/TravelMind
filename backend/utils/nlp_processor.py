"""
NLP Processor for extracting travel preferences from natural language.
"""
import re
import logging
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)

INTEREST_KEYWORDS = {
    "culture": ["culture", "cultural", "tradition", "local", "heritage", "customs"],
    "food": ["food", "cuisine", "restaurant", "eat", "dining", "culinary", "street food"],
    "adventure": ["adventure", "thrill", "extreme", "hiking", "climbing", "zip"],
    "nature": ["nature", "park", "garden", "beach", "mountain", "lake", "scenic"],
    "nightlife": ["nightlife", "bar", "club", "party", "pub", "lounge"],
    "shopping": ["shopping", "mall", "market", "souvenir", "boutique", "store"],
    "history": ["history", "historical", "ancient", "old", "monument", "ruins"],
    "art": ["art", "gallery", "museum", "painting", "sculpture", "exhibition"],
    "sports": ["sports", "stadium", "game", "match", "tournament", "fitness"],
    "wellness": ["spa", "wellness", "yoga", "meditation", "retreat", "relax"],
    "family": ["family", "kids", "children", "family-friendly", "theme park"],
    "photography": ["photo", "photography", "instagram", "scenic", "viewpoint"],
}

BUDGET_KEYWORDS = {
    "budget": ["budget", "cheap", "affordable", "economical", "backpack", "hostel"],
    "moderate": ["moderate", "mid-range", "reasonable", "comfortable", "standard"],
    "luxury": ["luxury", "premium", "5-star", "exclusive", "upscale", "first class"],
}


def extract_interests(text: str) -> List[str]:
    text_lower = text.lower()
    found = []
    for interest, keywords in INTEREST_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            found.append(interest)
    return found if found else ["culture", "food", "sightseeing"]


def extract_budget_level(text: str) -> str:
    text_lower = text.lower()
    for level, keywords in BUDGET_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return level
    return "moderate"


def extract_budget_amount(text: str) -> float | None:
    patterns = [
        r'\$\s*([\d,]+)', r'(\d{3,})\s*(?:usd|dollars?)',
        r'budget\s*(?:of|is|:)?\s*\$?\s*([\d,]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                continue
    return None


def extract_dates(text: str) -> Dict[str, str]:
    date_patterns = [
        r'(\d{4}-\d{2}-\d{2})',
        r'(\d{1,2}/\d{1,2}/\d{4})',
        r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4})',
    ]
    dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        dates.extend(matches)
    result = {}
    if len(dates) >= 2:
        result["departure_date"] = dates[0]
        result["return_date"] = dates[1]
    elif len(dates) == 1:
        result["departure_date"] = dates[0]
    return result


def extract_travelers(text: str) -> int:
    patterns = [
        r'(\d+)\s*(?:people|persons|travelers|adults|guests|travellers)',
        r'(?:for|group of)\s*(\d+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            try:
                return min(int(match.group(1)), 10)
            except ValueError:
                pass
    return 1


def process_natural_language(text: str) -> Dict[str, Any]:
    return {
        "interests": extract_interests(text),
        "budget_level": extract_budget_level(text),
        "budget_amount": extract_budget_amount(text),
        "dates": extract_dates(text),
        "travelers": extract_travelers(text),
        "raw_input": text,
    }
