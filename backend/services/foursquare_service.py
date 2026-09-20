"""
Foursquare API Service for places and attractions discovery.
"""

import logging
import httpx
from typing import List, Dict, Optional
from config import settings

logger = logging.getLogger(__name__)


class FoursquareService:
    """Service for Foursquare Places API."""

    # Category mappings for Foursquare v3
    CATEGORY_MAP = {
        "culture": "10000,12000",        # Arts & Entertainment, Landmarks
        "food": "13000",                  # Food & Dining
        "adventure": "16000,18000",       # Outdoors & Recreation
        "nature": "16000",               # Outdoors
        "nightlife": "10032",             # Nightlife
        "shopping": "17000",             # Shopping
        "history": "12000,12085",         # Landmarks, Museums
        "art": "10000,12080",            # Arts, Galleries
        "sports": "18000",              # Sports
        "wellness": "11000",             # Health & Wellness
        "family": "10000,16000",         # Entertainment, Outdoors
        "photography": "12000,16000",    # Landmarks, Outdoors
        "sightseeing": "12000,16000",    # Landmarks, Outdoors
    }

    def __init__(self):
        self.base_url = settings.FOURSQUARE_BASE_URL
        self.api_key = settings.FOURSQUARE_API_KEY

    async def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Make authenticated API request to Foursquare."""
        if not self.api_key or self.api_key.startswith("your_"):
            logger.warning("Foursquare API key not configured")
            return None

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}{endpoint}",
                    params=params,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Accept": "application/json",
                        "X-Places-Api-Version": "2025-06-17"
                    },
                    timeout=15.0
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Foursquare API error: {response.status_code} - {response.text}")
                    return None
        except Exception as e:
            logger.error(f"Foursquare request exception: {e}")
            return None

    async def search_places(self, location: str, categories: List[str] = None,
                             limit: int = 20) -> List[Dict]:
        """Search for places and attractions near a location."""
        params = {
            "near": location,
            "limit": limit,
            "sort": "RELEVANCE",
            "fields": "name,categories,location,rating,stats,description,hours,price,photos,tips,geocodes"
        }

        # Map user interests to Foursquare categories
        if categories:
            cat_ids = set()
            for cat in categories:
                if cat.lower() in self.CATEGORY_MAP:
                    cat_ids.update(self.CATEGORY_MAP[cat.lower()].split(","))
            if cat_ids:
                params["categories"] = ",".join(cat_ids)

        data = await self._make_request("/places/search", params)

        if not data:
            return self._get_sample_places(location, categories)

        places = []
        for result in data.get("results", []):
            try:
                location_data = result.get("location", {})
                geocodes = result.get("geocodes", {}).get("main", {})
                cats = result.get("categories", [])
                photos = result.get("photos", [])

                photo_url = None
                if photos:
                    p = photos[0]
                    photo_url = f"{p.get('prefix', '')}300x300{p.get('suffix', '')}"

                tips_list = []
                for tip in result.get("tips", [])[:3]:
                    tips_list.append(tip.get("text", ""))

                place = {
                    "name": result.get("name", "Unknown Place"),
                    "category": cats[0].get("name", "Attraction") if cats else "Attraction",
                    "address": location_data.get("formatted_address", location_data.get("address", "")),
                    "rating": result.get("rating", 0) / 2.0 if result.get("rating") else 0.0,
                    "description": result.get("description", ""),
                    "latitude": geocodes.get("latitude", 0),
                    "longitude": geocodes.get("longitude", 0),
                    "price_level": self._get_price_level(result.get("price")),
                    "image_url": photo_url,
                    "tips": tips_list,
                    "score": 0.0
                }
                places.append(place)
            except Exception as e:
                logger.warning(f"Error parsing Foursquare place: {e}")
                continue

        return places if places else self._get_sample_places(location, categories)

    async def get_place_details(self, fsq_id: str) -> Optional[Dict]:
        """Get detailed information about a specific place."""
        params = {
            "fields": "name,categories,location,rating,stats,description,hours,price,photos,tips,geocodes,website,tel"
        }
        return await self._make_request(f"/places/{fsq_id}", params)

    async def search_nearby(self, lat: float, lng: float, radius: int = 2000,
                             categories: List[str] = None, limit: int = 10) -> List[Dict]:
        """Search for places near specific coordinates."""
        params = {
            "ll": f"{lat},{lng}",
            "radius": radius,
            "limit": limit,
            "sort": "DISTANCE",
            "fields": "name,categories,location,rating,distance,geocodes"
        }

        if categories:
            cat_ids = set()
            for cat in categories:
                if cat.lower() in self.CATEGORY_MAP:
                    cat_ids.update(self.CATEGORY_MAP[cat.lower()].split(","))
            if cat_ids:
                params["categories"] = ",".join(cat_ids)

        data = await self._make_request("/places/search", params)
        if not data:
            return []

        return [
            {
                "name": r.get("name", ""),
                "category": r.get("categories", [{}])[0].get("name", "Place") if r.get("categories") else "Place",
                "distance": r.get("distance", 0),
                "latitude": r.get("geocodes", {}).get("main", {}).get("latitude", 0),
                "longitude": r.get("geocodes", {}).get("main", {}).get("longitude", 0),
            }
            for r in data.get("results", [])
        ]

    def _get_price_level(self, price_data) -> str:
        """Convert price data to display string."""
        if not price_data:
            return "$$"
        tier = price_data if isinstance(price_data, int) else 2
        return "$" * min(tier, 4)

    def _get_sample_places(self, location: str, categories: List[str] = None) -> List[Dict]:
        """Generate sample places for demo/fallback."""
        import random
        city = location.split(",")[0].strip()

        sample_places = [
            {
                "name": f"{city} Central Museum",
                "category": "Museum",
                "address": f"Museum Street, {city}",
                "rating": 4.5,
                "description": f"A world-renowned museum featuring art and artifacts from {city}'s rich history.",
                "latitude": round(random.uniform(30, 50), 4),
                "longitude": round(random.uniform(-120, 30), 4),
                "price_level": "$$",
                "tips": ["Visit early morning to avoid crowds", "Audio guide is worth it"],
                "score": round(random.uniform(0.7, 0.95), 2)
            },
            {
                "name": f"Old Town {city}",
                "category": "Historic Site",
                "address": f"Heritage Quarter, {city}",
                "rating": 4.7,
                "description": f"Explore the charming old town area with cobblestone streets and historic architecture.",
                "latitude": round(random.uniform(30, 50), 4),
                "longitude": round(random.uniform(-120, 30), 4),
                "price_level": "$",
                "tips": ["Best explored on foot", "Great photo opportunities"],
                "score": round(random.uniform(0.7, 0.95), 2)
            },
            {
                "name": f"{city} Food Market",
                "category": "Food Market",
                "address": f"Market Street, {city}",
                "rating": 4.3,
                "description": f"Vibrant local food market where you can taste authentic {city} cuisine.",
                "latitude": round(random.uniform(30, 50), 4),
                "longitude": round(random.uniform(-120, 30), 4),
                "price_level": "$",
                "tips": ["Try the local street food", "Bargaining is expected"],
                "score": round(random.uniform(0.7, 0.95), 2)
            },
            {
                "name": f"{city} Botanical Gardens",
                "category": "Park",
                "address": f"Garden Avenue, {city}",
                "rating": 4.4,
                "description": f"Beautiful botanical gardens with diverse plant collections and peaceful walking paths.",
                "latitude": round(random.uniform(30, 50), 4),
                "longitude": round(random.uniform(-120, 30), 4),
                "price_level": "$",
                "tips": ["Perfect for photography", "Visit during spring for blooming flowers"],
                "score": round(random.uniform(0.7, 0.95), 2)
            },
            {
                "name": f"{city} Observation Tower",
                "category": "Landmark",
                "address": f"Sky District, {city}",
                "rating": 4.6,
                "description": f"Get panoramic 360° views of {city} from this iconic observation deck.",
                "latitude": round(random.uniform(30, 50), 4),
                "longitude": round(random.uniform(-120, 30), 4),
                "price_level": "$$$",
                "tips": ["Go at sunset for best views", "Book tickets online to skip the line"],
                "score": round(random.uniform(0.7, 0.95), 2)
            },
            {
                "name": f"The {city} Art Gallery",
                "category": "Art Gallery",
                "address": f"Arts District, {city}",
                "rating": 4.2,
                "description": f"Contemporary art gallery showcasing local and international artists.",
                "latitude": round(random.uniform(30, 50), 4),
                "longitude": round(random.uniform(-120, 30), 4),
                "price_level": "$$",
                "tips": ["Free admission on first Sundays", "Check for special exhibitions"],
                "score": round(random.uniform(0.7, 0.95), 2)
            },
            {
                "name": f"{city} Night Market",
                "category": "Night Market",
                "address": f"Downtown {city}",
                "rating": 4.4,
                "description": f"Bustling night market with local crafts, street food, and live entertainment.",
                "latitude": round(random.uniform(30, 50), 4),
                "longitude": round(random.uniform(-120, 30), 4),
                "price_level": "$",
                "tips": ["Opens at 6 PM every evening", "Cash is preferred"],
                "score": round(random.uniform(0.7, 0.95), 2)
            },
            {
                "name": f"{city} Adventure Park",
                "category": "Adventure",
                "address": f"Outskirts of {city}",
                "rating": 4.3,
                "description": f"Thrilling outdoor adventure park with zip lines, rock climbing, and nature trails.",
                "latitude": round(random.uniform(30, 50), 4),
                "longitude": round(random.uniform(-120, 30), 4),
                "price_level": "$$$",
                "tips": ["Book in advance", "Wear comfortable shoes"],
                "score": round(random.uniform(0.7, 0.95), 2)
            },
        ]
        return sample_places


# Singleton instance
foursquare_service = FoursquareService()
