"""
Ticketmaster API Service for event discovery.
"""
import logging
import httpx
from typing import List, Dict, Optional
from config import settings

logger = logging.getLogger(__name__)


class TicketmasterService:
    INTEREST_GENRE_MAP = {
        "culture": "KnvZfZ7vAe1,KnvZfZ7vAJ1", "food": "KnvZfZ7vAAJ",
        "adventure": "KnvZfZ7vAdv", "sports": "KnvZfZ7vAde",
        "nightlife": "KnvZfZ7vAv1,KnvZfZ7vAvF", "art": "KnvZfZ7vAe1",
        "family": "KnvZfZ7vA1n", "history": "KnvZfZ7vAe1",
    }

    def __init__(self):
        self.base_url = settings.TICKETMASTER_BASE_URL
        self.api_key = settings.TICKETMASTER_API_KEY

    async def search_events(self, city: str, start_date: str = None,
                             end_date: str = None, interests: List[str] = None,
                             size: int = 20) -> List[Dict]:
        if not self.api_key or self.api_key.startswith("your_"):
            return self._get_sample_events(city, start_date, end_date)
        params = {"apikey": self.api_key, "city": city, "size": size, "sort": "date,asc"}
        if start_date:
            params["startDateTime"] = f"{start_date}T00:00:00Z"
        if end_date:
            params["endDateTime"] = f"{end_date}T23:59:59Z"
        if interests:
            genre_ids = set()
            for i in interests:
                if i.lower() in self.INTEREST_GENRE_MAP:
                    genre_ids.update(self.INTEREST_GENRE_MAP[i.lower()].split(","))
            if genre_ids:
                params["classificationId"] = ",".join(list(genre_ids)[:3])
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.base_url}/events.json", params=params, timeout=15.0)
                if resp.status_code != 200:
                    return self._get_sample_events(city, start_date, end_date)
                data = resp.json()
        except Exception as e:
            logger.error(f"Ticketmaster error: {e}")
            return self._get_sample_events(city, start_date, end_date)

        events = []
        for ev in data.get("_embedded", {}).get("events", []):
            try:
                venue = ev.get("_embedded", {}).get("venues", [{}])[0]
                dates = ev.get("dates", {}).get("start", {})
                prices = ev.get("priceRanges", [{}])
                price_str = f"${prices[0].get('min',0)}-${prices[0].get('max',0)}" if prices else "Check website"
                imgs = ev.get("images", [])
                img = imgs[0].get("url") if imgs else None
                classifications = ev.get("classifications", [{}])
                etype = classifications[0].get("segment", {}).get("name", "Event") if classifications else "Event"
                events.append({
                    "name": ev.get("name", ""), "event_type": etype,
                    "venue": venue.get("name", ""), "date": dates.get("localDate", ""),
                    "time": dates.get("localTime", ""), "price_range": price_str,
                    "description": ev.get("info", ev.get("pleaseNote", "")),
                    "url": ev.get("url"), "image_url": img, "relevance_score": 0.0
                })
            except Exception as e:
                logger.warning(f"Error parsing event: {e}")
        return events or self._get_sample_events(city, start_date, end_date)

    def _get_sample_events(self, city, start_date, end_date):
        import random
        from datetime import datetime, timedelta
        base = datetime.strptime(start_date, "%Y-%m-%d") if start_date else datetime.now()
        samples = [
            (f"{city} Music Festival", "Music", "City Arena"),
            (f"{city} Food & Wine Gala", "Arts & Theatre", "Convention Center"),
            (f"International Jazz Night in {city}", "Music", "Jazz Club Downtown"),
            (f"{city} Cultural Dance Show", "Arts & Theatre", "Heritage Theater"),
            (f"Comedy Night at {city}", "Arts & Theatre", "Laugh Factory"),
            (f"{city} Marathon", "Sports", "City Park"),
            (f"Tech Expo {city}", "Miscellaneous", "Exhibition Hall"),
            (f"{city} Art Walk", "Arts & Theatre", "Arts District"),
        ]
        events = []
        for i, (name, etype, venue) in enumerate(samples):
            dt = base + timedelta(days=random.randint(0, 5))
            events.append({
                "name": name, "event_type": etype, "venue": venue,
                "date": dt.strftime("%Y-%m-%d"),
                "time": f"{random.choice(['18','19','20'])}:00:00",
                "price_range": f"${random.randint(15,50)}-${random.randint(60,200)}",
                "description": f"Join us for an amazing {etype.lower()} experience in {city}.",
                "url": None, "image_url": None,
                "relevance_score": round(random.uniform(0.5, 0.95), 2)
            })
        return events

ticketmaster_service = TicketmasterService()
