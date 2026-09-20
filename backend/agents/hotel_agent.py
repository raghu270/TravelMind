"""
Hotel Recommendation Agent.
Uses Neural Collaborative Filtering (NCF) + Knowledge Graph Neural Networks.
"""
import logging
from typing import List, Dict, Any
from services.amadeus_service import amadeus_service

logger = logging.getLogger(__name__)

_hotel_model = None
def _get_model():
    global _hotel_model
    if _hotel_model is None:
        try:
            from ml_models.hotel_recommender import hotel_recommender
            _hotel_model = hotel_recommender
            logger.info("🏨 NCF+KG hotel model loaded")
        except Exception as e:
            logger.warning(f"Hotel ML model load warning: {e}")
    return _hotel_model


class HotelAgent:
    """Agent with NCF + Knowledge Graph for personalized hotel ranking."""

    BUDGET_PRICE_RANGES = {
        "budget": (0, 100), "moderate": (80, 250), "luxury": (200, 1000)
    }

    def __init__(self):
        self.name = "HotelAgent"

    async def get_recommendations(self, city: str, check_in: str, check_out: str,
                                   travelers: int = 1, budget_level: str = "moderate",
                                   interests: list = None) -> List[Dict]:
        logger.info(f"🏨 {self.name}: Searching hotels in {city}")
        hotels = await amadeus_service.search_hotels(city, check_in, check_out, travelers)

        # ML Model scoring (NCF + Knowledge Graph)
        model = _get_model()
        if model:
            hotels = model.score_hotels(hotels, budget_level, interests or [])
            logger.info(f"🏨 {self.name}: NCF+KG scored {len(hotels)} hotels")
        else:
            hotels = self._fallback_score(hotels, budget_level, interests or [])

        hotels.sort(key=lambda x: x.get("score", 0), reverse=True)
        logger.info(f"🏨 {self.name}: Returning {len(hotels)} hotels")
        return hotels

    def _fallback_score(self, hotels, budget_level, interests):
        if not hotels:
            return []
        for h in hotels:
            s = 0.0
            s += min(h.get("rating", 0) / 5.0 * 30, 30)
            s += min(h.get("stars", 0) / 5.0 * 20, 20)
            lo, hi = self.BUDGET_PRICE_RANGES.get(budget_level, (80, 250))
            ppn = h.get("price_per_night", 0)
            if lo <= ppn <= hi:
                s += 25
            elif ppn < lo:
                s += 15
            else:
                s += max(0, 25 - (ppn - hi) / 50 * 5)
            s += max(0, 15 - h.get("distance_to_center", 5) * 1.5)
            s += 5
            h["score"] = round(min(s, 100) / 100, 2)
        return hotels

hotel_agent = HotelAgent()
