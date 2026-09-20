"""
Flight Recommendation Agent.
Uses SASRec Transformer + RL optimization for personalized flight recommendations.
"""
import logging
from typing import List, Dict, Any
from services.amadeus_service import amadeus_service
from utils.helpers import format_duration

logger = logging.getLogger(__name__)

# Lazy-load ML model
_flight_model = None
def _get_model():
    global _flight_model
    if _flight_model is None:
        try:
            from ml_models.flight_recommender import flight_recommender
            _flight_model = flight_recommender
            logger.info("✈️ SASRec+RL flight model loaded")
        except Exception as e:
            logger.warning(f"Flight ML model load warning: {e}")
    return _flight_model


class FlightAgent:
    """Agent with SASRec Transformer + RL price optimization."""

    def __init__(self):
        self.name = "FlightAgent"

    async def get_recommendations(self, source: str, destination: str,
                                   departure_date: str, return_date: str = None,
                                   travelers: int = 1, budget_level: str = "moderate",
                                   preferences: Dict = None) -> List[Dict]:
        logger.info(f"✈️ {self.name}: Searching flights {source} → {destination}")
        flights = await amadeus_service.search_flights(
            source, destination, departure_date, return_date, travelers
        )

        # ML Model scoring (SASRec + RL)
        model = _get_model()
        if model:
            flights = model.score_flights(flights, budget_level, preferences or {})
            logger.info(f"✈️ {self.name}: SASRec+RL scored {len(flights)} flights")
        else:
            flights = self._fallback_score(flights, budget_level)

        flights.sort(key=lambda x: x.get("score", 0), reverse=True)
        for f in flights:
            if isinstance(f.get("duration", ""), str) and f["duration"].startswith("PT"):
                f["duration"] = format_duration(f["duration"])
        logger.info(f"✈️ {self.name}: Returning {len(flights)} flights")
        return flights

    def _fallback_score(self, flights: List[Dict], budget_level: str) -> List[Dict]:
        if not flights:
            return []
        prices = [f.get("price", 0) for f in flights if f.get("price", 0) > 0]
        if not prices:
            return flights
        min_p, max_p = min(prices), max(prices)
        rng = max_p - min_p if max_p > min_p else 1
        for f in flights:
            s = 0.0
            if budget_level == "budget":
                s += 40 * (1 - (f.get("price", 0) - min_p) / rng)
            else:
                s += 30 * (1 - (f.get("price", 0) - min_p) / rng)
            s += max(0, 30 - f.get("stops", 0) * 15)
            dep = f.get("departure_time", "")
            if "T" in dep:
                h = int(dep.split("T")[1][:2])
                s += 20 if 8 <= h <= 11 else 10
            s += 8
            f["score"] = round(min(s, 100) / 100, 2)
        return flights

flight_agent = FlightAgent()
