"""
Place Discovery Agent.
Uses Transformer Sequential Recommender + OpenAI LLM for place recommendations.
"""
import logging
from typing import List, Dict
from services.foursquare_service import foursquare_service
from services.openai_service import openai_service

logger = logging.getLogger(__name__)

_place_model = None
def _get_model():
    global _place_model
    if _place_model is None:
        try:
            from ml_models.place_recommender import place_recommender
            _place_model = place_recommender
            logger.info("📍 Sequential Place Recommender loaded")
        except Exception as e:
            logger.warning(f"Place ML model load warning: {e}")
    return _place_model


class PlaceAgent:
    """Agent with Transformer Sequential Rec + LLM contextual reasoning."""

    def __init__(self):
        self.name = "PlaceAgent"

    async def get_recommendations(self, destination: str,
                                   interests: list = None) -> List[Dict]:
        logger.info(f"📍 {self.name}: Discovering places in {destination}")
        places = await foursquare_service.search_places(destination, interests)

        # ML Model scoring (Transformer Sequential Recommender)
        model = _get_model()
        if model and places:
            places = model.score_places(places, interests or [])

        # LLM-enhanced ranking
        if places:
            try:
                enhanced = await openai_service.generate_place_recommendations(
                    destination, interests or [], places
                )
                if enhanced:
                    places = self._merge_enhancements(places, enhanced)
            except Exception as e:
                logger.warning(f"LLM enhancement error: {e}")

        # Final scoring
        for p in places:
            if not p.get("score"):
                p["score"] = round(min(p.get("rating", 3) / 5.0, 1.0), 2)

        places.sort(key=lambda x: x.get("score", 0), reverse=True)
        logger.info(f"📍 {self.name}: Found {len(places)} places")
        return places

    def _merge_enhancements(self, places, enhanced):
        enh_map = {e.get("name", "").lower(): e for e in enhanced}
        for p in places:
            enh = enh_map.get(p["name"].lower(), {})
            if enh:
                llm_score = enh.get("score", 0)
                if isinstance(llm_score, (int, float)) and llm_score > 1:
                    llm_score = llm_score / 100.0
                # Blend ML score with LLM score
                ml_score = p.get("score", 0.5)
                p["score"] = round(0.6 * ml_score + 0.4 * llm_score, 3)
                if enh.get("recommendation"):
                    p["description"] = enh["recommendation"]
                if enh.get("insider_tip"):
                    p.setdefault("tips", []).insert(0, enh["insider_tip"])
        return places

place_agent = PlaceAgent()
