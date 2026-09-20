"""
Event Awareness Agent.
Uses BERT/RoBERTa NLP Classifier + Knowledge Graph for event recommendations.
"""
import logging
from typing import List, Dict
from services.ticketmaster_service import ticketmaster_service
from services.openai_service import openai_service

logger = logging.getLogger(__name__)

_event_model = None
def _get_model():
    global _event_model
    if _event_model is None:
        try:
            from ml_models.event_classifier import event_classifier
            _event_model = event_classifier
            logger.info("🎉 Event BERT/NLP + KG Model loaded")
        except Exception as e:
            logger.warning(f"Event ML model load warning: {e}")
    return _event_model


class EventAgent:
    """Agent with BERT-style NLP classifier + Knowledge Graph event modeling."""

    def __init__(self):
        self.name = "EventAgent"

    async def get_recommendations(self, city: str, start_date: str,
                                   end_date: str, interests: list = None) -> List[Dict]:
        logger.info(f"🎉 {self.name}: Searching events in {city}")
        events = await ticketmaster_service.search_events(city, start_date, end_date, interests)

        # ML Model scoring (BERT/NLP + Knowledge Graph)
        model = _get_model()
        if model and events:
            events = model.score_events(events, interests or [])
            logger.info(f"🎉 {self.name}: BERT+KG scored {len(events)} events")

        # LLM analysis
        if events:
            try:
                analyzed = await openai_service.analyze_events(city, events, interests or [])
                if analyzed:
                    events = self._merge_analysis(events, analyzed)
            except Exception as e:
                logger.warning(f"Event analysis error: {e}")

        for ev in events:
            if not ev.get("relevance_score"):
                ev["relevance_score"] = 0.5

        events.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        logger.info(f"🎉 {self.name}: Found {len(events)} events")
        return events

    def _merge_analysis(self, events, analyzed):
        anal_map = {a.get("name", "").lower(): a for a in analyzed}
        for ev in events:
            a = anal_map.get(ev["name"].lower(), {})
            if a:
                llm_score = a.get("relevance_score", 0)
                if llm_score > 1:
                    llm_score = llm_score / 100.0
                ml_score = ev.get("relevance_score", 0.5)
                # Blend: 50% ML + 50% LLM
                ev["relevance_score"] = round(0.5 * ml_score + 0.5 * llm_score, 3)
                if a.get("recommendation"):
                    ev["description"] = a["recommendation"]
        return events

event_agent = EventAgent()
