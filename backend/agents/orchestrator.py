"""
Agent Orchestrator — Coordinates all specialized agents for travel planning.
"""
import logging
import asyncio
import time
from typing import Dict, Any
from datetime import datetime

from agents.flight_agent import flight_agent
from agents.hotel_agent import hotel_agent
from agents.place_agent import place_agent
from agents.weather_agent import weather_agent
from agents.event_agent import event_agent
from agents.budget_agent import budget_agent
from agents.distance_agent import distance_agent
from agents.itinerary_agent import itinerary_agent
from services.openai_service import openai_service
from utils.nlp_processor import process_natural_language
from utils.helpers import generate_request_id, calculate_trip_days

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Orchestrates all specialized agents to generate a complete travel plan."""

    async def plan_trip(self, request: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        request_id = generate_request_id()
        logger.info(f"🚀 Starting travel planning: {request_id}")

        # Step 1: Extract preferences
        preferences = {}
        if request.get("natural_language_input"):
            preferences = await openai_service.extract_preferences(
                request["natural_language_input"]
            )
            nlp_data = process_natural_language(request["natural_language_input"])
            if not request.get("interests"):
                request["interests"] = preferences.get("interests", nlp_data.get("interests", []))

        source = request["source"]
        destination = request["destination"]
        dep_date = request["departure_date"]
        ret_date = request["return_date"]
        travelers = request.get("travelers", 1)
        budget_level = request.get("budget_level", "moderate")
        interests = request.get("interests", ["culture", "food"])
        trip_days = calculate_trip_days(dep_date, ret_date)

        logger.info(f"📋 Trip: {source} → {destination}, {trip_days} days, interests: {interests}")

        # Step 2: Parallel data collection from all agents
        flight_task = flight_agent.get_recommendations(
            source, destination, dep_date, ret_date, travelers, budget_level, preferences)
        hotel_task = hotel_agent.get_recommendations(
            destination, dep_date, ret_date, travelers, budget_level, interests)
        place_task = place_agent.get_recommendations(destination, interests)
        weather_task = weather_agent.get_forecast(destination, min(trip_days, 7))
        async def _empty_list():
            return []

        event_task = event_agent.get_recommendations(
            destination, dep_date, ret_date, interests) if request.get("include_events", True) else _empty_list()

        results = await asyncio.gather(
            flight_task, hotel_task, place_task, weather_task, event_task,
            return_exceptions=True
        )

        flights = results[0] if not isinstance(results[0], Exception) else []
        hotels = results[1] if not isinstance(results[1], Exception) else []
        places = results[2] if not isinstance(results[2], Exception) else []
        weather = results[3] if not isinstance(results[3], Exception) else []
        events = results[4] if not isinstance(results[4], Exception) else []

        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(f"Agent {i} error: {r}")

        # Step 3: Distance optimization
        if places:
            places = distance_agent.optimize_daily_route(places)

        # Step 4: Build travel data for itinerary + budget
        travel_data = {
            "source": source, "destination": destination,
            "departure_date": dep_date, "return_date": ret_date,
            "travelers": travelers, "interests": interests,
            "budget_level": budget_level,
            "budget_amount": request.get("budget_amount"),
            "flights": [f for f in flights[:5]],
            "hotels": [h for h in hotels[:5]],
            "places": [p for p in places[:10]],
            "weather": weather,
            "events": [e for e in events[:8]],
        }

        # Step 5: Generate itinerary and budget in parallel
        itin_task = itinerary_agent.generate(travel_data)
        budget_task = budget_agent.optimize(travel_data)
        summary_task = openai_service.generate_travel_summary(travel_data)

        itin_results = await asyncio.gather(
            itin_task, budget_task, summary_task, return_exceptions=True
        )

        itinerary = itin_results[0] if not isinstance(itin_results[0], Exception) else []
        budget_data = itin_results[1] if not isinstance(itin_results[1], Exception) else {}
        summary_data = itin_results[2] if not isinstance(itin_results[2], Exception) else {}

        # Step 6: Compile final response
        processing_time = round(time.time() - start_time, 2)
        budget_breakdown = budget_data.get("breakdown", {})

        response = {
            "request_id": request_id,
            "status": "success",
            "source": source,
            "destination": destination,
            "departure_date": dep_date,
            "return_date": ret_date,
            "travelers": travelers,
            "flights": flights[:6],
            "hotels": hotels[:8],
            "places": places[:12],
            "weather": weather,
            "events": events[:8],
            "itinerary": itinerary,
            "budget": {
                "flights": budget_breakdown.get("flights", 0),
                "accommodation": budget_breakdown.get("accommodation", 0),
                "food": budget_breakdown.get("food", 0),
                "activities": budget_breakdown.get("activities", 0),
                "transportation": budget_breakdown.get("transportation", 0),
                "events": budget_breakdown.get("events", 0),
                "miscellaneous": budget_breakdown.get("miscellaneous", 0),
                "total": budget_breakdown.get("total", 0),
                "currency": "USD",
                "savings_tips": budget_data.get("savings_tips", []),
            },
            "travel_summary": summary_data.get("summary", f"Your trip to {destination} is planned!"),
            "ai_insights": summary_data.get("insights", []),
            "personalization_notes": summary_data.get("personalization_notes", []),
            "generated_at": datetime.utcnow().isoformat(),
            "processing_time": processing_time,
            "data_sources": ["Amadeus", "Foursquare", "OpenWeather", "Ticketmaster", "OpenAI"],
        }

        logger.info(f"✅ Travel plan {request_id} generated in {processing_time}s")
        return response


orchestrator = AgentOrchestrator()
