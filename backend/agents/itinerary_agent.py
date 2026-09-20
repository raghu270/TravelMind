"""
Itinerary Planning Agent.
Generates day-wise travel itineraries using LLM reasoning.
"""
import logging
from typing import List, Dict, Any
from services.openai_service import openai_service

logger = logging.getLogger(__name__)


class ItineraryAgent:
    def __init__(self):
        self.name = "ItineraryAgent"

    async def generate(self, travel_data: Dict[str, Any]) -> List[Dict]:
        logger.info(f"📋 {self.name}: Generating itinerary")
        try:
            itinerary = await openai_service.generate_itinerary(travel_data)
            if itinerary:
                logger.info(f"📋 {self.name}: Generated {len(itinerary)} day itinerary")
                return itinerary
        except Exception as e:
            logger.warning(f"LLM itinerary error: {e}")

        return self._generate_fallback(travel_data)

    def _generate_fallback(self, data: Dict) -> List[Dict]:
        from utils.helpers import calculate_trip_days, get_date_range
        days = calculate_trip_days(data.get("departure_date", ""), data.get("return_date", ""))
        dates = get_date_range(data.get("departure_date", ""), data.get("return_date", ""))
        places = data.get("places", [])
        weather = data.get("weather", [])
        events = data.get("events", [])
        dest = data.get("destination", "destination")

        itinerary = []
        for i in range(days):
            dt = dates[i] if i < len(dates) else f"Day {i+1}"
            w = weather[i] if i < len(weather) else {}
            day_places = places[i*2:(i+1)*2] if places else []
            day_events = [e for e in events if e.get("date", "") == dt]

            activities = []
            if i == 0:
                activities.append({"time": "09:00", "activity": "Arrive & Check-in",
                    "location": dest, "description": "Arrive and settle in", "duration": "2h", "cost_estimate": 0, "category": "travel"})
            for j, p in enumerate(day_places):
                activities.append({"time": f"{10+j*3}:00", "activity": f"Visit {p.get('name', 'Attraction')}",
                    "location": p.get("address", ""), "description": p.get("description", ""), "duration": "2h", "cost_estimate": 15, "category": p.get("category", "attraction")})
            if i == days - 1:
                activities.append({"time": "14:00", "activity": "Departure",
                    "location": dest, "description": "Check out and head to airport", "duration": "3h", "cost_estimate": 0, "category": "travel"})

            meals = [
                {"time": "08:00", "meal_type": "Breakfast", "suggestion": f"Hotel breakfast or local cafe", "cuisine": "Local", "estimated_cost": 15},
                {"time": "12:30", "meal_type": "Lunch", "suggestion": f"Local restaurant in {dest}", "cuisine": "Local", "estimated_cost": 25},
                {"time": "19:00", "meal_type": "Dinner", "suggestion": f"Popular dining spot", "cuisine": "Mixed", "estimated_cost": 35},
            ]

            themes = ["Arrival & Exploration", "Cultural Discovery", "Adventure Day", "Local Experiences", "Scenic Exploration", "Shopping & Leisure", "Departure Day"]
            itinerary.append({
                "day_number": i + 1, "date": dt,
                "theme": themes[i % len(themes)],
                "weather_summary": w.get("recommendation", "Check weather before heading out"),
                "activities": activities, "meals": meals,
                "estimated_cost": 120, "travel_tips": ["Stay hydrated", "Carry a local map"],
                "events": [{"name": e.get("name",""), "time": e.get("time","")} for e in day_events],
            })
        return itinerary

itinerary_agent = ItineraryAgent()
