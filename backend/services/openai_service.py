"""
OpenAI LLM Service for reasoning, NLP processing, and itinerary generation.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from config import settings

logger = logging.getLogger(__name__)


class OpenAIService:
    """Service for OpenAI GPT interactions."""

    def __init__(self):
        self.client = None
        if settings.OPENAI_API_KEY and not settings.OPENAI_API_KEY.startswith("your_"):
            self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL

    async def _chat_completion(self, messages: List[Dict], temperature: float = 0.7,
                                max_tokens: int = 4000, response_format: str = None) -> str:
        """Make a chat completion request."""
        if not self.client:
            logger.warning("OpenAI client not initialized. Using fallback.")
            return self._get_fallback_response(messages)
        try:
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_format == "json":
                params["response_format"] = {"type": "json_object"}

            response = await self.client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._get_fallback_response(messages)

    def _get_fallback_response(self, messages: List[Dict]) -> str:
        """Generate fallback response when API is unavailable."""
        last_msg = messages[-1]["content"] if messages else ""
        if "json" in last_msg.lower() or "extract" in last_msg.lower():
            return json.dumps({
                "interests": ["culture", "food", "sightseeing"],
                "budget_level": "moderate",
                "travel_style": "balanced",
                "priorities": ["experiences", "local culture"],
                "special_needs": []
            })
        return "AI service is currently initializing. Please configure your OpenAI API key."

    async def extract_preferences(self, user_input: str) -> Dict[str, Any]:
        """Extract structured travel preferences from natural language input."""
        messages = [
            {
                "role": "system",
                "content": """You are a travel preference extraction AI. Analyze the user's travel request 
                and extract structured preferences. Return a JSON object with these fields:
                - interests: list of interest categories (culture, food, adventure, nature, nightlife, shopping, history, art, sports, wellness, family, photography)
                - budget_level: "budget", "moderate", or "luxury"
                - travel_style: "relaxed", "balanced", or "packed"
                - priorities: list of top priorities
                - special_needs: list of special requirements
                - event_interests: list of event types they'd enjoy
                - food_preferences: list of cuisine preferences
                - accommodation_style: "hostel", "hotel", "boutique", "resort", or "apartment"
                """
            },
            {
                "role": "user",
                "content": f"Extract travel preferences from this request: {user_input}"
            }
        ]
        try:
            result = await self._chat_completion(messages, temperature=0.3, response_format="json")
            return json.loads(result)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Preference extraction error: {e}")
            return {
                "interests": ["culture", "food", "sightseeing"],
                "budget_level": "moderate",
                "travel_style": "balanced",
                "priorities": ["experiences"],
                "special_needs": [],
                "event_interests": ["concerts", "festivals"],
                "food_preferences": ["local cuisine"],
                "accommodation_style": "hotel"
            }

    async def generate_place_recommendations(self, destination: str, interests: List[str],
                                              places_data: List[Dict]) -> List[Dict]:
        """Use LLM to rank and enhance place recommendations."""
        places_text = json.dumps(places_data[:20], indent=2) if places_data else "No places data available"
        messages = [
            {
                "role": "system",
                "content": """You are a travel expert AI. Analyze the available places and rank them based on 
                the user's interests. For each place, provide a relevance score (0-100) and a brief personalized 
                recommendation. Return a JSON array of objects with: name, category, score, recommendation, 
                best_time_to_visit, estimated_duration_hours, insider_tip."""
            },
            {
                "role": "user",
                "content": f"""Destination: {destination}
                User Interests: {', '.join(interests)}
                Available Places: {places_text}
                
                Rank and enhance these place recommendations. Return JSON array."""
            }
        ]
        try:
            result = await self._chat_completion(messages, temperature=0.5, response_format="json")
            parsed = json.loads(result)
            return parsed if isinstance(parsed, list) else parsed.get("places", parsed.get("recommendations", []))
        except Exception as e:
            logger.error(f"Place recommendation error: {e}")
            return []

    async def analyze_events(self, destination: str, events_data: List[Dict],
                              interests: List[str]) -> List[Dict]:
        """Analyze and rank events based on user interests."""
        events_text = json.dumps(events_data[:15], indent=2) if events_data else "No events data available"
        messages = [
            {
                "role": "system",
                "content": """You are an event recommendation AI. Analyze available events at the destination 
                and rank them based on user interests. Return a JSON array with: name, event_type, 
                relevance_score (0-100), recommendation, why_attend."""
            },
            {
                "role": "user",
                "content": f"""Destination: {destination}
                User Interests: {', '.join(interests)}
                Available Events: {events_text}
                
                Rank and recommend events. Return JSON array."""
            }
        ]
        try:
            result = await self._chat_completion(messages, temperature=0.5, response_format="json")
            parsed = json.loads(result)
            return parsed if isinstance(parsed, list) else parsed.get("events", [])
        except Exception as e:
            logger.error(f"Event analysis error: {e}")
            return []

    async def generate_itinerary(self, travel_data: Dict[str, Any]) -> List[Dict]:
        """Generate a comprehensive day-wise itinerary."""
        messages = [
            {
                "role": "system",
                "content": """You are an expert travel itinerary planner. Create a detailed day-wise itinerary 
                based on all the travel data provided. Consider weather, events, distances, budget, and user preferences.
                
                Return a JSON object with an "itinerary" key containing an array where each element represents a day with:
                - day_number: int
                - date: string (YYYY-MM-DD)
                - theme: string (e.g., "Cultural Exploration", "Adventure Day")
                - weather_summary: string
                - activities: array of {time, activity, location, description, duration, cost_estimate, category}
                - meals: array of {time, meal_type, suggestion, cuisine, estimated_cost}
                - estimated_cost: total daily cost float
                - travel_tips: array of strings
                - events: array of relevant events for that day
                
                Make the itinerary practical, well-paced, and personalized."""
            },
            {
                "role": "user",
                "content": f"""Create a detailed day-wise itinerary with this travel data:

                Destination: {travel_data.get('destination', 'Unknown')}
                Source: {travel_data.get('source', 'Unknown')}
                Dates: {travel_data.get('departure_date', '')} to {travel_data.get('return_date', '')}
                Interests: {travel_data.get('interests', [])}
                Budget Level: {travel_data.get('budget_level', 'moderate')}
                
                Flights: {json.dumps(travel_data.get('flights', [])[:3], indent=2)}
                Hotels: {json.dumps(travel_data.get('hotels', [])[:3], indent=2)}
                Places: {json.dumps(travel_data.get('places', [])[:10], indent=2)}
                Weather: {json.dumps(travel_data.get('weather', []), indent=2)}
                Events: {json.dumps(travel_data.get('events', [])[:5], indent=2)}
                
                Generate a complete itinerary. Return JSON."""
            }
        ]
        try:
            result = await self._chat_completion(messages, temperature=0.7, max_tokens=6000,
                                                  response_format="json")
            parsed = json.loads(result)
            return parsed.get("itinerary", parsed) if isinstance(parsed, dict) else parsed
        except Exception as e:
            logger.error(f"Itinerary generation error: {e}")
            return []

    async def generate_travel_summary(self, travel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an AI travel summary with insights."""
        messages = [
            {
                "role": "system",
                "content": """You are a travel advisor AI. Generate a comprehensive travel summary with 
                actionable insights. Return a JSON object with:
                - summary: string (2-3 paragraph overview)
                - insights: array of strings (key travel insights)
                - personalization_notes: array of strings (personalized tips)
                - packing_suggestions: array of strings
                - cultural_tips: array of strings
                - safety_tips: array of strings
                - best_experiences: array of strings (must-do experiences)"""
            },
            {
                "role": "user",
                "content": f"""Generate travel summary for:
                Destination: {travel_data.get('destination')}
                Dates: {travel_data.get('departure_date')} to {travel_data.get('return_date')}
                Interests: {travel_data.get('interests', [])}
                Weather: {json.dumps(travel_data.get('weather', [])[:3])}
                Budget: {travel_data.get('budget_level', 'moderate')}
                
                Return JSON with travel summary and insights."""
            }
        ]
        try:
            result = await self._chat_completion(messages, temperature=0.7, response_format="json")
            return json.loads(result)
        except Exception as e:
            logger.error(f"Travel summary error: {e}")
            return {
                "summary": f"Exciting trip planned to {travel_data.get('destination', 'your destination')}!",
                "insights": ["Pack comfortable walking shoes", "Try local cuisine"],
                "personalization_notes": ["Trip customized to your interests"],
                "packing_suggestions": ["Weather-appropriate clothing"],
                "cultural_tips": ["Respect local customs"],
                "safety_tips": ["Keep valuables secure"],
                "best_experiences": ["Explore local markets"]
            }

    async def chat_response(self, message: str, chat_history: List[Dict],
                             travel_context: Dict = None) -> str:
        """Generate conversational response about travel planning."""
        system_msg = """You are an expert AI travel assistant. Help users plan their trips with personalized 
        recommendations. Be friendly, knowledgeable, and provide actionable advice. If the user asks about 
        specific destinations, provide detailed local knowledge."""

        if travel_context:
            system_msg += f"\n\nCurrent travel context:\n{json.dumps(travel_context, indent=2)}"

        messages = [{"role": "system", "content": system_msg}]
        messages.extend(chat_history[-10:])  # Last 10 messages for context
        messages.append({"role": "user", "content": message})

        return await self._chat_completion(messages, temperature=0.8, max_tokens=2000)

    async def optimize_budget(self, travel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate budget optimization recommendations."""
        messages = [
            {
                "role": "system",
                "content": """You are a travel budget optimization AI. Analyze the travel data and provide 
                a detailed budget breakdown with optimization tips. Return a JSON object with:
                - breakdown: {flights, accommodation, food, activities, transportation, events, miscellaneous, total}
                - savings_tips: array of specific money-saving tips
                - splurge_worthy: array of experiences worth spending more on
                - budget_rating: "under_budget", "on_budget", or "over_budget"
                - daily_average: float"""
            },
            {
                "role": "user",
                "content": f"""Optimize budget for:
                Destination: {travel_data.get('destination')}
                Duration: {travel_data.get('departure_date')} to {travel_data.get('return_date')}
                Budget Level: {travel_data.get('budget_level', 'moderate')}
                Budget Amount: {travel_data.get('budget_amount', 'Not specified')}
                Flight Prices: {json.dumps([f.get('price', 0) for f in travel_data.get('flights', [])[:3]])}
                Hotel Prices: {json.dumps([h.get('price_per_night', 0) for h in travel_data.get('hotels', [])[:3]])}
                
                Return JSON budget analysis."""
            }
        ]
        try:
            result = await self._chat_completion(messages, temperature=0.5, response_format="json")
            return json.loads(result)
        except Exception as e:
            logger.error(f"Budget optimization error: {e}")
            return {
                "breakdown": {"flights": 0, "accommodation": 0, "food": 0, "activities": 0,
                              "transportation": 0, "events": 0, "miscellaneous": 0, "total": 0},
                "savings_tips": ["Compare prices across platforms"],
                "splurge_worthy": ["Local experiences"],
                "budget_rating": "on_budget",
                "daily_average": 0
            }


# Singleton instance
openai_service = OpenAIService()
