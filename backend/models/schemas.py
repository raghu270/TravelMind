"""
Pydantic schemas for request/response models.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import date, datetime
from enum import Enum


# ==================== ENUMS ====================

class InterestCategory(str, Enum):
    CULTURE = "culture"
    FOOD = "food"
    ADVENTURE = "adventure"
    NATURE = "nature"
    NIGHTLIFE = "nightlife"
    SHOPPING = "shopping"
    HISTORY = "history"
    ART = "art"
    SPORTS = "sports"
    WELLNESS = "wellness"
    FAMILY = "family"
    PHOTOGRAPHY = "photography"


class BudgetLevel(str, Enum):
    BUDGET = "budget"
    MODERATE = "moderate"
    LUXURY = "luxury"


# ==================== REQUEST MODELS ====================

class TravelRequest(BaseModel):
    """Main travel planning request from the user."""
    source: str = Field(..., description="Source/departure city", min_length=2)
    destination: str = Field(..., description="Destination city", min_length=2)
    departure_date: str = Field(..., description="Departure date (YYYY-MM-DD)")
    return_date: str = Field(..., description="Return date (YYYY-MM-DD)")
    travelers: int = Field(default=1, ge=1, le=10, description="Number of travelers")
    interests: List[str] = Field(default=[], description="User interests")
    budget_level: Optional[str] = Field(default="moderate", description="Budget level")
    budget_amount: Optional[float] = Field(default=None, description="Total budget in USD")
    special_requirements: Optional[str] = Field(default=None, description="Special requirements")
    include_events: bool = Field(default=True, description="Include event recommendations")
    natural_language_input: Optional[str] = Field(default=None, description="Free-form travel request")


class ChatMessage(BaseModel):
    """Chat message for conversational interaction."""
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(default=None, description="Session ID for context")
    travel_context: Optional[Dict[str, Any]] = Field(default=None, description="Current travel context")


# ==================== RESPONSE MODELS ====================

class FlightOption(BaseModel):
    """Single flight option."""
    airline: str = ""
    flight_number: str = ""
    departure_airport: str = ""
    arrival_airport: str = ""
    departure_time: str = ""
    arrival_time: str = ""
    duration: str = ""
    price: float = 0.0
    currency: str = "USD"
    stops: int = 0
    cabin_class: str = "economy"
    booking_url: Optional[str] = None
    score: float = 0.0


class HotelOption(BaseModel):
    """Single hotel option."""
    name: str = ""
    address: str = ""
    rating: float = 0.0
    stars: int = 0
    price_per_night: float = 0.0
    total_price: float = 0.0
    currency: str = "USD"
    amenities: List[str] = []
    distance_to_center: float = 0.0
    image_url: Optional[str] = None
    booking_url: Optional[str] = None
    score: float = 0.0
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class PlaceRecommendation(BaseModel):
    """Tourist place/attraction recommendation."""
    name: str = ""
    category: str = ""
    address: str = ""
    rating: float = 0.0
    description: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    distance: Optional[float] = None
    opening_hours: Optional[str] = None
    price_level: Optional[str] = None
    image_url: Optional[str] = None
    tips: List[str] = []
    score: float = 0.0


class WeatherForecast(BaseModel):
    """Weather forecast for a specific day."""
    date: str = ""
    temperature_high: float = 0.0
    temperature_low: float = 0.0
    condition: str = ""
    description: str = ""
    humidity: float = 0.0
    wind_speed: float = 0.0
    precipitation_chance: float = 0.0
    icon: str = ""
    recommendation: str = ""


class EventRecommendation(BaseModel):
    """Event/activity recommendation."""
    name: str = ""
    event_type: str = ""
    venue: str = ""
    date: str = ""
    time: str = ""
    price_range: str = ""
    description: str = ""
    url: Optional[str] = None
    image_url: Optional[str] = None
    relevance_score: float = 0.0


class ItineraryDay(BaseModel):
    """Single day itinerary."""
    day_number: int = 0
    date: str = ""
    theme: str = ""
    weather_summary: str = ""
    activities: List[Dict[str, Any]] = []
    meals: List[Dict[str, Any]] = []
    estimated_cost: float = 0.0
    travel_tips: List[str] = []
    events: List[Dict[str, Any]] = []


class BudgetBreakdown(BaseModel):
    """Budget breakdown for the trip."""
    flights: float = 0.0
    accommodation: float = 0.0
    food: float = 0.0
    activities: float = 0.0
    transportation: float = 0.0
    events: float = 0.0
    miscellaneous: float = 0.0
    total: float = 0.0
    currency: str = "USD"
    savings_tips: List[str] = []


class TravelPlanResponse(BaseModel):
    """Complete travel plan response."""
    request_id: str = ""
    status: str = "success"
    source: str = ""
    destination: str = ""
    departure_date: str = ""
    return_date: str = ""
    travelers: int = 1

    # Recommendations
    flights: List[FlightOption] = []
    hotels: List[HotelOption] = []
    places: List[PlaceRecommendation] = []
    weather: List[WeatherForecast] = []
    events: List[EventRecommendation] = []

    # Itinerary
    itinerary: List[ItineraryDay] = []

    # Budget
    budget: Optional[BudgetBreakdown] = None

    # AI Insights
    travel_summary: str = ""
    ai_insights: List[str] = []
    personalization_notes: List[str] = []

    # Metadata
    generated_at: str = ""
    processing_time: float = 0.0
    data_sources: List[str] = []


class AgentStatus(BaseModel):
    """Status of an individual agent."""
    agent_name: str
    status: str  # "pending", "running", "completed", "error"
    message: str = ""
    data_count: int = 0
    processing_time: float = 0.0


class PlanningProgress(BaseModel):
    """Real-time planning progress update."""
    request_id: str
    overall_status: str
    progress_percentage: float
    agents: List[AgentStatus] = []
    current_step: str = ""
