"""
Configuration module for Travel Agentic AI System.
Loads environment variables and provides centralized config access.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)
else:
    load_dotenv(Path(__file__).parent / ".env.example", override=True)


class Settings:
    """Application settings loaded from environment variables."""

    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Amadeus API
    AMADEUS_API_KEY: str = os.getenv("AMADEUS_API_KEY", "")
    AMADEUS_API_SECRET: str = os.getenv("AMADEUS_API_SECRET", "")
    AMADEUS_BASE_URL: str = "https://test.api.amadeus.com"

    # Foursquare API
    FOURSQUARE_API_KEY: str = os.getenv("FOURSQUARE_API_KEY", "")
    FOURSQUARE_BASE_URL: str = "https://places-api.foursquare.com"

    # OpenWeather API
    OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "")
    OPENWEATHER_BASE_URL: str = "https://api.openweathermap.org/data/2.5"

    # Ticketmaster API
    TICKETMASTER_API_KEY: str = os.getenv("TICKETMASTER_API_KEY", "")
    TICKETMASTER_BASE_URL: str = "https://app.ticketmaster.com/discovery/v2"

    # MongoDB
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "travel_agent_ai")

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"

    # ChromaDB
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    @classmethod
    def validate(cls):
        """Validate that critical API keys are set."""
        warnings = []
        if not cls.OPENAI_API_KEY or cls.OPENAI_API_KEY.startswith("your_"):
            warnings.append("OPENAI_API_KEY is not configured")
        if not cls.AMADEUS_API_KEY or cls.AMADEUS_API_KEY.startswith("your_"):
            warnings.append("AMADEUS_API_KEY is not configured")
        if not cls.FOURSQUARE_API_KEY or cls.FOURSQUARE_API_KEY.startswith("your_"):
            warnings.append("FOURSQUARE_API_KEY is not configured")
        if not cls.OPENWEATHER_API_KEY or cls.OPENWEATHER_API_KEY.startswith("your_"):
            warnings.append("OPENWEATHER_API_KEY is not configured")
        if not cls.TICKETMASTER_API_KEY or cls.TICKETMASTER_API_KEY.startswith("your_"):
            warnings.append("TICKETMASTER_API_KEY is not configured")
        return warnings


settings = Settings()
