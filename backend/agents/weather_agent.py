"""
Weather Analysis Agent.
Uses LSTM/GRU time-series forecasting + Context-aware activity recommendations.
"""
import logging
from typing import List, Dict
from services.weather_service import weather_service

logger = logging.getLogger(__name__)

_weather_model = None
def _get_model():
    global _weather_model
    if _weather_model is None:
        try:
            from ml_models.weather_forecaster import weather_forecaster
            _weather_model = weather_forecaster
            logger.info("🌦️ LSTM/GRU Weather Model loaded")
        except Exception as e:
            logger.warning(f"Weather ML model load warning: {e}")
    return _weather_model


class WeatherAgent:
    """Agent with LSTM/GRU forecasting + context-aware activity model."""

    def __init__(self):
        self.name = "WeatherAgent"

    async def get_forecast(self, city: str, days: int = 5) -> List[Dict]:
        logger.info(f"🌦 {self.name}: Getting weather for {city}")
        forecast = await weather_service.get_forecast(city, days)

        # ML Model analysis (LSTM/GRU)
        model = _get_model()
        if model and forecast:
            ml_analysis = model.analyze_weather(forecast)
            # Merge ML insights into forecast data
            for i, day in enumerate(forecast):
                if i < len(ml_analysis):
                    analysis = ml_analysis[i]
                    day["ml_analysis"] = {
                        "outdoor_suitability": analysis.get("outdoor_suitability", 0.5),
                        "indoor_suitability": analysis.get("indoor_suitability", 0.5),
                        "recommended_activities": analysis.get("recommended_activities", []),
                        "activity_scores": analysis.get("activity_scores", {}),
                        "model": "LSTM-GRU-ContextAware"
                    }
                    # Enhance recommendation with ML insights
                    activities = analysis.get("recommended_activities", [])
                    if activities:
                        day["recommendation"] = (
                            day.get("recommendation", "") +
                            f" | AI suggests: {', '.join(activities[:3])}"
                        )
            logger.info(f"🌦 {self.name}: LSTM/GRU analyzed {len(forecast)} days")

        logger.info(f"🌦 {self.name}: Got {len(forecast)} days of forecast")
        return forecast

    def get_activity_recommendations(self, weather_data: List[Dict],
                                      interests: list = None) -> List[Dict]:
        recs = []
        for day in weather_data:
            cond = day.get("condition", "Clear").lower()
            hi = day.get("temperature_high", 25)
            outdoor_ok = cond in ("clear", "clouds", "snow") and hi < 38

            # Use ML analysis if available
            ml = day.get("ml_analysis", {})
            if ml:
                activities = ml.get("recommended_activities", [])[:5]
            elif outdoor_ok:
                activities = ["Walking tours", "Parks", "Sightseeing", "Outdoor dining"]
                if "adventure" in (interests or []):
                    activities.extend(["Hiking", "Biking"])
            else:
                activities = ["Museums", "Shopping", "Indoor dining", "Spa"]
                if "culture" in (interests or []):
                    activities.extend(["Art galleries", "Cultural shows"])

            recs.append({
                "date": day.get("date", ""),
                "outdoor_suitable": outdoor_ok,
                "recommended_activities": activities[:5],
                "weather_summary": day.get("recommendation", ""),
                "outdoor_score": ml.get("outdoor_suitability", 0.7 if outdoor_ok else 0.3),
                "indoor_score": ml.get("indoor_suitability", 0.3 if outdoor_ok else 0.7),
            })
        return recs

weather_agent = WeatherAgent()
