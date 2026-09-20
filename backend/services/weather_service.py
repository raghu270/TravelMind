"""
OpenWeather API Service for weather forecasting.
"""
import logging
import httpx
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from config import settings

logger = logging.getLogger(__name__)


class WeatherService:
    WEATHER_ACTIVITY_MAP = {
        "clear": {"outdoor": True, "suggestions": ["Sightseeing", "Walking tours", "Outdoor dining"]},
        "clouds": {"outdoor": True, "suggestions": ["City tours", "Outdoor markets", "Photography"]},
        "rain": {"outdoor": False, "suggestions": ["Museums", "Indoor markets", "Spas"]},
        "drizzle": {"outdoor": False, "suggestions": ["Cafes", "Shopping malls"]},
        "thunderstorm": {"outdoor": False, "suggestions": ["Stay indoors", "Hotel spa"]},
        "snow": {"outdoor": True, "suggestions": ["Skiing", "Snow activities", "Warm cafes"]},
    }

    def __init__(self):
        self.base_url = settings.OPENWEATHER_BASE_URL
        self.api_key = settings.OPENWEATHER_API_KEY

    async def get_forecast(self, city: str, days: int = 5) -> List[Dict]:
        if not self.api_key or self.api_key.startswith("your_"):
            return self._get_sample_forecast(city, days)
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.base_url}/forecast",
                    params={"q": city, "cnt": days * 8, "appid": self.api_key, "units": "metric"}, timeout=15.0)
                if resp.status_code != 200:
                    return self._get_sample_forecast(city, days)
                data = resp.json()
        except Exception as e:
            logger.error(f"Weather API error: {e}")
            return self._get_sample_forecast(city, days)

        daily = {}
        for item in data.get("list", []):
            dt = datetime.fromtimestamp(item["dt"])
            dk = dt.strftime("%Y-%m-%d")
            if dk not in daily:
                daily[dk] = {"temps": [], "conditions": [], "descs": [], "humidity": [], "wind": [], "pop": [], "icons": []}
            w = item.get("weather", [{}])[0]
            m = item.get("main", {})
            daily[dk]["temps"].append(m.get("temp", 0))
            daily[dk]["conditions"].append(w.get("main", "Clear"))
            daily[dk]["descs"].append(w.get("description", ""))
            daily[dk]["humidity"].append(m.get("humidity", 0))
            daily[dk]["wind"].append(item.get("wind", {}).get("speed", 0))
            daily[dk]["pop"].append(item.get("pop", 0))
            daily[dk]["icons"].append(w.get("icon", "01d"))

        forecasts = []
        for dk, d in sorted(daily.items())[:days]:
            cond = max(set(d["conditions"]), key=d["conditions"].count)
            act = self.WEATHER_ACTIVITY_MAP.get(cond.lower(), self.WEATHER_ACTIVITY_MAP["clear"])
            rec = self._make_rec(cond.lower(), max(d["temps"]), act)
            forecasts.append({
                "date": dk, "temperature_high": round(max(d["temps"]), 1),
                "temperature_low": round(min(d["temps"]), 1), "condition": cond,
                "description": max(set(d["descs"]), key=d["descs"].count),
                "humidity": round(sum(d["humidity"]) / len(d["humidity"]), 1),
                "wind_speed": round(sum(d["wind"]) / len(d["wind"]), 1),
                "precipitation_chance": round(max(d["pop"]) * 100),
                "icon": d["icons"][len(d["icons"]) // 2], "recommendation": rec
            })
        return forecasts or self._get_sample_forecast(city, days)

    def _make_rec(self, cond, high, act):
        sugg = ", ".join(act.get("suggestions", [])[:3])
        if cond in ("rain", "thunderstorm", "drizzle"):
            return f"☔ Rainy — Indoor activities recommended: {sugg}"
        if cond == "snow":
            return f"❄️ Snowy — Dress warm! Try: {sugg}"
        if cond == "clear" and high > 30:
            return f"☀️ Hot & sunny — Stay hydrated! Try: {sugg}"
        return f"🌤 {cond.title()} — Great for: {sugg}"

    def _get_sample_forecast(self, city, days):
        import random
        conds = [("Clear","clear sky","01d"),("Clouds","scattered clouds","03d"),
                 ("Clear","clear sky","01d"),("Rain","light rain","10d"),
                 ("Clouds","overcast clouds","04d"),("Clear","clear sky","02d")]
        forecasts = []
        base = datetime.now()
        for i in range(min(days, 7)):
            dt = base + timedelta(days=i)
            c = conds[i % len(conds)]
            hi = round(random.uniform(18, 32), 1)
            lo = round(hi - random.uniform(5, 12), 1)
            act = self.WEATHER_ACTIVITY_MAP.get(c[0].lower(), self.WEATHER_ACTIVITY_MAP["clear"])
            forecasts.append({
                "date": dt.strftime("%Y-%m-%d"), "temperature_high": hi, "temperature_low": lo,
                "condition": c[0], "description": c[1],
                "humidity": round(random.uniform(40, 80), 1),
                "wind_speed": round(random.uniform(2, 15), 1),
                "precipitation_chance": round(random.uniform(0, 60) if "rain" in c[1] else random.uniform(0, 20)),
                "icon": c[2], "recommendation": self._make_rec(c[0].lower(), hi, act)
            })
        return forecasts

weather_service = WeatherService()
