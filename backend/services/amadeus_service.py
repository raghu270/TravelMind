"""
Amadeus API Service for flight and hotel search.
"""

import logging
import httpx
from typing import List, Dict, Optional
from datetime import datetime
from config import settings

logger = logging.getLogger(__name__)


class AmadeusService:
    """Service for Amadeus flight and hotel search API."""

    def __init__(self):
        self.base_url = settings.AMADEUS_BASE_URL
        self.api_key = settings.AMADEUS_API_KEY
        self.api_secret = settings.AMADEUS_API_SECRET
        self._access_token = None
        self._token_expires = None

    async def _get_access_token(self) -> Optional[str]:
        """Get or refresh OAuth2 access token."""
        if not self.api_key or self.api_key.startswith("your_"):
            logger.warning("Amadeus API key not configured")
            return None

        if self._access_token and self._token_expires and datetime.now() < self._token_expires:
            return self._access_token

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/v1/security/oauth2/token",
                    data={
                        "grant_type": "client_credentials",
                        "client_id": self.api_key,
                        "client_secret": self.api_secret
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=15.0
                )
                if response.status_code == 200:
                    data = response.json()
                    self._access_token = data["access_token"]
                    from datetime import timedelta
                    self._token_expires = datetime.now() + timedelta(seconds=data.get("expires_in", 1799) - 60)
                    logger.info("✅ Amadeus access token obtained")
                    return self._access_token
                else:
                    logger.error(f"Amadeus auth error: {response.status_code} - {response.text}")
                    return None
        except Exception as e:
            logger.error(f"Amadeus auth exception: {e}")
            return None

    async def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Make authenticated API request."""
        token = await self._get_access_token()
        if not token:
            return None

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}{endpoint}",
                    params=params,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=30.0
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Amadeus API error {endpoint}: {response.status_code}")
                    return None
        except Exception as e:
            logger.error(f"Amadeus request exception: {e}")
            return None

    # ---- IATA code lookup ----
    CITY_IATA_MAP = {
        "new york": "NYC", "los angeles": "LAX", "chicago": "CHI",
        "london": "LON", "paris": "PAR", "tokyo": "TYO",
        "dubai": "DXB", "singapore": "SIN", "bangkok": "BKK",
        "rome": "ROM", "barcelona": "BCN", "amsterdam": "AMS",
        "mumbai": "BOM", "delhi": "DEL", "sydney": "SYD",
        "san francisco": "SFO", "miami": "MIA", "berlin": "BER",
        "istanbul": "IST", "hong kong": "HKG", "seoul": "ICN",
        "toronto": "YYZ", "bali": "DPS", "denver": "DEN",
        "kuala lumpur": "KUL", "cairo": "CAI", "lisbon": "LIS",
        "prague": "PRG", "vienna": "VIE", "zurich": "ZRH",
        "madrid": "MAD", "athens": "ATH", "dublin": "DUB",
        "moscow": "MOW", "beijing": "PEK", "shanghai": "PVG",
        "mexico city": "MEX", "buenos aires": "BUE", "rio de janeiro": "GIG",
        "cape town": "CPT", "nairobi": "NBO", "marrakech": "RAK",
        "hanoi": "HAN", "ho chi minh": "SGN", "osaka": "KIX",
        "phuket": "HKT", "maldives": "MLE", "male": "MLE",
        "goa": "GOI", "jaipur": "JAI", "kolkata": "CCU",
        "hyderabad": "HYD", "bangalore": "BLR", "chennai": "MAA",
        "washington": "WAS", "boston": "BOS", "seattle": "SEA",
        "las vegas": "LAS", "orlando": "MCO", "atlanta": "ATL",
    }

    def _get_iata_code(self, city: str) -> str:
        """Get IATA code for a city."""
        city_lower = city.lower().strip()
        if len(city_lower) == 3 and city_lower.isupper():
            return city_lower.upper()
        return self.CITY_IATA_MAP.get(city_lower, city[:3].upper())

    async def search_flights(self, origin: str, destination: str,
                              departure_date: str, return_date: str = None,
                              adults: int = 1, max_results: int = 10) -> List[Dict]:
        """Search for flights."""
        origin_code = self._get_iata_code(origin)
        dest_code = self._get_iata_code(destination)

        params = {
            "originLocationCode": origin_code,
            "destinationLocationCode": dest_code,
            "departureDate": departure_date,
            "adults": adults,
            "max": max_results,
            "currencyCode": "USD"
        }
        if return_date:
            params["returnDate"] = return_date

        data = await self._make_request("/v2/shopping/flight-offers", params)

        if not data:
            return self._get_sample_flights(origin, destination, departure_date, return_date)

        flights = []
        for offer in data.get("data", []):
            try:
                itinerary = offer.get("itineraries", [{}])[0]
                segments = itinerary.get("segments", [{}])
                first_seg = segments[0] if segments else {}

                flight = {
                    "airline": first_seg.get("carrierCode", "Unknown"),
                    "flight_number": f"{first_seg.get('carrierCode', 'XX')}{first_seg.get('number', '000')}",
                    "departure_airport": first_seg.get("departure", {}).get("iataCode", origin_code),
                    "arrival_airport": segments[-1].get("arrival", {}).get("iataCode", dest_code) if segments else dest_code,
                    "departure_time": first_seg.get("departure", {}).get("at", ""),
                    "arrival_time": segments[-1].get("arrival", {}).get("at", "") if segments else "",
                    "duration": itinerary.get("duration", "PT0H"),
                    "price": float(offer.get("price", {}).get("total", 0)),
                    "currency": offer.get("price", {}).get("currency", "USD"),
                    "stops": len(segments) - 1 if segments else 0,
                    "cabin_class": offer.get("travelerPricings", [{}])[0].get("fareDetailsBySegment", [{}])[0].get("cabin", "ECONOMY"),
                    "score": 0.0
                }
                flights.append(flight)
            except (IndexError, KeyError, TypeError) as e:
                logger.warning(f"Error parsing flight offer: {e}")
                continue

        return flights if flights else self._get_sample_flights(origin, destination, departure_date, return_date)

    async def search_hotels(self, city: str, check_in: str, check_out: str,
                             adults: int = 1, max_results: int = 10) -> List[Dict]:
        """Search for hotels in a city."""
        city_code = self._get_iata_code(city)

        # First get hotel list by city
        params = {
            "cityCode": city_code,
            "radius": 30,
            "radiusUnit": "KM",
            "ratings": "3,4,5",
        }

        data = await self._make_request("/v1/reference-data/locations/hotels/by-city", params)

        if not data:
            return self._get_sample_hotels(city, check_in, check_out)

        hotels = []
        for hotel_data in data.get("data", [])[:max_results]:
            try:
                geo = hotel_data.get("geoCode", {})
                hotel = {
                    "name": hotel_data.get("name", "Unknown Hotel"),
                    "address": hotel_data.get("address", {}).get("countryCode", city),
                    "rating": 4.0,
                    "stars": int(hotel_data.get("rating", 3)),
                    "price_per_night": 0,
                    "total_price": 0,
                    "currency": "USD",
                    "amenities": [],
                    "distance_to_center": float(hotel_data.get("distance", {}).get("value", 0)),
                    "latitude": geo.get("latitude", 0),
                    "longitude": geo.get("longitude", 0),
                    "score": 0.0
                }
                hotels.append(hotel)
            except Exception as e:
                logger.warning(f"Error parsing hotel: {e}")
                continue

        return hotels if hotels else self._get_sample_hotels(city, check_in, check_out)

    def _get_sample_flights(self, origin: str, destination: str,
                             departure_date: str, return_date: str = None) -> List[Dict]:
        """Generate sample flight data for demo/fallback."""
        import random
        airlines = [
            ("AA", "American Airlines"), ("UA", "United Airlines"),
            ("DL", "Delta Airlines"), ("EK", "Emirates"),
            ("BA", "British Airways"), ("LH", "Lufthansa"),
            ("SQ", "Singapore Airlines"), ("QR", "Qatar Airways"),
        ]
        flights = []
        for i in range(6):
            code, name = airlines[i % len(airlines)]
            base_price = random.randint(250, 1200)
            flights.append({
                "airline": f"{name} ({code})",
                "flight_number": f"{code}{random.randint(100, 999)}",
                "departure_airport": self._get_iata_code(origin),
                "arrival_airport": self._get_iata_code(destination),
                "departure_time": f"{departure_date}T{random.choice(['06', '08', '10', '14', '16', '20'])}:00:00",
                "arrival_time": f"{departure_date}T{random.choice(['12', '15', '18', '22', '23'])}:30:00",
                "duration": f"PT{random.randint(2, 14)}H{random.randint(0, 59)}M",
                "price": base_price + (i * random.randint(20, 80)),
                "currency": "USD",
                "stops": random.choice([0, 0, 1, 1, 2]),
                "cabin_class": "economy",
                "score": round(random.uniform(0.6, 0.95), 2)
            })
        return sorted(flights, key=lambda x: x["price"])

    def _get_sample_hotels(self, city: str, check_in: str, check_out: str) -> List[Dict]:
        """Generate sample hotel data for demo/fallback."""
        import random
        hotel_names = [
            f"Grand {city} Hotel", f"{city} Marriott Suites", f"The Ritz-Carlton {city}",
            f"Hilton {city} Downtown", f"Hyatt Regency {city}", f"Four Seasons {city}",
            f"Holiday Inn {city} Central", f"Boutique Hotel {city}", f"Novotel {city}",
        ]
        amenities_pool = ["WiFi", "Pool", "Spa", "Gym", "Restaurant", "Bar",
                          "Room Service", "Airport Shuttle", "Parking", "Breakfast"]

        hotels = []
        for i, name in enumerate(hotel_names[:8]):
            stars = random.choice([3, 4, 4, 5])
            ppn = random.randint(60, 350) + (stars * 30)
            hotels.append({
                "name": name,
                "address": f"Downtown {city}",
                "rating": round(random.uniform(3.5, 4.9), 1),
                "stars": stars,
                "price_per_night": ppn,
                "total_price": ppn * 3,
                "currency": "USD",
                "amenities": random.sample(amenities_pool, random.randint(4, 8)),
                "distance_to_center": round(random.uniform(0.5, 8.0), 1),
                "latitude": round(random.uniform(30, 50), 4),
                "longitude": round(random.uniform(-120, 30), 4),
                "score": round(random.uniform(0.6, 0.95), 2)
            })
        return sorted(hotels, key=lambda x: x["price_per_night"])


# Singleton instance
amadeus_service = AmadeusService()
