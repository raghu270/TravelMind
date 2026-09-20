"""
Test script to verify all API connections for Travel Agent AI.
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from config import settings

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"


async def test_mongodb():
    print("\n--- 1. MongoDB ---")
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        client = AsyncIOMotorClient(settings.MONGODB_URI, serverSelectionTimeoutMS=5000)
        await client.admin.command("ping")
        db_names = await client.list_database_names()
        print(f"  {PASS} Connected to MongoDB")
        print(f"       URI: {settings.MONGODB_URI}")
        print(f"       Database: {settings.MONGODB_DB_NAME}")
        print(f"       Existing DBs: {db_names}")
        client.close()
        return True
    except Exception as e:
        print(f"  {FAIL} MongoDB: {e}")
        return False


async def test_openai():
    print("\n--- 2. OpenAI API ---")
    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.startswith("your_"):
        print(f"  {FAIL} API key not configured")
        return False
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": "Say 'API working' in 2 words"}],
            max_tokens=10
        )
        reply = response.choices[0].message.content.strip()
        print(f"  {PASS} OpenAI connected")
        print(f"       Model: {settings.OPENAI_MODEL}")
        print(f"       Test response: {reply}")
        return True
    except Exception as e:
        print(f"  {FAIL} OpenAI: {e}")
        return False


async def test_amadeus():
    print("\n--- 3. Amadeus API (Flights & Hotels) ---")
    if not settings.AMADEUS_API_KEY or settings.AMADEUS_API_KEY.startswith("your_"):
        print(f"  {FAIL} API key not configured")
        return False
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.AMADEUS_BASE_URL}/v1/security/oauth2/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": settings.AMADEUS_API_KEY,
                    "client_secret": settings.AMADEUS_API_SECRET
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=15.0
            )
            if response.status_code == 200:
                data = response.json()
                print(f"  {PASS} Amadeus authenticated")
                print(f"       Token type: {data.get('type', 'N/A')}")
                print(f"       Expires in: {data.get('expires_in', 'N/A')}s")
                return True
            else:
                print(f"  {FAIL} Amadeus auth failed: HTTP {response.status_code}")
                print(f"       Response: {response.text[:200]}")
                return False
    except Exception as e:
        print(f"  {FAIL} Amadeus: {e}")
        return False


async def test_foursquare():
    print("\n--- 4. Foursquare API (Places) ---")
    if not settings.FOURSQUARE_API_KEY or settings.FOURSQUARE_API_KEY.startswith("your_"):
        print(f"  {FAIL} API key not configured")
        return False
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.FOURSQUARE_BASE_URL}/places/search",
                params={"near": "Paris", "limit": 1},
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {settings.FOURSQUARE_API_KEY}",
                    "X-Places-Api-Version": "2025-06-17"
                },
                timeout=15.0
            )
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                place_name = results[0].get("name", "N/A") if results else "No results"
                print(f"  {PASS} Foursquare connected")
                print(f"       Test search (Paris): {place_name}")
                print(f"       Results returned: {len(results)}")
                return True
            else:
                print(f"  {FAIL} Foursquare: HTTP {response.status_code}")
                print(f"       Response: {response.text[:200]}")
                return False
    except Exception as e:
        print(f"  {FAIL} Foursquare: {e}")
        return False


async def test_openweather():
    print("\n--- 5. OpenWeather API ---")
    if not settings.OPENWEATHER_API_KEY or settings.OPENWEATHER_API_KEY.startswith("your_"):
        print(f"  {FAIL} API key not configured")
        return False
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.OPENWEATHER_BASE_URL}/weather",
                params={
                    "q": "London",
                    "appid": settings.OPENWEATHER_API_KEY,
                    "units": "metric"
                },
                timeout=15.0
            )
            if response.status_code == 200:
                data = response.json()
                temp = data.get("main", {}).get("temp", "N/A")
                weather = data.get("weather", [{}])[0].get("description", "N/A")
                print(f"  {PASS} OpenWeather connected")
                print(f"       Test city (London): {temp}C, {weather}")
                return True
            else:
                print(f"  {FAIL} OpenWeather: HTTP {response.status_code}")
                print(f"       Response: {response.text[:200]}")
                return False
    except Exception as e:
        print(f"  {FAIL} OpenWeather: {e}")
        return False


async def test_ticketmaster():
    print("\n--- 6. Ticketmaster API (Events) ---")
    if not settings.TICKETMASTER_API_KEY or settings.TICKETMASTER_API_KEY.startswith("your_"):
        print(f"  {FAIL} API key not configured")
        return False
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.TICKETMASTER_BASE_URL}/events.json",
                params={
                    "apikey": settings.TICKETMASTER_API_KEY,
                    "city": "New York",
                    "size": 1
                },
                timeout=15.0
            )
            if response.status_code == 200:
                data = response.json()
                events = data.get("_embedded", {}).get("events", [])
                event_name = events[0].get("name", "N/A") if events else "No events found"
                total = data.get("page", {}).get("totalElements", 0)
                print(f"  {PASS} Ticketmaster connected")
                print(f"       Test city (New York): {event_name}")
                print(f"       Total events available: {total}")
                return True
            else:
                print(f"  {FAIL} Ticketmaster: HTTP {response.status_code}")
                print(f"       Response: {response.text[:200]}")
                return False
    except Exception as e:
        print(f"  {FAIL} Ticketmaster: {e}")
        return False


async def main():
    print("=" * 55)
    print("  Travel Agent AI - API Connection Test")
    print("=" * 55)

    results = {}
    results["MongoDB"] = await test_mongodb()
    results["OpenAI"] = await test_openai()
    results["Amadeus"] = await test_amadeus()
    results["Foursquare"] = await test_foursquare()
    results["OpenWeather"] = await test_openweather()
    results["Ticketmaster"] = await test_ticketmaster()

    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, status in results.items():
        icon = PASS if status else FAIL
        print(f"  {icon} {name}")
    print(f"\n  Result: {passed}/{total} APIs connected successfully")
    print("=" * 55)


if __name__ == "__main__":
    asyncio.run(main())
