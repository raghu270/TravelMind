"""
Utility helpers for the Travel Agentic AI System.
"""
import uuid
import math
from datetime import datetime, timedelta
from typing import List, Tuple, Dict


def generate_request_id() -> str:
    return f"TR-{uuid.uuid4().hex[:12].upper()}"


def calculate_trip_days(departure: str, return_date: str) -> int:
    try:
        d1 = datetime.strptime(departure, "%Y-%m-%d")
        d2 = datetime.strptime(return_date, "%Y-%m-%d")
        return max((d2 - d1).days, 1)
    except ValueError:
        return 3


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def optimize_route(places: List[Dict]) -> List[Dict]:
    """Simple nearest-neighbor route optimization."""
    if len(places) <= 2:
        return places
    visited = [places[0]]
    remaining = places[1:]
    while remaining:
        current = visited[-1]
        nearest = min(remaining, key=lambda p: haversine_distance(
            current.get("latitude", 0), current.get("longitude", 0),
            p.get("latitude", 0), p.get("longitude", 0)
        ))
        visited.append(nearest)
        remaining.remove(nearest)
    return visited


def estimate_travel_time(distance_km: float, mode: str = "driving") -> float:
    speeds = {"walking": 5, "driving": 40, "transit": 25, "cycling": 15}
    speed = speeds.get(mode, 40)
    return round(distance_km / speed * 60, 0)  # minutes


def format_duration(iso_duration: str) -> str:
    import re
    match = re.match(r'PT(\d+)H(\d+)M', iso_duration)
    if match:
        h, m = int(match.group(1)), int(match.group(2))
        return f"{h}h {m}m"
    match = re.match(r'PT(\d+)H', iso_duration)
    if match:
        return f"{int(match.group(1))}h"
    return iso_duration


def get_date_range(start: str, end: str) -> List[str]:
    try:
        d1 = datetime.strptime(start, "%Y-%m-%d")
        d2 = datetime.strptime(end, "%Y-%m-%d")
        return [(d1 + timedelta(days=i)).strftime("%Y-%m-%d")
                for i in range((d2 - d1).days + 1)]
    except ValueError:
        return [start]
