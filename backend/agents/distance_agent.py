"""
Distance Optimization Agent.
Uses Graph Attention Networks (GAT) + RL for route optimization.
"""
import logging
from typing import List, Dict
from utils.helpers import haversine_distance, optimize_route, estimate_travel_time

logger = logging.getLogger(__name__)

_distance_model = None
def _get_model():
    global _distance_model
    if _distance_model is None:
        try:
            from ml_models.distance_optimizer import distance_optimizer
            _distance_model = distance_optimizer
            logger.info("📏 GNN+RL Distance Optimizer loaded")
        except Exception as e:
            logger.warning(f"Distance ML model load warning: {e}")
    return _distance_model


class DistanceAgent:
    """Agent with Graph Neural Network + RL for route optimization."""

    def __init__(self):
        self.name = "DistanceAgent"

    def optimize_daily_route(self, places: List[Dict], hotel: Dict = None) -> List[Dict]:
        logger.info(f"📏 {self.name}: Optimizing route for {len(places)} places")
        if not places:
            return []

        # Try GNN + RL model
        model = _get_model()
        if model and len(places) > 2:
            try:
                result = model.optimize_route(places)
                optimized = result.get("places", places)
                total = result.get("total_distance", 0)
                logger.info(f"📏 {self.name}: GNN+RL optimized {total:.1f}km for {len(places)} places")
                return optimized
            except Exception as e:
                logger.warning(f"GNN model error: {e}, falling back to heuristic")

        # Fallback: heuristic optimization
        for i, p in enumerate(places):
            if i > 0:
                d = haversine_distance(
                    places[i-1].get("latitude", 0), places[i-1].get("longitude", 0),
                    p.get("latitude", 0), p.get("longitude", 0)
                )
                p["distance_from_prev"] = round(d, 2)
                p["travel_time_min"] = estimate_travel_time(d, "driving")
            else:
                p["distance_from_prev"] = 0
                p["travel_time_min"] = 0

        optimized = optimize_route(places)
        total = sum(p.get("distance_from_prev", 0) for p in optimized)
        logger.info(f"📏 {self.name}: Heuristic route distance: {total:.1f} km")
        return optimized

    def cluster_places_by_day(self, places: List[Dict], num_days: int) -> List[List[Dict]]:
        if not places or num_days <= 0:
            return []

        # Use GNN model for smarter clustering if available
        model = _get_model()
        if model:
            try:
                result = model.optimize_route(places)
                optimized = result.get("places", places)
            except Exception:
                optimized = optimize_route(places)
        else:
            optimized = optimize_route(places)

        per_day = max(len(optimized) // num_days, 1)
        clusters = []
        for i in range(num_days):
            start = i * per_day
            end = start + per_day if i < num_days - 1 else len(optimized)
            if start < len(optimized):
                clusters.append(optimized[start:end])
        return clusters

    def calculate_total_distance(self, places: List[Dict]) -> float:
        total = 0
        for i in range(1, len(places)):
            total += haversine_distance(
                places[i-1].get("latitude", 0), places[i-1].get("longitude", 0),
                places[i].get("latitude", 0), places[i].get("longitude", 0)
            )
        return round(total, 2)

distance_agent = DistanceAgent()
