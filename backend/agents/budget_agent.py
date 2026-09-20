"""
Budget Optimization Agent.
Uses RL (DQN + Policy Gradient) + LSTM price prediction for cost optimization.
"""
import logging
from typing import Dict, List, Any
from services.openai_service import openai_service

logger = logging.getLogger(__name__)

_budget_model = None
def _get_model():
    global _budget_model
    if _budget_model is None:
        try:
            from ml_models.budget_optimizer import budget_optimizer
            _budget_model = budget_optimizer
            logger.info("💰 RL Budget Optimizer loaded")
        except Exception as e:
            logger.warning(f"Budget ML model load warning: {e}")
    return _budget_model

DAILY_ESTIMATES = {
    "budget": {"food": 30, "transport": 15, "activities": 20, "misc": 10},
    "moderate": {"food": 60, "transport": 30, "activities": 50, "misc": 20},
    "luxury": {"food": 120, "transport": 60, "activities": 100, "misc": 40},
}


class BudgetAgent:
    """Agent with DQN + Policy Gradient + LSTM for budget optimization."""

    def __init__(self):
        self.name = "BudgetAgent"

    async def optimize(self, travel_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"💰 {self.name}: Optimizing budget")

        # Try RL model first
        model = _get_model()
        if model:
            try:
                rl_result = model.optimize_budget(travel_data)
                if rl_result and rl_result.get("breakdown", {}).get("total", 0) > 0:
                    logger.info(f"💰 {self.name}: RL model optimized budget")

                    # Try to enhance with LLM tips
                    try:
                        llm_budget = await openai_service.optimize_budget(travel_data)
                        if llm_budget and llm_budget.get("savings_tips"):
                            rl_result["savings_tips"] = llm_budget["savings_tips"][:5]
                    except Exception:
                        pass

                    return rl_result
            except Exception as e:
                logger.warning(f"RL budget error: {e}")

        # Try LLM budget
        try:
            llm_budget = await openai_service.optimize_budget(travel_data)
            if llm_budget and llm_budget.get("breakdown", {}).get("total", 0) > 0:
                return llm_budget
        except Exception as e:
            logger.warning(f"LLM budget error: {e}")

        return self._calculate_budget(travel_data)

    def _calculate_budget(self, data: Dict) -> Dict:
        from utils.helpers import calculate_trip_days
        days = calculate_trip_days(data.get("departure_date", ""), data.get("return_date", ""))
        level = data.get("budget_level", "moderate")
        daily = DAILY_ESTIMATES.get(level, DAILY_ESTIMATES["moderate"])
        travelers = data.get("travelers", 1)

        flights = data.get("flights", [])
        flight_cost = flights[0].get("price", 300) * travelers if flights else 300 * travelers
        hotels = data.get("hotels", [])
        hotel_cost = hotels[0].get("price_per_night", 100) * days if hotels else 100 * days

        breakdown = {
            "flights": round(flight_cost, 2),
            "accommodation": round(hotel_cost, 2),
            "food": round(daily["food"] * days * travelers, 2),
            "activities": round(daily["activities"] * days, 2),
            "transportation": round(daily["transport"] * days, 2),
            "events": round(daily["activities"] * 0.3 * days, 2),
            "miscellaneous": round(daily["misc"] * days, 2),
        }
        breakdown["total"] = round(sum(breakdown.values()), 2)

        return {
            "breakdown": breakdown,
            "savings_tips": [
                "Book flights in advance for better prices",
                "Consider local street food for authentic & affordable meals",
                "Use public transit instead of taxis",
                "Look for free walking tours",
                "Visit attractions during off-peak hours for discounts",
            ],
            "splurge_worthy": ["Local signature experience", "One fine dining meal"],
            "budget_rating": "on_budget",
            "daily_average": round(breakdown["total"] / max(days, 1), 2),
        }

budget_agent = BudgetAgent()
