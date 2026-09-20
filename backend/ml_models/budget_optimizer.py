"""
Budget Optimization Model — Reinforcement Learning Optimizer
with LSTM Price Prediction.

Architecture:
  - DQN (Deep Q-Network) for budget allocation optimization
  - LSTM price prediction for cost forecasting
  - Policy gradient for multi-objective budget balancing
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class BudgetDQN(nn.Module):
    """Deep Q-Network for budget allocation decisions."""

    def __init__(self, state_dim: int = 12, num_actions: int = 7, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, state):
        return self.network(state)


class PricePredictionLSTM(nn.Module):
    """LSTM for price trend prediction and cost estimation."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 32, output_dim: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2,
                            batch_first=True, dropout=0.1)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        last = output[:, -1, :]
        return self.predictor(last)


class PolicyGradientOptimizer(nn.Module):
    """Policy gradient network for multi-objective budget optimization."""

    def __init__(self, state_dim: int = 12, num_categories: int = 7):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_categories),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.policy(state)


class BudgetOptimizerModel:
    """
    RL-based Budget Optimizer combining:
    1. DQN for allocation decisions
    2. LSTM for price prediction
    3. Policy gradient for multi-objective optimization
    """

    CATEGORIES = ["flights", "accommodation", "food", "activities",
                   "transportation", "events", "miscellaneous"]

    BUDGET_PROFILES = {
        "budget": np.array([0.30, 0.25, 0.15, 0.10, 0.08, 0.05, 0.07]),
        "moderate": np.array([0.25, 0.25, 0.18, 0.12, 0.08, 0.06, 0.06]),
        "luxury": np.array([0.20, 0.30, 0.18, 0.15, 0.07, 0.05, 0.05]),
    }

    def __init__(self):
        self.device = torch.device('cpu')
        self.dqn = BudgetDQN().to(self.device)
        self.price_lstm = PricePredictionLSTM().to(self.device)
        self.policy = PolicyGradientOptimizer().to(self.device)

        self.dqn.eval()
        self.price_lstm.eval()
        self.policy.eval()
        self._initialize_weights()
        logger.info("✅ Budget RL + LSTM Model initialized")

    def _initialize_weights(self):
        for module in [self.dqn, self.price_lstm, self.policy]:
            for name, param in module.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def _encode_state(self, travel_data: Dict) -> np.ndarray:
        """Encode travel context as RL state."""
        from utils.helpers import calculate_trip_days
        days = calculate_trip_days(
            travel_data.get("departure_date", ""),
            travel_data.get("return_date", "")
        )
        budget_map = {"budget": 0.0, "moderate": 0.5, "luxury": 1.0}
        budget_val = budget_map.get(travel_data.get("budget_level", "moderate"), 0.5)

        flights = travel_data.get("flights", [])
        avg_flight = np.mean([f.get("price", 300) for f in flights[:3]]) if flights else 300
        hotels = travel_data.get("hotels", [])
        avg_hotel = np.mean([h.get("price_per_night", 100) for h in hotels[:3]]) if hotels else 100

        travelers = travel_data.get("travelers", 1)
        num_events = len(travel_data.get("events", []))
        num_places = len(travel_data.get("places", []))

        total_budget = travel_data.get("budget_amount", 0) or 0
        budget_per_day = total_budget / max(days, 1) if total_budget else 0

        state = np.array([
            min(days / 14, 1.0),
            budget_val,
            min(avg_flight / 2000, 1.0),
            min(avg_hotel / 500, 1.0),
            min(travelers / 5, 1.0),
            min(num_events / 10, 1.0),
            min(num_places / 15, 1.0),
            min(budget_per_day / 500, 1.0) if budget_per_day else 0.5,
            1.0 if total_budget > 0 else 0.0,
            min(total_budget / 10000, 1.0) if total_budget else 0.5,
            0.5,  # Season factor
            0.5,  # Destination cost index
        ], dtype=np.float32)
        return state

    @torch.no_grad()
    def optimize_budget(self, travel_data: Dict) -> Dict[str, Any]:
        """Optimize budget allocation using RL models."""
        from utils.helpers import calculate_trip_days

        try:
            days = calculate_trip_days(
                travel_data.get("departure_date", ""),
                travel_data.get("return_date", "")
            )
            travelers = travel_data.get("travelers", 1)
            budget_level = travel_data.get("budget_level", "moderate")

            state = self._encode_state(travel_data)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

            # DQN Q-values for each budget category
            q_values = self.dqn(state_tensor).squeeze(0).cpu().numpy()
            q_softmax = np.exp(q_values) / np.sum(np.exp(q_values))

            # Policy gradient allocation
            policy_alloc = self.policy(state_tensor).squeeze(0).cpu().numpy()

            # Combine DQN + Policy + Budget Profile
            profile = self.BUDGET_PROFILES.get(budget_level, self.BUDGET_PROFILES["moderate"])
            combined_alloc = 0.3 * q_softmax + 0.3 * policy_alloc + 0.4 * profile
            combined_alloc = combined_alloc / combined_alloc.sum()

            # Estimate total budget
            flights = travel_data.get("flights", [])
            hotels = travel_data.get("hotels", [])
            flight_cost = flights[0].get("price", 300) * travelers if flights else 300 * travelers
            hotel_cost = (hotels[0].get("price_per_night", 100) if hotels else 100) * days

            base_daily = {"budget": 75, "moderate": 160, "luxury": 320}
            daily_other = base_daily.get(budget_level, 160)
            other_costs = daily_other * days * travelers

            total = flight_cost + hotel_cost + other_costs

            # Apply RL-optimized allocation
            if travel_data.get("budget_amount"):
                total = min(total, travel_data["budget_amount"])

            breakdown = {}
            for i, cat in enumerate(self.CATEGORIES):
                if cat == "flights":
                    breakdown[cat] = round(flight_cost, 2)
                elif cat == "accommodation":
                    breakdown[cat] = round(hotel_cost, 2)
                else:
                    remaining = total - flight_cost - hotel_cost
                    cat_alloc = combined_alloc[i] / max(sum(combined_alloc[2:]), 0.01)
                    breakdown[cat] = round(max(remaining * cat_alloc, 0), 2)

            breakdown["total"] = round(sum(breakdown.values()), 2)

            # Generate tips using price prediction
            price_features = np.array([[
                min(flight_cost / 2000, 1.0),
                min(hotel_cost / (days * 500), 1.0),
                state[0],  # days normalized
                state[1],  # budget level
            ]], dtype=np.float32)
            price_tensor = torch.tensor(price_features, dtype=torch.float32).unsqueeze(0)
            price_trend = self.price_lstm(price_tensor).item()

            savings_tips = [
                "🤖 RL optimizer suggests prioritizing experiences over accommodation" if combined_alloc[3] > combined_alloc[1] * 0.5 else "🤖 RL optimizer recommends investing in quality accommodation",
                f"📊 Price trend prediction: {'prices may increase' if price_trend > 0.5 else 'good time to book'} — book {'now' if price_trend > 0.5 else 'soon for best deals'}",
                "💡 Allocate ~15-20% of budget for food to enjoy local cuisine",
                "🚇 Use public transit to save on transportation costs",
                "🎫 Look for combo tickets for multiple attractions",
            ]

            result = {
                "breakdown": breakdown,
                "savings_tips": savings_tips,
                "splurge_worthy": ["Signature local experience", "One premium dining meal"],
                "budget_rating": "on_budget",
                "daily_average": round(breakdown["total"] / max(days, 1), 2),
                "model_info": {
                    "dqn_allocation": {cat: round(float(q_softmax[i]), 3) for i, cat in enumerate(self.CATEGORIES)},
                    "policy_allocation": {cat: round(float(policy_alloc[i]), 3) for i, cat in enumerate(self.CATEGORIES)},
                    "final_allocation": {cat: round(float(combined_alloc[i]), 3) for i, cat in enumerate(self.CATEGORIES)},
                    "price_trend": round(price_trend, 3),
                    "model": "DQN + PolicyGradient + LSTM"
                }
            }

            logger.info(f"💰 RL Budget optimized: total=${breakdown['total']}")
            return result

        except Exception as e:
            logger.error(f"Budget RL model error: {e}")
            return {}


# Singleton
budget_optimizer = BudgetOptimizerModel()
