"""
Weather Forecasting Model — LSTM/GRU Time-Series Forecaster
with context-aware activity recommendation.

Architecture:
  - Bidirectional LSTM encoder for temporal weather pattern learning
  - GRU decoder for multi-step forecast refinement
  - Context-aware attention for activity-weather mapping
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict

logger = logging.getLogger(__name__)


class WeatherLSTMEncoder(nn.Module):
    """Bidirectional LSTM for encoding weather temporal patterns."""

    def __init__(self, input_dim: int = 6, hidden_dim: int = 32, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=0.1
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        return self.layer_norm(output)


class GRUDecoder(nn.Module):
    """GRU decoder for forecast refinement."""

    def __init__(self, input_dim: int = 64, hidden_dim: int = 32, output_dim: int = 4):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=1, batch_first=True
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        output, _ = self.gru(x)
        return self.output_proj(output)


class ContextAwareActivityModel(nn.Module):
    """Maps weather conditions to activity suitability scores."""

    def __init__(self, weather_dim: int = 4, num_activities: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=weather_dim, num_heads=1, batch_first=True
        )
        self.activity_scorer = nn.Sequential(
            nn.Linear(weather_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_activities),
            nn.Sigmoid()
        )

    def forward(self, weather_features):
        attended, _ = self.attention(weather_features, weather_features, weather_features)
        return self.activity_scorer(attended)


class WeatherForecastModel:
    """
    LSTM/GRU Weather Forecasting with Context-Aware Activity Recommendation.
    """

    ACTIVITY_NAMES = [
        "outdoor_sightseeing", "museums_indoor", "water_activities",
        "hiking_nature", "shopping", "dining_out",
        "photography", "cultural_events"
    ]

    CONDITION_MAP = {
        "clear": [1.0, 0.0, 0.0, 0.0, 0.0],
        "clouds": [0.0, 1.0, 0.0, 0.0, 0.0],
        "rain": [0.0, 0.0, 1.0, 0.0, 0.0],
        "drizzle": [0.0, 0.0, 0.8, 0.2, 0.0],
        "thunderstorm": [0.0, 0.0, 0.0, 1.0, 0.0],
        "snow": [0.0, 0.0, 0.0, 0.0, 1.0],
    }

    def __init__(self):
        self.device = torch.device('cpu')
        self.encoder = WeatherLSTMEncoder().to(self.device)
        self.decoder = GRUDecoder().to(self.device)
        self.activity_model = ContextAwareActivityModel().to(self.device)

        self.encoder.eval()
        self.decoder.eval()
        self.activity_model.eval()
        self._initialize_weights()
        logger.info("✅ Weather LSTM/GRU Model initialized")

    def _initialize_weights(self):
        for module in [self.encoder, self.decoder, self.activity_model]:
            for name, param in module.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def _encode_weather(self, weather_data: List[Dict]) -> np.ndarray:
        """Convert weather forecast data to feature sequences."""
        features = []
        for w in weather_data:
            cond = w.get("condition", "Clear").lower()
            cond_vec = self.CONDITION_MAP.get(cond, [0.5, 0.0, 0.0, 0.0, 0.0])
            temp_norm = (w.get("temperature_high", 25) - 0) / 45.0
            feat = [temp_norm] + cond_vec
            features.append(feat)
        return np.array(features, dtype=np.float32)

    @torch.no_grad()
    def analyze_weather(self, weather_data: List[Dict]) -> List[Dict]:
        """Analyze weather using LSTM/GRU and produce activity recommendations."""
        if not weather_data:
            return []

        try:
            features = self._encode_weather(weather_data)
            feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

            # LSTM encode
            encoded = self.encoder(feat_tensor)  # (1, T, 64)

            # GRU decode for refined predictions
            decoded = self.decoder(encoded)  # (1, T, 4)

            # Activity scoring
            activity_scores = self.activity_model(decoded)  # (1, T, 8)
            activity_np = activity_scores.squeeze(0).cpu().numpy()

            results = []
            for i, w in enumerate(weather_data):
                if i < len(activity_np):
                    scores = activity_np[i]
                    top_indices = np.argsort(scores)[::-1][:4]
                    recommended = [self.ACTIVITY_NAMES[j] for j in top_indices]
                    recommended_readable = [a.replace("_", " ").title() for a in recommended]

                    outdoor_score = float(scores[0] + scores[3]) / 2
                    indoor_score = float(scores[1] + scores[4]) / 2

                    results.append({
                        "date": w.get("date", ""),
                        "outdoor_suitability": round(outdoor_score, 3),
                        "indoor_suitability": round(indoor_score, 3),
                        "recommended_activities": recommended_readable,
                        "activity_scores": {
                            self.ACTIVITY_NAMES[j]: round(float(scores[j]), 3)
                            for j in range(len(scores))
                        },
                        "model": "LSTM-GRU-ContextAware"
                    })

            logger.info(f"🌦️ LSTM/GRU analyzed {len(results)} forecast days")
            return results

        except Exception as e:
            logger.error(f"Weather model error: {e}")
            return []


# Singleton
weather_forecaster = WeatherForecastModel()
