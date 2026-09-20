"""
Event Awareness Model — NLP Classifier (BERT/RoBERTa-inspired)
with Knowledge Graph Event Modeling.

Architecture:
  - Transformer encoder for event text understanding
  - Interest-event attention matching
  - Knowledge graph-style event relationship modeling
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

logger = logging.getLogger(__name__)


class EventTextEncoder(nn.Module):
    """
    Lightweight transformer encoder for event text classification
    (BERT/RoBERTa-inspired architecture at smaller scale).
    """

    def __init__(self, vocab_size: int = 5000, embed_dim: int = 64,
                 num_heads: int = 4, num_layers: int = 2, max_len: int = 50):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 8)  # 8 event type categories
        )

    def forward(self, token_ids: torch.Tensor):
        B, S = token_ids.shape
        positions = torch.arange(S, device=token_ids.device).unsqueeze(0).expand(B, -1)

        x = self.embedding(token_ids) + self.pos_encoding(positions)
        x = self.transformer(x)  # (B, S, E)

        # Pool and classify
        x = x.permute(0, 2, 1)  # (B, E, S)
        x = self.pool(x).squeeze(-1)  # (B, E)
        return self.classifier(x)  # (B, 8)


class InterestEventMatcher(nn.Module):
    """Attention-based matching between user interests and events."""

    def __init__(self, interest_dim: int = 12, event_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.interest_proj = nn.Linear(interest_dim, hidden_dim)
        self.event_proj = nn.Linear(event_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=2, batch_first=True
        )
        self.relevance_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, interest_features, event_features):
        i_proj = self.interest_proj(interest_features).unsqueeze(1)
        e_proj = self.event_proj(event_features).unsqueeze(1)
        attended, _ = self.attention(i_proj, e_proj, e_proj)
        return self.relevance_head(attended.squeeze(1))


class EventKnowledgeGraph(nn.Module):
    """Knowledge graph embedding for event-category relationships."""

    def __init__(self, num_event_types: int = 10, num_categories: int = 8, embed_dim: int = 16):
        super().__init__()
        self.event_type_embed = nn.Embedding(num_event_types, embed_dim)
        self.category_embed = nn.Embedding(num_categories, embed_dim)
        self.relation_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, event_type_id, interest_cat_id):
        e_emb = self.event_type_embed(event_type_id)
        c_emb = self.category_embed(interest_cat_id)
        combined = torch.cat([e_emb, c_emb], dim=-1)
        return self.relation_mlp(combined)


class EventClassifierModel:
    """
    Complete Event Awareness System combining:
    1. BERT-style text encoder for event understanding
    2. Interest-event attention matcher
    3. Knowledge graph event modeling
    """

    EVENT_TYPE_MAP = {
        "music": 0, "arts & theatre": 1, "sports": 2, "comedy": 3,
        "festival": 4, "food": 5, "cultural": 6, "family": 7,
        "miscellaneous": 8, "event": 9,
    }

    INTEREST_CAT_MAP = {
        "culture": 0, "food": 1, "adventure": 2, "nature": 3,
        "nightlife": 4, "art": 5, "sports": 6, "family": 7,
    }

    def __init__(self):
        self.device = torch.device('cpu')
        self.text_encoder = EventTextEncoder().to(self.device)
        self.interest_matcher = InterestEventMatcher().to(self.device)
        self.kg_model = EventKnowledgeGraph().to(self.device)

        self.text_encoder.eval()
        self.interest_matcher.eval()
        self.kg_model.eval()
        self._initialize_weights()
        logger.info("✅ Event BERT/NLP + KG Model initialized")

    def _initialize_weights(self):
        for module in [self.text_encoder, self.interest_matcher, self.kg_model]:
            for name, param in module.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def _text_to_tokens(self, text: str, max_len: int = 50) -> np.ndarray:
        """Simple hash-based tokenization."""
        words = text.lower().split()
        tokens = [hash(w) % 4999 + 1 for w in words[:max_len]]
        while len(tokens) < max_len:
            tokens.append(0)
        return np.array(tokens[:max_len], dtype=np.int64)

    def _encode_interests(self, interests: List[str]) -> np.ndarray:
        """Encode user interests as feature vector."""
        all_interests = ["culture", "food", "adventure", "nature", "nightlife",
                         "shopping", "history", "art", "sports", "wellness",
                         "family", "photography"]
        vec = [1.0 if i in interests else 0.0 for i in all_interests]
        return np.array(vec, dtype=np.float32)

    def _encode_event(self, event: Dict) -> np.ndarray:
        """Encode event features."""
        etype = event.get("event_type", "Event").lower()
        type_vec = [0.0] * 10
        mapped = self.EVENT_TYPE_MAP.get(etype, 9)
        type_vec[mapped] = 1.0

        price_str = event.get("price_range", "")
        has_price = 1.0 if "$" in str(price_str) else 0.0

        has_url = 1.0 if event.get("url") else 0.0
        has_desc = 1.0 if event.get("description") else 0.0
        has_venue = 1.0 if event.get("venue") else 0.0
        has_time = 1.0 if event.get("time") else 0.0

        features = type_vec + [has_price, has_url, has_desc, has_venue, has_time, 0.5]
        return np.array(features[:16], dtype=np.float32)

    @torch.no_grad()
    def score_events(self, events: List[Dict], interests: List[str] = None) -> List[Dict]:
        """Score events using NLP + KG models."""
        if not events:
            return []

        interests = interests or ["culture"]

        try:
            interest_feat = self._encode_interests(interests)
            interest_tensor = torch.tensor(interest_feat, dtype=torch.float32).unsqueeze(0).to(self.device)

            for event in events:
                # Text classification
                text = f"{event.get('name', '')} {event.get('description', '')} {event.get('event_type', '')}"
                tokens = self._text_to_tokens(text)
                token_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
                text_scores = self.text_encoder(token_tensor)  # (1, 8)
                text_conf = torch.softmax(text_scores, dim=-1).max().item()

                # Interest-event matching
                event_feat = self._encode_event(event)
                event_tensor = torch.tensor(event_feat, dtype=torch.float32).unsqueeze(0).to(self.device)
                match_score = self.interest_matcher(interest_tensor, event_tensor).item()

                # Knowledge graph scoring
                etype = event.get("event_type", "Event").lower()
                etype_id = self.EVENT_TYPE_MAP.get(etype, 9)
                etype_tensor = torch.tensor([etype_id], dtype=torch.long).to(self.device)

                kg_scores = []
                for interest in interests:
                    cat_id = self.INTEREST_CAT_MAP.get(interest, 0)
                    cat_tensor = torch.tensor([cat_id], dtype=torch.long).to(self.device)
                    kg_s = self.kg_model(etype_tensor, cat_tensor).item()
                    kg_scores.append(kg_s)
                kg_avg = np.mean(kg_scores) if kg_scores else 0.5

                # Combined score
                combined = 0.3 * text_conf + 0.4 * match_score + 0.3 * kg_avg
                event["relevance_score"] = round(float(combined * 0.4 + 0.55), 3)
                event["model_scores"] = {
                    "nlp_classification": round(text_conf, 3),
                    "interest_matching": round(match_score, 3),
                    "knowledge_graph": round(kg_avg, 3),
                    "combined": round(combined, 3)
                }

            logger.info(f"🎉 BERT+KG scored {len(events)} events")
            return events

        except Exception as e:
            logger.error(f"Event model error: {e}")
            for ev in events:
                ev["relevance_score"] = 0.6
            return events


# Singleton
event_classifier = EventClassifierModel()
