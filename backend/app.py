"""
Travel Agentic AI System — FastAPI Main Application.
"""
import logging
import sys
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("TravelAI")

# Add backend dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings
from models.database import MongoDB, VectorDB
from agents.orchestrator import orchestrator
from services.openai_service import openai_service
from utils.nlp_processor import process_natural_language


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle events."""
    logger.info("🚀 Starting Travel Agentic AI System...")
    warnings = settings.validate()
    for w in warnings:
        logger.warning(f"⚠️ {w}")
    await MongoDB.connect()
    VectorDB.initialize()
    logger.info("✅ Travel Agentic AI System is ready!")
    yield
    await MongoDB.disconnect()
    logger.info("👋 Travel Agentic AI System shutdown complete")


app = FastAPI(
    title="Travel Agentic AI System",
    description="Intelligent multi-agent travel planning with real-time APIs and LLM reasoning",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static files
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


# ======================== SCHEMAS ========================

class TravelRequest(BaseModel):
    source: str
    destination: str
    departure_date: str
    return_date: str
    travelers: int = 1
    interests: List[str] = []
    budget_level: Optional[str] = "moderate"
    budget_amount: Optional[float] = None
    special_requirements: Optional[str] = None
    include_events: bool = True
    natural_language_input: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    travel_context: Optional[Dict[str, Any]] = None


# ======================== ENDPOINTS ========================

@app.get("/")
async def root():
    """Redirect root to frontend UI."""
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(str(frontend_path))
    return {
        "service": "Travel Agentic AI System",
        "version": "1.0.0",
        "status": "running",
        "agents": [
            "FlightAgent", "HotelAgent", "PlaceAgent", "WeatherAgent",
            "EventAgent", "BudgetAgent", "DistanceAgent", "ItineraryAgent"
        ],
        "apis": ["Amadeus", "Foursquare", "OpenWeather", "Ticketmaster", "OpenAI"],
    }


@app.get("/ui")
async def serve_ui():
    """Serve the frontend UI."""
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(str(frontend_path))
    return RedirectResponse(url="/docs")


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "database": "connected" if MongoDB.get_db() is not None else "not connected",
    }


@app.post("/api/plan")
async def plan_trip(request: TravelRequest):
    """Main endpoint: Generate a complete travel plan."""
    try:
        logger.info(f"📩 New travel request: {request.source} → {request.destination}")
        result = await orchestrator.plan_trip(request.model_dump())

        # Save to database (use a copy so MongoDB's _id doesn't mutate result)
        try:
            await MongoDB.save_travel_plan(dict(result))
        except Exception as e:
            logger.warning(f"DB save warning: {e}")

        # Remove MongoDB _id if present (not JSON serializable)
        result.pop("_id", None)
        return result
    except Exception as e:
        logger.error(f"Planning error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Conversational AI endpoint for travel assistance."""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        history = []
        try:
            msgs = await MongoDB.get_chat_history(session_id)
            history = [{"role": m["role"], "content": m["content"]} for m in msgs]
        except Exception:
            pass

        response = await openai_service.chat_response(
            request.message, history, request.travel_context
        )

        try:
            await MongoDB.save_chat_message(session_id, "user", request.message)
            await MongoDB.save_chat_message(session_id, "assistant", response)
        except Exception:
            pass

        return {
            "session_id": session_id,
            "response": response,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/nlp/extract")
async def extract_preferences(text: Dict[str, str]):
    """Extract travel preferences from natural language text."""
    try:
        raw = text.get("text", "")
        nlp_result = process_natural_language(raw)
        ai_result = await openai_service.extract_preferences(raw)
        return {"nlp_extraction": nlp_result, "ai_extraction": ai_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/plans/{request_id}")
async def get_plan(request_id: str):
    """Retrieve a previously generated travel plan."""
    plan = await MongoDB.get_travel_plan(request_id)
    if not plan:
        raise HTTPException(status_code=404, detail="Travel plan not found")
    plan["_id"] = str(plan["_id"])
    return plan


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
