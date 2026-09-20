"""
Database module for MongoDB and ChromaDB connections.
"""

import logging
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure
import chromadb
from chromadb.config import Settings as ChromaSettings
from config import settings

logger = logging.getLogger(__name__)


class MongoDB:
    """MongoDB async database handler."""

    _client: AsyncIOMotorClient = None
    _db = None

    @classmethod
    async def connect(cls):
        """Initialize MongoDB connection."""
        try:
            cls._client = AsyncIOMotorClient(settings.MONGODB_URI)
            cls._db = cls._client[settings.MONGODB_DB_NAME]
            # Test connection
            await cls._client.admin.command("ping")
            logger.info(f"✅ Connected to MongoDB: {settings.MONGODB_DB_NAME}")

            # Create indexes
            await cls._create_indexes()
        except ConnectionFailure as e:
            logger.warning(f"⚠️ MongoDB connection failed: {e}. Running without database.")
            cls._client = None
            cls._db = None
        except Exception as e:
            logger.warning(f"⚠️ MongoDB error: {e}. Running without database.")
            cls._client = None
            cls._db = None

    @classmethod
    async def _create_indexes(cls):
        """Create necessary database indexes."""
        if cls._db is None:
            return
        try:
            # User profiles
            await cls._db.user_profiles.create_index("user_id", unique=True)
            # Travel plans
            await cls._db.travel_plans.create_index("request_id", unique=True)
            await cls._db.travel_plans.create_index("created_at")
            # Chat sessions
            await cls._db.chat_sessions.create_index("session_id")
            logger.info("✅ MongoDB indexes created")
        except Exception as e:
            logger.warning(f"⚠️ Index creation warning: {e}")

    @classmethod
    async def disconnect(cls):
        """Close MongoDB connection."""
        if cls._client:
            cls._client.close()
            logger.info("MongoDB disconnected")

    @classmethod
    def get_db(cls):
        """Get database instance."""
        return cls._db

    @classmethod
    async def save_travel_plan(cls, plan_data: dict):
        """Save a generated travel plan."""
        if cls._db is None:
            logger.warning("MongoDB not connected. Skipping save.")
            return None
        try:
            result = await cls._db.travel_plans.insert_one(plan_data)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error saving travel plan: {e}")
            return None

    @classmethod
    async def get_travel_plan(cls, request_id: str):
        """Retrieve a travel plan by request ID."""
        if cls._db is None:
            return None
        try:
            return await cls._db.travel_plans.find_one({"request_id": request_id})
        except Exception as e:
            logger.error(f"Error retrieving travel plan: {e}")
            return None

    @classmethod
    async def save_user_profile(cls, user_data: dict):
        """Save or update user profile."""
        if cls._db is None:
            return None
        try:
            result = await cls._db.user_profiles.update_one(
                {"user_id": user_data["user_id"]},
                {"$set": user_data},
                upsert=True
            )
            return result.upserted_id or user_data["user_id"]
        except Exception as e:
            logger.error(f"Error saving user profile: {e}")
            return None

    @classmethod
    async def save_chat_message(cls, session_id: str, role: str, content: str):
        """Save a chat message."""
        if cls._db is None:
            return None
        try:
            from datetime import datetime
            result = await cls._db.chat_sessions.insert_one({
                "session_id": session_id,
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow()
            })
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error saving chat message: {e}")
            return None

    @classmethod
    async def get_chat_history(cls, session_id: str, limit: int = 20):
        """Get chat history for a session."""
        if cls._db is None:
            return []
        try:
            cursor = cls._db.chat_sessions.find(
                {"session_id": session_id}
            ).sort("timestamp", 1).limit(limit)
            return await cursor.to_list(length=limit)
        except Exception as e:
            logger.error(f"Error retrieving chat history: {e}")
            return []


class VectorDB:
    """ChromaDB vector database handler for embeddings and similarity search."""

    _client = None
    _collections = {}

    @classmethod
    def initialize(cls):
        """Initialize ChromaDB client."""
        try:
            cls._client = chromadb.Client(ChromaSettings(
                anonymized_telemetry=False,
            ))

            # Create collections
            cls._collections["user_preferences"] = cls._client.get_or_create_collection(
                name="user_preferences",
                metadata={"description": "User travel preference embeddings"}
            )
            cls._collections["destinations"] = cls._client.get_or_create_collection(
                name="destinations",
                metadata={"description": "Destination information embeddings"}
            )
            cls._collections["travel_context"] = cls._client.get_or_create_collection(
                name="travel_context",
                metadata={"description": "Travel context and memory"}
            )
            logger.info("✅ ChromaDB initialized with collections")
        except Exception as e:
            logger.warning(f"⚠️ ChromaDB initialization warning: {e}")
            cls._client = None

    @classmethod
    def store_embedding(cls, collection_name: str, doc_id: str,
                        document: str, metadata: dict = None):
        """Store a document with auto-generated embedding."""
        if cls._client is None or collection_name not in cls._collections:
            return
        try:
            cls._collections[collection_name].upsert(
                ids=[doc_id],
                documents=[document],
                metadatas=[metadata or {}]
            )
        except Exception as e:
            logger.error(f"Error storing embedding: {e}")

    @classmethod
    def search_similar(cls, collection_name: str, query: str,
                       n_results: int = 5, where: dict = None):
        """Search for similar documents."""
        if cls._client is None or collection_name not in cls._collections:
            return []
        try:
            params = {
                "query_texts": [query],
                "n_results": n_results
            }
            if where:
                params["where"] = where
            results = cls._collections[collection_name].query(**params)
            return results
        except Exception as e:
            logger.error(f"Error searching embeddings: {e}")
            return []
