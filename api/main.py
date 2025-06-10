

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import asyncio
from datetime import datetime
import json

# imports
from core.memory_manager import MemoryManager, MemoryContext, ConversationEntry
from clients.nas_client import NASClient
from clients.gdrive_client import GoogleDriveClient
from core.command_parser import CommandParser
from models.memory import DatabaseManager
from utils.logger import logger
from config.settings import settings

# Pydantic models for API
class ConversationMessage(BaseModel):
    session_id: Optional[str] = None
    user_id: str
    content: str
    role: str = "user"
    metadata: Optional[Dict[str, Any]] = {}

class ConversationResponse(BaseModel):
    entry_id: str
    session_id: str
    timestamp: str
    stored_in_vector: bool
    auto_saved_to_nas: bool

class MemorySearchRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    include_nas: bool = True
    max_results: int = 10

class MemorySearchResponse(BaseModel):
    vector_results: List[Dict]
    nas_results: List[Dict]
    relevance_score: float
    total_results: int
    search_query: str

class SessionSummaryRequest(BaseModel):
    session_id: str
    force_save: bool = False

class SessionSummaryResponse(BaseModel):
    session_id: str
    summary: str
    message_count: int
    saved_to_nas: bool
    nas_path: Optional[str]

class SystemHealthResponse(BaseModel):
    nas_connected: bool
    gdrive_connected: bool
    vector_db_connected: bool
    database_accessible: bool
    pinecone_index_stats: Dict
    active_sessions: int
    total_memories: int

app = FastAPI(title="Trinity Memory System API", version="2.0.0")

# Global memory manager
memory_manager: MemoryManager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize system on startup"""
    global memory_manager
    
    logger.info("Starting Trinity Memory System API")
    
    # Create necessary directories
    directories = ["data", "logs", "temp", "credentials", "data/sessions"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Initialize memory manager
    try:
        memory_manager = MemoryManager()
        logger.info("memory manager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize memory manager: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown
    if memory_manager:
        logger.info("Performing cleanup...")
        await memory_manager.cleanup_old_memories()

app.router.lifespan_context = lifespan

@app.get("/")
async def root():
    return {
        "message": "Trinity Memory System API", 
        "version": "2.0.0",
        "features": ["vector_memory", "nas_storage", "conversation_tracking", "auto_summarization"]
    }

@app.post("/api/conversation/message", response_model=ConversationResponse)
async def add_conversation_message(message: ConversationMessage):
    """Add a message to conversation memory"""
    try:
        # Create session if not provided
        session_id = message.session_id
        if not session_id:
            session_id = memory_manager.session_manager.create_session(
                message.user_id, 
                {"created_via": "api"}
            )
        
        # Add message to memory
        entry_id = await memory_manager.add_conversation_message(
            session_id=session_id,
            user_id=message.user_id,
            content=message.content,
            role=message.role,
            metadata=message.metadata
        )
        
        # Check if auto-saved (simplified check)
        auto_saved = any(keyword in message.content.lower() 
                        for keyword in memory_manager.auto_save_config["important_keywords"])
        
        return ConversationResponse(
            entry_id=entry_id,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            stored_in_vector=True,
            auto_saved_to_nas=auto_saved
        )
        
    except Exception as e:
        logger.error(f"Failed to add conversation message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add message: {str(e)}")

@app.post("/api/memory/search", response_model=MemorySearchResponse)
async def search_memory(search_request: MemorySearchRequest):
    """Search across vector and NAS memory"""
    try:
        context = await memory_manager.search_memory(
            query=search_request.query,
            user_id=search_request.user_id,
            include_nas=search_request.include_nas
        )
        
        # Format NAS results
        nas_results = [
            {
                "content": memory.content,
                "title": memory.title,
                "category": memory.category,
                "timestamp": memory.timestamp.isoformat() if memory.timestamp else None,
                "tags": json.loads(memory.tags) if memory.tags else [],
                "source": "nas"
            }
            for memory in context.nas_entries[:search_request.max_results]
        ]
        
        return MemorySearchResponse(
            vector_results=context.vector_results[:search_request.max_results],
            nas_results=nas_results,
            relevance_score=context.relevance_score,
            total_results=len(context.vector_results) + len(context.nas_entries),
            search_query=context.search_query
        )
        
    except Exception as e:
        logger.error(f"Memory search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/api/session/summarize", response_model=SessionSummaryResponse)
async def summarize_session(request: SessionSummaryRequest):
    """Generate summary and optionally save session to NAS"""
    try:
        # Get session entries
        entries = memory_manager.session_manager.get_session_entries(request.session_id)
        
        if not entries:
            raise HTTPException(status_code=404, detail="Session not found")
        
        saved_to_nas = False
        nas_path = None
        
        if request.force_save or len(entries) >= memory_manager.auto_save_config["min_messages"]:
            success = await memory_manager.save_session_to_nas(
                request.session_id, 
                reason="manual" if request.force_save else "threshold"
            )
            
            if success:
                saved_to_nas = True
                # Get the NAS path from the most recent memory record
                recent_memories = memory_manager.db_manager.search_memories(request.session_id)
                if recent_memories:
                    nas_path = recent_memories[0].nas_path
        
        # Generate quick summary
        conversation_text = "\n".join([
            f"{entry.role}: {entry.content}" for entry in entries[-10:]  # Last 10 messages
        ])
        
        summary = f"Conversation with {len(entries)} messages. Recent topics: {conversation_text[:200]}..."
        
        return SessionSummaryResponse(
            session_id=request.session_id,
            summary=summary,
            message_count=len(entries),
            saved_to_nas=saved_to_nas,
            nas_path=nas_path
        )
        
    except Exception as e:
        logger.error(f"Session summarization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

@app.get("/api/session/{session_id}/messages")
async def get_session_messages(session_id: str, limit: int = 100):
    """Get conversation messages for a session"""
    try:
        entries = memory_manager.session_manager.get_session_entries(session_id, limit)
        
        messages = [
            {
                "id": entry.id,
                "role": entry.role,
                "content": entry.content,
                "timestamp": entry.timestamp.isoformat(),
                "metadata": entry.metadata
            }
            for entry in entries
        ]
        
        return {
            "session_id": session_id,
            "messages": messages,
            "total_messages": len(messages),
            "limited_to": limit
        }
        
    except Exception as e:
        logger.error(f"Failed to get session messages: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get messages: {str(e)}")

@app.get("/api/sessions/user/{user_id}")
async def get_user_sessions(user_id: str, limit: int = 20):
    """Get sessions for a specific user"""
    try:
        import sqlite3
        sessions = []
        
        with sqlite3.connect(memory_manager.session_manager.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT id, created_at, updated_at, message_count, status, metadata
                FROM sessions 
                WHERE user_id = ? 
                ORDER BY updated_at DESC 
                LIMIT ?
            """, (user_id, limit))
            
            for row in cursor:
                sessions.append({
                    "session_id": row['id'],
                    "created_at": row['created_at'],
                    "updated_at": row['updated_at'],
                    "message_count": row['message_count'],
                    "status": row['status'],
                    "metadata": json.loads(row['metadata'] or '{}')
                })
        
        return {
            "user_id": user_id,
            "sessions": sessions,
            "total_sessions": len(sessions)
        }
        
    except Exception as e:
        logger.error(f"Failed to get user sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get sessions: {str(e)}")

@app.get("/api/health", response_model=SystemHealthResponse)
async def get_system_health():
    """Comprehensive system health check"""
    try:
        # Check NAS connection
        nas_connected = memory_manager.nas_client.fs is not None
        
        # Check Google Drive connection
        gdrive_connected = memory_manager.gdrive_client.service is not None
        
        # Check vector database
        vector_db_connected = False
        pinecone_stats = {}
        try:
            index_stats = memory_manager.vector_manager.index.describe_index_stats()
            vector_db_connected = True
            pinecone_stats = {
                "total_vector_count": index_stats.get("total_vector_count", 0),
                "dimension": index_stats.get("dimension", 0),
                "index_fullness": index_stats.get("index_fullness", 0.0)
            }
        except Exception as e:
            logger.error(f"Vector DB health check failed: {e}")
        
        # Check database
        try:
            total_memories = len(memory_manager.db_manager.get_memories(limit=1))
            db_accessible = True
        except:
            total_memories = 0
            db_accessible = False
        
        # Check active sessions
        active_sessions = 0
        try:
            import sqlite3
            with sqlite3.connect(memory_manager.session_manager.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM sessions WHERE status = 'active'")
                active_sessions = cursor.fetchone()[0]
        except:
            pass
        
        return SystemHealthResponse(
            nas_connected=nas_connected,
            gdrive_connected=gdrive_connected,
            vector_db_connected=vector_db_connected,
            database_accessible=db_accessible,
            pinecone_index_stats=pinecone_stats,
            active_sessions=active_sessions,
            total_memories=total_memories
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/api/system/cleanup")
async def cleanup_system(background_tasks: BackgroundTasks, 
                        vector_days: int = 30, 
                        session_days: int = 90):
    """Trigger system cleanup"""
    try:
        background_tasks.add_task(
            memory_manager.cleanup_old_memories,
            vector_days,
            session_days
        )
        
        return {
            "message": "Cleanup task started",
            "vector_cleanup_days": vector_days,
            "session_cleanup_days": session_days
        }
        
    except Exception as e:
        logger.error(f"Cleanup task failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@app.post("/api/system/backup-sessions")
async def backup_active_sessions(background_tasks: BackgroundTasks, 
                                min_messages: int = 5):
    """Backup active sessions to NAS"""
    try:
        session_ids = memory_manager.session_manager.get_sessions_for_nas_storage(min_messages)
        
        async def backup_sessions():
            for session_id in session_ids:
                await memory_manager.save_session_to_nas(session_id, reason="manual_backup")
        
        background_tasks.add_task(backup_sessions)
        
        return {
            "message": f"Backup task started for {len(session_ids)} sessions",
            "sessions_to_backup": len(session_ids),
            "min_messages_threshold": min_messages
        }
        
    except Exception as e:
        logger.error(f"Backup task failed: {e}")
        raise HTTPException(status_code=500, detail=f"Backup failed: {str(e)}")

# Legacy endpoint compatibility
@app.post("/api/memory/save")
async def save_memory_legacy(memory_input: dict):
    """Legacy endpoint for backward compatibility"""
    try:
        # Convert to conversation message format
        message = ConversationMessage(
            user_id=memory_input.get("user_id", "legacy_user"),
            content=memory_input.get("content", ""),
            role="user",
            metadata={
                "tags": memory_input.get("tags", []),
                "category": memory_input.get("category", "general"),
                "priority": memory_input.get("priority", 1),
                "legacy_import": True
            }
        )
        
        # Use the new conversation endpoint
        response = await add_conversation_message(message)
        
        # Format response in legacy format
        return {
            "id": response.entry_id,
            "session_id": response.session_id,
            "content": memory_input.get("content", ""),
            "timestamp": response.timestamp,
            "stored": True
        }
        
    except Exception as e:
        logger.error(f"Legacy save failed: {e}")
        raise HTTPException(status_code=500, detail=f"Legacy save failed: {str(e)}")

@app.get("/api/analytics/memory-stats")
async def get_memory_analytics():
    """Get analytics about memory usage"""
    try:
        import sqlite3
        from collections import defaultdict
        
        stats = {
            "vector_db": {},
            "sessions": {},
            "nas_storage": {},
            "daily_activity": {}
        }
        
        # Vector DB stats
        try:
            index_stats = memory_manager.vector_manager.index.describe_index_stats()
            stats["vector_db"] = {
                "total_vectors": index_stats.get("total_vector_count", 0),
                "dimension": index_stats.get("dimension", 0),
                "index_fullness": index_stats.get("index_fullness", 0.0)
            }
        except:
            pass
        
        # Session stats
        with sqlite3.connect(memory_manager.session_manager.db_path) as conn:
            # Active vs archived sessions
            cursor = conn.execute("""
                SELECT status, COUNT(*) as count 
                FROM sessions 
                GROUP BY status
            """)
            session_status = dict(cursor.fetchall())
            
            # Messages per session distribution
            cursor = conn.execute("""
                SELECT 
                    CASE 
                        WHEN message_count < 5 THEN 'short'
                        WHEN message_count < 20 THEN 'medium'
                        ELSE 'long'
                    END as session_type,
                    COUNT(*) as count
                FROM sessions 
                GROUP BY session_type
            """)
            session_lengths = dict(cursor.fetchall())
            
            stats["sessions"] = {
                "by_status": session_status,
                "by_length": session_lengths
            }
        
        # NAS storage stats
        nas_memories = memory_manager.db_manager.get_memories(limit=1000)
        category_counts = defaultdict(int)
        for memory in nas_memories:
            category_counts[memory.category or "uncategorized"] += 1
        
        stats["nas_storage"] = {
            "total_memories": len(nas_memories),
            "by_category": dict(category_counts)
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Analytics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")

@app.websocket("/ws/memory-updates")
async def websocket_memory_updates(websocket):
    """WebSocket endpoint for real-time memory updates"""
    await websocket.accept()
    try:
        while True:
            # This is a placeholder for real-time updates
            # In a full implementation, you'd have a proper pub/sub system
            await asyncio.sleep(30)
            await websocket.send_json({
                "type": "health_check",
                "timestamp": datetime.now().isoformat(),
                "status": "active"
            })
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )