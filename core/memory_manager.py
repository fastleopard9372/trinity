import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import asyncio
import httpx
import sqlite3
from pathlib import Path

from pinecone import Pinecone

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document

from clients.nas_client import NASClient
from clients.gdrive_client import GoogleDriveClient
from models.memory import DatabaseManager, MemoryRecord
from utils.logger import logger
from config.settings import settings

@dataclass
class ConversationEntry:
    """Individual conversation message or entry"""
    id: str
    session_id: str
    user_id: str
    content: str
    role: str  # 'user', 'assistant', 'system'
    timestamp: datetime
    metadata: Dict[str, Any]
    
@dataclass
class MemoryContext:
    """Context retrieved from both vector and NAS memory"""
    vector_results: List[Dict]
    nas_entries: List[MemoryRecord]
    relevance_score: float
    search_query: str

class VectorMemoryManager:
    """Manages fast vector-based memory using Pinecone"""
    
    def __init__(self):
        self.index_name = settings.pinecone_index_name
        self.embeddings = OpenAIEmbeddings(
            openai_api_key= settings.openai_api_key,
            model="text-embedding-3-small"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        # Initialize Pinecone
        self.pinecone = Pinecone(api_key=settings.pinecone_api_key)

        # Create or connect to index
        if self.index_name not in self.pinecone.list_indexes().names():
            self.pinecone.create_index_for_model(
                name=self.index_name,
                cloud="aws",
                region=settings.pinecone_environment,
                embed={
                    "model": "text-embedding-3-small",
                    "field_map": {"text": "chunk_text"}
                }
            )

        self.index = self.pinecone.Index(self.index_name)

        self.vectorstore = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings, 
            text_key="text"
        )
        
    async def store_conversation_entry(self, entry: ConversationEntry) -> bool:
        """Store a conversation entry in vector database"""
        try:
            # Create document
            doc = Document(
                page_content=entry.content,
                metadata={
                    "entry_id": entry.id,
                    "session_id": entry.session_id,
                    "user_id": entry.user_id,
                    "role": entry.role,
                    "timestamp": entry.timestamp.isoformat(),
                    "type": "conversation",
                    **entry.metadata
                }
            )
            # docs = self.text_splitter.split_documents(doc)
            await self.vectorstore.aadd_documents([doc], ids=[entry.id])
            
            logger.info(f"Stored conversation entry in vector DB: {entry.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store vector entry: {e}")
            return False
    
    async def semantic_search(self, query: str, k: int = 5, 
                            filters: Dict = None) -> List[Dict]:
        """Perform semantic search on vector memory"""
        try:
            # Build filter if provided
            search_kwargs = {"k": k}
            if filters:
                search_kwargs["filter"] = filters
            
            results = await asyncio.to_thread(
                self.vectorstore.similarity_search_with_score,
                query,
                **search_kwargs
            )
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": float(score),
                    "source": "vector"
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def cleanup_old_entries(self, days_old: int = 30):
        """Remove entries older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            # Query old entries (this is a simplified approach)
            # In production, you might want to implement batch deletion
            logger.info(f"Cleaning up vector entries older than {days_old} days")
            
        except Exception as e:
            logger.error(f"Vector cleanup failed: {e}")

class SQLiteSessionManager:
    """Manages conversation sessions using SQLite with Prisma-like interface"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                message_count INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active'
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_entries (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                role TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                vector_stored BOOLEAN DEFAULT FALSE,
                nas_stored BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        """)
        
        # self.conn.execute("""
        #     CREATE INDEX IF NOT EXISTS idx_session_user ON sessions(user_id);
        #     CREATE INDEX IF NOT EXISTS idx_entry_session ON conversation_entries(session_id);
        #     CREATE INDEX IF NOT EXISTS idx_entry_timestamp ON conversation_entries(timestamp);
        # """)
    
    def create_session(self, user_id: str, metadata: Dict = None) -> str:
        """Create a new conversation session"""
        session_id = f"session_{hashlib.md5(f'{user_id}_{datetime.now().isoformat()}'.encode()).hexdigest()}"
        
        self.conn.execute(
            "INSERT INTO sessions (id, user_id, metadata) VALUES (?, ?, ?)",
            (session_id, user_id, json.dumps(metadata or {}))
        )
        logger.info(f"Created session: {session_id}")
        return session_id
    
    def add_conversation_entry(self, entry: ConversationEntry) -> bool:
        """Add conversation entry to SQLite"""
        try:
            self.conn.execute("""
                INSERT INTO conversation_entries 
                (id, session_id, user_id, content, role, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.id,
                entry.session_id,
                entry.user_id,
                entry.content,
                entry.role,
                entry.timestamp.isoformat(),
                json.dumps(entry.metadata)
            ))
            
            # Update session message count
            self.conn.execute("""
                UPDATE sessions 
                SET message_count = message_count + 1, 
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (entry.session_id,))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add conversation entry: {e}")
            return False
    
    def get_session_entries(self, session_id: str, limit: int = 100) -> List[ConversationEntry]:
        """Get conversation entries for a session"""
        entries = []

        self.conn.row_factory = sqlite3.Row
        cursor = self.conn.execute("""
            SELECT * FROM conversation_entries 
            WHERE session_id = ? 
            ORDER BY timestamp ASC 
            LIMIT ?
        """, (session_id, limit))
        
        for row in cursor:
            entries.append(ConversationEntry(
                id=row['id'],
                session_id=row['session_id'],
                user_id=row['user_id'],
                content=row['content'],
                role=row['role'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                metadata=json.loads(row['metadata'] or '{}')
            ))
        
        return entries
    
    def get_sessions_for_nas_storage(self, min_messages: int = 10) -> List[str]:
        """Get sessions that should be stored to NAS"""
        session_ids = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id FROM sessions 
                WHERE message_count >= ? 
                AND status = 'active'
                ORDER BY updated_at ASC
            """, (min_messages,))
            
            session_ids = [row[0] for row in cursor]
        
        return session_ids

class MemoryManager:
    """Main dual memory manager combining vector and NAS storage"""
    
    def __init__(self):
        # Initialize components
        self.vector_manager = VectorMemoryManager()
        self.session_manager = SQLiteSessionManager(
            os.path.join(settings.memory_db_path.replace('.db', '_sessions.db'))
        )
        self.nas_client = NASClient()
        self.gdrive_client = GoogleDriveClient()
        self.db_manager = DatabaseManager(settings.memory_db_path)
        
        # LangChain for summarization
        self.llm = ChatOpenAI(openai_api_key=settings.openai_api_key, temperature=0.3)
        self.summarize_chain = load_summarize_chain(self.llm, chain_type="map_reduce")
        
        # Auto-save triggers
        self.auto_save_config = {
            "min_messages": 10,
            "max_session_age_hours": 24,
            "important_keywords": ["save this", "remember this", "log this", "important"]
        }
    
    async def add_conversation_message(self, session_id: str, user_id: str, 
                                     content: str, role: str, 
                                     metadata: Dict = None) -> str:
        """Add a message to both vector and session storage"""
        # Create entry
        entry_id = f"entry_{hashlib.md5(f'{session_id}_{datetime.now().isoformat()}'.encode()).hexdigest()}"
        entry = ConversationEntry(
            id=entry_id,
            session_id=session_id,
            user_id=user_id,
            content=content,
            role=role,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        # Store in SQLite
        self.session_manager.add_conversation_entry(entry)
        
        # Store in vector database
        await self.vector_manager.store_conversation_entry(entry)
        
        # Check for auto-save triggers
        await self.check_auto_save_triggers(session_id, content)
        
        logger.info(f"Added conversation message: {entry_id}")
        return entry_id
    
    async def search_memory(self, query: str, user_id: str = None, 
                          include_nas: bool = True) -> MemoryContext:
        """Search across both vector and NAS memory"""
        # Vector search
        vector_filters = {"user_id": user_id} if user_id else None
        vector_results = await self.vector_manager.semantic_search(
            query, k=5, filters=vector_filters
        )
        
        # NAS search (through existing database)
        nas_entries = []
        if include_nas:
            nas_entries = self.db_manager.search_memories(query, limit=5)
        
        # Calculate relevance
        relevance_score = self._calculate_relevance(vector_results, nas_entries, query)
        
        return MemoryContext(
            vector_results=vector_results,
            nas_entries=nas_entries,
            relevance_score=relevance_score,
            search_query=query
        )
    
    async def check_auto_save_triggers(self, session_id: str, content: str):
        """Check if conversation should be saved to NAS"""
        # Check for important keywords
        for keyword in self.auto_save_config["important_keywords"]:
            if keyword.lower() in content.lower():
                await self.save_session_to_nas(session_id, reason=f"keyword: {keyword}")
                return
        
        # Check message count
        with sqlite3.connect(self.session_manager.db_path) as conn:
            cursor = conn.execute(
                "SELECT message_count FROM sessions WHERE id = ?", 
                (session_id,)
            )
            result = cursor.fetchone()
            
            if result and result[0] >= self.auto_save_config["min_messages"]:
                await self.save_session_to_nas(session_id, reason="message_count")
    
    async def save_session_to_nas(self, session_id: str, reason: str = "manual"):
        """Save complete session to NAS storage"""
        try:
            # Get session entries
            entries = self.session_manager.get_session_entries(session_id)
            
            if not entries:
                logger.warning(f"No entries found for session: {session_id}")
                return False
            
            # Create summary
            conversation_text = "\n".join([
                f"{entry.role}: {entry.content}" for entry in entries
            ])
            
            docs = [Document(page_content=conversation_text)]
            summary = await asyncio.to_thread(
                self.summarize_chain.run, docs
            )
            # Prepare session data
            session_data = {
                "session_id": session_id,
                "user_id": entries[0].user_id,
                "created_at": entries[0].timestamp.isoformat(),
                "updated_at": entries[-1].timestamp.isoformat(),
                "message_count": len(entries),
                "save_reason": reason,
                "summary": summary,
                "entries": [
                    {
                        "id": entry.id,
                        "role": entry.role,
                        "content": entry.content,
                        "timestamp": entry.timestamp.isoformat(),
                        "metadata": entry.metadata
                    }
                    for entry in entries
                ]
            }
            # Save to NAS
            filename = f"session_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            nas_path = await self._save_session_data_to_nas(session_data, filename)
            logger.log("-----------------------save----------------------")
            # Save to database for indexing
            memory_data = {
                "content": summary,
                "title": f"Session Summary - {session_id[:8]}",
                "tags": ["conversation", "session", reason],
                "category": "sessions",
                "nas_path": nas_path,
                "metadata": {
                    "session_id": session_id,
                    "message_count": len(entries),
                    "save_reason": reason,
                    "full_data_path": nas_path
                },
                "source": "dual_memory"
            }
            
            self.db_manager.add_memory(memory_data)
            
            # Mark session as archived
            with sqlite3.connect(self.session_manager.db_path) as conn:
                conn.execute(
                    "UPDATE sessions SET status = 'archived' WHERE id = ?",
                    (session_id,)
                )
            
            logger.info(f"Session {session_id} saved to NAS: {nas_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save session to NAS: {e}")
            return False
    
    async def _save_session_data_to_nas(self, session_data: Dict, filename: str) -> str:
        """Save session data to NAS and Google Drive"""
        # Save to temp file first
        temp_file = f"temp/{filename}"
        os.makedirs("temp", exist_ok=True)
        
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        # Upload to NAS
        nas_path = self.nas_client.save_memory_content(
            json.dumps(session_data, indent=2),
            "sessions",
            filename
        )
        
        # Upload to Google Drive
        gdrive_file_id = self.gdrive_client.upload_file(
            temp_file,
            filename,
            self.gdrive_client.folder_id
        )
        
        # Cleanup
        os.remove(temp_file)
        
        return nas_path
    
    def _calculate_relevance(self, vector_results: List[Dict], 
                           nas_entries: List[MemoryRecord], 
                           query: str) -> float:
        """Calculate overall relevance score"""
        if not vector_results and not nas_entries:
            return 0.0
        
        # Weight vector results higher for recency
        vector_score = sum(r.get("relevance_score", 0) for r in vector_results) * 0.7
        nas_score = len(nas_entries) * 0.3  # Simple count-based scoring
        
        return min((vector_score + nas_score) / max(len(vector_results) + len(nas_entries), 1), 1.0)
    
    async def cleanup_old_memories(self, vector_days: int = 30, session_days: int = 90):
        """Cleanup old entries from vector DB and sessions"""
        await self.vector_manager.cleanup_old_entries(vector_days)
        
        # Archive old sessions
        cutoff_date = datetime.now() - timedelta(days=session_days)
        old_sessions = self.session_manager.get_sessions_for_nas_storage(min_messages=1)
        
        for session_id in old_sessions:
            # Check if already archived
            with sqlite3.connect(self.session_manager.db_path) as conn:
                cursor = conn.execute(
                    "SELECT status, updated_at FROM sessions WHERE id = ?",
                    (session_id,)
                )
                result = cursor.fetchone()
                
                if result and result[0] == 'active':
                    updated_at = datetime.fromisoformat(result[1])
                    if updated_at < cutoff_date:
                        await self.save_session_to_nas(session_id, reason="auto_archive")

#Settings for the new system
class Settings:
    """Extended settings for dual memory system"""
    
    # Existing settings...
    # Vector database settings
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "trinity-memory")
    
    # OpenAI settings
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    
    # Memory management
    vector_cleanup_days: int = int(os.getenv("VECTOR_CLEANUP_DAYS", "30"))
    session_archive_days: int = int(os.getenv("SESSION_ARCHIVE_DAYS", "90"))
    auto_save_threshold: int = int(os.getenv("AUTO_SAVE_THRESHOLD", "10"))

# Usage example
async def main():
    """Example usage of the dual memory system"""
    memory_manager = MemoryManager()
    
    # Create a session
    session_id = memory_manager.session_manager.create_session("user123")
    
    # Add some conversation messages
    await memory_manager.add_conversation_message(
        session_id, "user123", "Hello, I want to learn about Python", "user"
    )
    
    await memory_manager.add_conversation_message(
        session_id, "user123", "Python is a great programming language!", "assistant"
    )
    
    # Search memory
    context = await memory_manager.search_memory("Python programming")
    print(f"Found {len(context.vector_results)} vector results")
    print(f"Found {len(context.nas_entries)} NAS entries")
    
    # Trigger save
    await memory_manager.add_conversation_message(
        session_id, "user123", "Save this conversation for future reference", "user"
    )

if __name__ == "__main__":
    asyncio.run(main())