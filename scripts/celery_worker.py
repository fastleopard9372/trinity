"""
Celery worker for background tasks in Trinity Memory System
"""

import os
import sys
import asyncio
from pathlib import Path
from celery import Celery
from datetime import timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.memory_manager import DualMemoryManager
from config.settings import settings
from utils.logger import logger

# Initialize Celery
celery_app = Celery(
    'trinity_tasks',
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

# Periodic tasks schedule
celery_app.conf.beat_schedule = {
    'cleanup-old-memories': {
        'task': 'scripts.celery_worker.cleanup_old_memories',
        'schedule': timedelta(hours=settings.cleanup_interval_hours),
    },
    'backup-active-sessions': {
        'task': 'scripts.celery_worker.backup_active_sessions',
        'schedule': timedelta(hours=12),  # Every 12 hours
    },
    'system-health-check': {
        'task': 'scripts.celery_worker.system_health_check',
        'schedule': timedelta(minutes=30),  # Every 30 minutes
    },
    'summarize-long-sessions': {
        'task': 'scripts.celery_worker.summarize_long_sessions',
        'schedule': timedelta(hours=6),  # Every 6 hours
    },
}

# Global memory manager instance
memory_manager = None

def get_memory_manager():
    """Get or create memory manager instance"""
    global memory_manager
    if memory_manager is None:
        memory_manager = DualMemoryManager()
    return memory_manager

@celery_app.task(bind=True)
def cleanup_old_memories(self, vector_days=None, session_days=None):
    """Background task to cleanup old memories"""
    try:
        logger.info("Starting cleanup task...")
        
        vector_days = vector_days or settings.vector_cleanup_days
        session_days = session_days or settings.session_archive_days
        
        # Run async cleanup
        async def run_cleanup():
            manager = get_memory_manager()
            await manager.cleanup_old_memories(vector_days, session_days)
        
        asyncio.run(run_cleanup())
        
        logger.info(f"Cleanup completed - vector: {vector_days} days, sessions: {session_days} days")
        return {"status": "success", "vector_days": vector_days, "session_days": session_days}
        
    except Exception as e:
        logger.error(f"Cleanup task failed: {e}")
        raise self.retry(exc=e, countdown=60, max_retries=3)

@celery_app.task(bind=True)
def backup_active_sessions(self, min_messages=None):
    """Background task to backup active sessions to NAS"""
    try:
        logger.info("Starting session backup task...")
        
        min_messages = min_messages or settings.auto_save_threshold
        manager = get_memory_manager()
        
        # Get sessions that need backing up
        session_ids = manager.session_manager.get_sessions_for_nas_storage(min_messages)
        
        backed_up = 0
        failed = 0
        
        async def backup_sessions():
            nonlocal backed_up, failed
            for session_id in session_ids:
                try:
                    success = await manager.save_session_to_nas(session_id, reason="auto_backup")
                    if success:
                        backed_up += 1
                    else:
                        failed += 1
                except Exception as e:
                    logger.error(f"Failed to backup session {session_id}: {e}")
                    failed += 1
        
        asyncio.run(backup_sessions())
        
        logger.info(f"Session backup completed - backed up: {backed_up}, failed: {failed}")
        return {
            "status": "success", 
            "backed_up": backed_up, 
            "failed": failed,
            "total_sessions": len(session_ids)
        }
        
    except Exception as e:
        logger.error(f"Session backup task failed: {e}")
        raise self.retry(exc=e, countdown=60, max_retries=3)

@celery_app.task(bind=True)
def system_health_check(self):
    """Background task for system health monitoring"""
    try:
        manager = get_memory_manager()
        
        # Check component health
        health_status = {
            "nas_connected": manager.nas_client.fs is not None,
            "gdrive_connected": manager.gdrive_client.service is not None,
            "vector_db_connected": False,
            "database_accessible": False,
            "timestamp": asyncio.run(self._get_current_time())
        }
        
        # Check vector DB
        try:
            index_stats = manager.vector_manager.index.describe_index_stats()
            health_status["vector_db_connected"] = True
            health_status["vector_count"] = index_stats.get("total_vector_count", 0)
        except:
            pass
        
        # Check database
        try:
            memories = manager.db_manager.get_memories(limit=1)
            health_status["database_accessible"] = True
            health_status["total_memories"] = len(memories)
        except:
            pass
        
        # Log warnings for failed components
        failed_components = []
        if not health_status["nas_connected"]:
            failed_components.append("NAS")
        if not health_status["gdrive_connected"]:
            failed_components.append("Google Drive")
        if not health_status["vector_db_connected"]:
            failed_components.append("Vector DB")
        if not health_status["database_accessible"]:
            failed_components.append("Database")
        
        if failed_components:
            logger.warning(f"Health check - Failed components: {', '.join(failed_components)}")
        else:
            logger.info("Health check - All systems operational")
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check task failed: {e}")
        raise self.retry(exc=e, countdown=60, max_retries=3)
    
    async def _get_current_time(self):
        from datetime import datetime
        return datetime.now().isoformat()

@celery_app.task(bind=True)
def summarize_long_sessions(self, min_messages=None):
    """Background task to auto-summarize long active sessions"""
    try:
        logger.info("Starting session summarization task...")
        
        min_messages = min_messages or (settings.auto_save_threshold * 2)  # Double the threshold
        manager = get_memory_manager()
        
        # Get long sessions that haven't been summarized
        import sqlite3
        session_ids = []
        
        with sqlite3.connect(manager.session_manager.db_path) as conn:
            cursor = conn.execute("""
                SELECT id FROM sessions 
                WHERE message_count >= ? 
                AND status = 'active'
                AND id NOT IN (
                    SELECT DISTINCT session_id FROM conversation_entries 
                    WHERE metadata LIKE '%auto_summarized%'
                )
                ORDER BY updated_at ASC
                LIMIT 10
            """, (min_messages,))
            
            session_ids = [row[0] for row in cursor]
        
        summarized = 0
        failed = 0
        
        async def summarize_sessions():
            nonlocal summarized, failed
            for session_id in session_ids:
                try:
                    success = await manager.save_session_to_nas(session_id, reason="auto_summarize")
                    if success:
                        summarized += 1
                        
                        # Mark as summarized
                        await manager.add_conversation_message(
                            session_id=session_id,
                            user_id="system",
                            content="Session auto-summarized and saved to NAS",
                            role="system",
                            metadata={"auto_summarized": True}
                        )
                    else:
                        failed += 1
                except Exception as e:
                    logger.error(f"Failed to summarize session {session_id}: {e}")
                    failed += 1
        
        asyncio.run(summarize_sessions())
        
        logger.info(f"Session summarization completed - summarized: {summarized}, failed: {failed}")
        return {
            "status": "success",
            "summarized": summarized,
            "failed": failed,
            "total_sessions": len(session_ids)
        }
        
    except Exception as e:
        logger.error(f"Session summarization task failed: {e}")
        raise self.retry(exc=e, countdown=60, max_retries=3)

@celery_app.task(bind=True)
def process_memory_file(self, file_path):
    """Background task to process uploaded memory files"""
    try:
        logger.info(f"Processing memory file: {file_path}")
        
        # Use existing memory listener processor
        from core.memory_listener import MemoryProcessor
        processor = MemoryProcessor()
        
        # Process the file asynchronously
        async def process_file():
            await processor.process_file(file_path)
        
        asyncio.run(process_file())
        
        logger.info(f"Memory file processed successfully: {file_path}")
        return {"status": "success", "file_path": file_path}
        
    except Exception as e:
        logger.error(f"Memory file processing failed: {e}")
        raise self.retry(exc=e, countdown=30, max_retries=3)

@celery_app.task(bind=True)
def bulk_vector_update(self, session_ids):
    """Background task for bulk vector database updates"""
    try:
        logger.info(f"Starting bulk vector update for {len(session_ids)} sessions")
        
        manager = get_memory_manager()
        updated = 0
        failed = 0
        
        async def update_vectors():
            nonlocal updated, failed
            for session_id in session_ids:
                try:
                    # Get session entries
                    entries = manager.session_manager.get_session_entries(session_id)
                    
                    # Store each entry in vector DB
                    for entry in entries:
                        await manager.vector_manager.store_conversation_entry(entry)
                    
                    updated += 1
                    
                except Exception as e:
                    logger.error(f"Failed to update vectors for session {session_id}: {e}")
                    failed += 1
        
        asyncio.run(update_vectors())
        
        logger.info(f"Bulk vector update completed - updated: {updated}, failed: {failed}")
        return {
            "status": "success",
            "updated": updated,
            "failed": failed,
            "total_sessions": len(session_ids)
        }
        
    except Exception as e:
        logger.error(f"Bulk vector update failed: {e}")
        raise self.retry(exc=e, countdown=60, max_retries=3)