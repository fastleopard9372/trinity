# ===== core/memory_listener.py =====
import os
import time
import asyncio
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from typing import Dict, List
from datetime import datetime
import aiofiles
from pathlib import Path
import json

from clients.nas_client import NASClient
from clients.gdrive_client import GoogleDriveClient
from core.command_parser import CommandParser
from models.memory import DatabaseManager
from utils.logger import logger
from config.settings import settings

class MemoryFileHandler(FileSystemEventHandler):
    """File system event handler for memory files"""
    
    def __init__(self, memory_processor):
        self.memory_processor = memory_processor
        self.processing_files = set()  # Track files being processed
        
    def on_created(self, event):
        """Handle file creation events"""
        if not event.is_directory:
            logger.info(f"New file detected: {event.src_path}")
            # Use thread-safe method to schedule async processing
            self._schedule_processing(event.src_path)
    
    def on_modified(self, event):
        """Handle file modification events"""
        if not event.is_directory and event.src_path not in self.processing_files:
            logger.info(f"File modified: {event.src_path}")
            self._schedule_processing(event.src_path)
    
    def on_moved(self, event):
        """Handle file move events"""
        if not event.is_directory:
            logger.info(f"File moved: {event.src_path} -> {event.dest_path}")
            self._schedule_processing(event.dest_path)
    
    def _schedule_processing(self, file_path: str):
        """Schedule async file processing in a thread-safe way"""
        if file_path in self.processing_files:
            logger.debug(f"File already being processed: {file_path}")
            return
        
        self.processing_files.add(file_path)
        
        # Create a new thread for async processing
        def process_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.memory_processor.process_file(file_path))
            finally:
                loop.close()
                self.processing_files.discard(file_path)
        
        thread = threading.Thread(target=process_in_thread, daemon=True)
        thread.start()

class MemoryProcessor:
    """Core memory processing engine"""
    
    def __init__(self):
        self.nas_client = NASClient()
        self.gdrive_client = GoogleDriveClient()
        self.command_parser = CommandParser()
        self.db_manager = DatabaseManager(settings.memory_db_path)
        
        # Ensure directories exist
        for directory in settings.watch_dirs_list:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Watching directory: {directory}")
        
        # Create other required directories
        os.makedirs("data/archive", exist_ok=True)
        os.makedirs("temp", exist_ok=True)
        
        # Track processing statistics
        self.stats = {
            'files_processed': 0,
            'files_failed': 0,
            'last_processed': None,
            'total_memories_created': 0
        }
    
    async def process_file(self, file_path: str):
        """Process a new or modified file"""
        try:
            logger.info(f"Processing file: {file_path}")
            
            # Wait a moment for file to be fully written
            await asyncio.sleep(2)
            
            # Check if file still exists and is readable
            if not os.path.exists(file_path):
                logger.warning(f"File no longer exists: {file_path}")
                return
            
            # Check file size to avoid processing empty files
            if os.path.getsize(file_path) == 0:
                logger.warning(f"Empty file ignored: {file_path}")
                await self._archive_file(file_path, "empty")
                return
            
            # Read file content with encoding detection
            content = await self._read_file_content(file_path)
            
            if not content or not content.strip():
                logger.warning(f"No content found in file: {file_path}")
                await self._archive_file(file_path, "no_content")
                return
            
            # Parse the content for memory commands
            command = self.command_parser.parse(content)
            logger.info(f"Parsed command - Action: {command.action}, Category: {command.category}")
            
            # Generate unique filename based on content and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:17]  # Include microseconds
            safe_filename = self._generate_safe_filename(command, timestamp)
            
            # Save to storage systems
            nas_path = await self._save_to_nas(command.content, command.category, safe_filename)
            gdrive_file_id = await self._save_to_gdrive(command.content, safe_filename)
            
            # Save to database
            memory_data = {
                "content": command.content,
                "title": self._generate_title(command.content),
                "tags": command.tags,
                "category": command.category,
                "file_path": file_path,
                "nas_path": nas_path,
                "gdrive_path": gdrive_file_id,
                "metadata": {
                    **command.metadata,
                    "original_filename": os.path.basename(file_path),
                    "processed_at": datetime.utcnow().isoformat(),
                    "file_size": os.path.getsize(file_path),
                    "priority": command.priority
                },
                "source": "file_watcher"
            }
            
            memory_record = self.db_manager.add_memory(memory_data)
            logger.info(f"Memory saved with ID: {memory_record.id}")
            
            # Update statistics
            self.stats['files_processed'] += 1
            self.stats['total_memories_created'] += 1
            self.stats['last_processed'] = datetime.utcnow().isoformat()
            
            # Move processed file to archive
            await self._archive_file(file_path, "processed")
            
            logger.info(f"Successfully processed file: {file_path} -> Memory ID: {memory_record.id}")
            
        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            self.stats['files_failed'] += 1
            
            # Try to archive the failed file for manual inspection
            try:
                await self._archive_file(file_path, "failed")
            except Exception as archive_error:
                logger.error(f"Failed to archive failed file: {archive_error}")
    
    async def _read_file_content(self, file_path: str) -> str:
        """Read file content with encoding detection and error handling"""
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
                    content = await f.read()
                logger.debug(f"Successfully read file with {encoding} encoding")
                return content
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error reading file with {encoding}: {e}")
                continue
        
        # If all encodings fail, try binary mode and decode with error handling
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                raw_content = await f.read()
            content = raw_content.decode('utf-8', errors='replace')
            logger.warning(f"File read with error replacement: {file_path}")
            return content
        except Exception as e:
            logger.error(f"Failed to read file in any encoding: {e}")
            raise
    
    def _generate_safe_filename(self, command, timestamp: str) -> str:
        """Generate a safe filename for storage"""
        # Create base name from title or content
        if command.content:
            title = command.content.split('\n')[0][:30]  # First line, max 30 chars
            # Remove invalid characters
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_title = safe_title.replace(' ', '_')
        else:
            safe_title = "memory"
        
        # Combine with timestamp and category
        filename = f"memory_{timestamp}_{command.category}_{safe_title}.txt"
        
        # Ensure filename isn't too long
        if len(filename) > 100:
            filename = f"memory_{timestamp}_{command.category}.txt"
        
        return filename
    
    async def _save_to_nas(self, content: str, category: str, filename: str) -> str:
        """Save content to NAS storage"""
        try:
            nas_path = self.nas_client.save_memory_content(content, category, filename)
            if nas_path:
                logger.info(f"Saved to NAS: {nas_path}")
                return nas_path
            else:
                logger.warning("Failed to save to NAS")
                return None
        except Exception as e:
            logger.error(f"NAS save error: {e}")
            return None
    
    async def _save_to_gdrive(self, content: str, filename: str) -> str:
        """Save content to Google Drive"""
        try:
            # Create temporary file
            temp_file = f"temp/{filename}"
            
            async with aiofiles.open(temp_file, 'w', encoding='utf-8') as f:
                await f.write(content)
            
            # Upload to Google Drive
            file_id = self.gdrive_client.upload_file(
                temp_file, 
                filename, 
                self.gdrive_client.folder_id
            )
            
            # Clean up temp file
            try:
                os.remove(temp_file)
            except:
                pass
            
            if file_id:
                logger.info(f"Saved to Google Drive: {file_id}")
                return file_id
            else:
                logger.warning("Failed to save to Google Drive")
                return None
                
        except Exception as e:
            logger.error(f"Google Drive save error: {e}")
            return None
    
    def _generate_title(self, content: str, max_length: int = 50) -> str:
        """Generate a title from content"""
        if not content:
            return "Untitled Memory"
        
        lines = content.strip().split('\n')
        
        # Try to find a meaningful first line
        for line in lines:
            clean_line = line.strip()
            if clean_line and not clean_line.lower().startswith(('save this', 'remember this', 'log this', 'tags:', 'category:')):
                if len(clean_line) <= max_length:
                    return clean_line
                else:
                    return clean_line[:max_length-3] + "..."
        
        # Fallback to first line
        first_line = lines[0].strip() if lines else ""
        if len(first_line) <= max_length:
            return first_line or "Untitled Memory"
        else:
            return first_line[:max_length-3] + "..."
    
    async def _archive_file(self, file_path: str, status: str = "processed"):
        """Move processed file to archive folder"""
        try:
            archive_dir = f"data/archive/{status}"
            os.makedirs(archive_dir, exist_ok=True)
            
            filename = os.path.basename(file_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_filename = f"{timestamp}_{filename}"
            archive_path = os.path.join(archive_dir, archive_filename)
            
            # Ensure unique filename
            counter = 1
            original_archive_path = archive_path
            while os.path.exists(archive_path):
                base, ext = os.path.splitext(original_archive_path)
                archive_path = f"{base}_{counter}{ext}"
                counter += 1
            
            os.rename(file_path, archive_path)
            logger.info(f"File archived ({status}): {file_path} -> {archive_path}")
            
        except Exception as e:
            logger.error(f"Failed to archive file {file_path}: {e}")
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        return {
            **self.stats,
            'nas_connected': self.nas_client.fs is not None,
            'gdrive_connected': self.gdrive_client.service is not None,
            'db_accessible': True  # If we got this far, DB is accessible
        }

class MemoryListener:
    """Main memory listener service"""
    
    def __init__(self):
        self.memory_processor = MemoryProcessor()
        self.observer = Observer()
        self.running = False
        self.start_time = None
        
        # Setup event handlers for each watch directory
        self.handlers = {}
        for directory in settings.watch_dirs_list:
            handler = MemoryFileHandler(self.memory_processor)
            self.handlers[directory] = handler
    
    def start(self):
        """Start watching directories for new files"""
        logger.info("Starting Trinity Memory Listener...")
        
        self.running = True
        self.start_time = datetime.utcnow()
        
        # Set up file watchers for each directory
        for directory in settings.watch_dirs_list:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created watch directory: {directory}")
            
            handler = self.handlers[directory]
            self.observer.schedule(handler, directory, recursive=True)
            logger.info(f"Started watching: {directory}")
        
        self.observer.start()
        logger.info("Memory listener started successfully")
        
        # Print helpful information
        logger.info(f"Watching {len(settings.watch_dirs_list)} directories:")
        for directory in settings.watch_dirs_list:
            logger.info(f"   - {os.path.abspath(directory)}")
        
        logger.info("Drop text files in watch directories to create memories")
        logger.info("Use commands like: 'Save this:', 'Tags:', 'Category:'")
        
        try:
            while self.running:
                time.sleep(1)
                
                # Periodic status logging (every 5 minutes)
                if int(time.time()) % 300 == 0:
                    stats = self.memory_processor.get_stats()
                    logger.info(f"Status - Processed: {stats['files_processed']}, Failed: {stats['files_failed']}")
                    
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
            self.stop()
    
    def stop(self):
        """Stop the file watcher"""
        logger.info("Stopping Memory Listener...")
        
        self.running = False
        self.observer.stop()
        self.observer.join()
        
        # Log final statistics
        stats = self.memory_processor.get_stats()
        uptime = datetime.utcnow() - self.start_time if self.start_time else None
        
        logger.info("Final Statistics:")
        logger.info(f"   - Files Processed: {stats['files_processed']}")
        logger.info(f"   - Files Failed: {stats['files_failed']}")
        logger.info(f"   - Memories Created: {stats['total_memories_created']}")
        logger.info(f"   - Uptime: {uptime}")
        
        logger.info("Memory listener stopped")
    
    def get_status(self) -> Dict:
        """Get current listener status"""
        stats = self.memory_processor.get_stats()
        return {
            'running': self.running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime_seconds': (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0,
            'watch_directories': settings.watch_dirs_list,
            'processing_stats': stats
        }
    
    def process_file_manually(self, file_path: str):
        """Manually process a specific file (for testing)"""
        logger.info(f"Manually processing file: {file_path}")
        
        # Create a new thread for async processing
        def process_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.memory_processor.process_file(file_path))
                logger.info(f"Manual processing completed: {file_path}")
            except Exception as e:
                logger.error(f"Manual processing failed: {e}")
            finally:
                loop.close()
        
        thread = threading.Thread(target=process_in_thread, daemon=True)
        thread.start()
        return thread

# ===== Utility functions for standalone usage =====

def run_memory_listener():
    """Standalone function to run the memory listener"""
    try:
        listener = MemoryListener()
        listener.start()
    except KeyboardInterrupt:
        logger.info("Memory listener stopped by user")
    except Exception as e:
        logger.error(f"Memory listener crashed: {e}")
        raise

def test_file_processing():
    """Test function to validate file processing"""
    logger.info("Testing file processing...")
    
    # Create test file
    test_content = """Save this: This is a test memory for validation.
Tags: test, validation, system-check
Category: test
Priority: 1

The memory listener should process this file automatically and:
1. Parse the command and extract metadata
2. Save to NAS storage
3. Upload to Google Drive
4. Store in database
5. Archive the original file
"""
    
    test_file = f"{settings.watch_dirs_list[0]}/test_processing.txt"
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    logger.info(f"Created test file: {test_file}")
    logger.info("File should be processed automatically by the memory listener")
    logger.info("Check logs and database for processing confirmation")

if __name__ == "__main__":
    """Run memory listener as standalone script"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_file_processing()
    else:
        run_memory_listener()