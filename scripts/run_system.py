#!/usr/bin/env python3
"""
Main script to run the Trinity Memory System
"""

import os
import sys
import signal
import threading
import time
import asyncio
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from core.memory_manager import MemoryManager
from core.memory_listener import MemoryListener
from api.main import app
from utils.logger import logger
from config.settings import settings
import uvicorn

class TrinitySystem:
    def __init__(self):
        self.memory_listener = None
        self.memory_manager = None
        self.api_server = None
        self.running = False
        self.health_monitor = None
    
    async def initialize_memory(self):
        """Initialize the memory system"""
        try:
            logger.info("Initializing memory system...")
            self.memory_manager = MemoryManager()
            logger.info("memory system initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize memory system: {e}")
            return False
        
    def start_memory_listener(self):
        """Start the file watcher in a separate thread"""
        def run_listener():
            self.memory_listener = MemoryListener()
            self.memory_listener.start()
        
        listener_thread = threading.Thread(target=run_listener, daemon=True)
        listener_thread.start()
        logger.info("Memory listener started in background")
    
    def start_api_server(self):
        """Start the FastAPI server"""
        def run_api():
            uvicorn.run(
                app,
                host=settings.api_host,
                port=settings.api_port,
                log_level=settings.log_level.lower(),
            )
        
        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()
        logger.info(f"API server started at http://{settings.api_host}:{settings.api_port}")
    
    def start_health_monitor(self):
        """Start background health monitoring"""
        def health_check_loop():
            while self.running:
                try:
                    # Simple health check - could be expanded
                    if self.memory_manager:
                        # Log system status every hour
                        logger.info("System health check - all components running")
                    time.sleep(3600)  # Check every hour
                except Exception as e:
                    logger.error(f"Health monitor error: {e}")
                    time.sleep(60)
        
        health_thread = threading.Thread(target=health_check_loop, daemon=True)
        health_thread.start()
        logger.info("Health monitor started")

    async def start(self):
        """Start the complete Trinity system"""
        logger.info("Starting Trinity Memory System...")
        
        self.running = True
        
        if not await self.initialize_memory():
            logger.error("Failed to initialize memory system. Exiting.")
            return False
        
        # Start memory listener
        self.start_memory_listener()
        
        # Start memory listener for file watching
        self.start_memory_listener()
        
        # Start health monitoring
        self.start_health_monitor()
        
        # Start API server
        self.start_api_server()
        
        logger.info("Trinity Enhanced Memory System is running!")
        logger.info("Features enabled:")
        logger.info("   • Vector-based fast memory (Pinecone)")
        logger.info("   • Persistent NAS storage")
        logger.info("   • Conversation tracking")
        logger.info("   • Auto-summarization")
        logger.info("   • File watching")
        logger.info(f"Watching directories: {settings.watch_dirs_list}")
        logger.info(f"API available at: http://{settings.api_host}:{settings.api_port}")
        logger.info(f"Health check: http://{settings.api_host}:{settings.api_port}/api/health")
        logger.info("Press Ctrl+C to stop")
        
        # Keep main thread alive
        try:
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            self.stop()
    
    async def stop(self):
        """Stop the Trinity system"""
        logger.info("Stopping Trinity Memory System...")
        self.running = False
        
        # Stop memory listener
        if self.memory_listener:
            self.memory_listener.stop()
        
        if self.memory_manager:
            await self.memory_manager.cleanup_old_memories()
        
        logger.info("Trinity Memory System stopped")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal")
    sys.exit(0)

async def main():
    """Main function"""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start the system
    trinity_system = TrinitySystem()
    await trinity_system.start()

if __name__ == "__main__":
    asyncio.run(main())