#!/usr/bin/env python3
"""
Main script to run the Trinity Memory System
"""

import os
import sys
import signal
import threading
import time
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from core.memory_listener import MemoryListener
from api.main import app
from utils.logger import logger
from config.settings import settings
import uvicorn

class TrinitySystem:
    def __init__(self):
        self.memory_listener = None
        self.api_server = None
        self.running = False
    
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
    
    def start(self):
        """Start the complete Trinity system"""
        logger.info("Starting Trinity Memory System...")
        
        self.running = True
        
        # Start memory listener
        self.start_memory_listener()
        
        # Start API server
        self.start_api_server()
        
        logger.info("Trinity Memory System is running!")
        logger.info(f"Watching directories: {settings.watch_dirs_list}")
        logger.info(f"API available at: http://{settings.api_host}:{settings.api_port}")
        logger.info("Drop files in watch directories to create memories")
        logger.info("Press Ctrl+C to stop")
        
        # Keep main thread alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop the Trinity system"""
        logger.info("Stopping Trinity Memory System...")
        self.running = False
        
        if self.memory_listener:
            self.memory_listener.stop()
        
        logger.info("Trinity Memory System stopped")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal")
    sys.exit(0)

def main():
    """Main function"""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start the system
    trinity_system = TrinitySystem()
    trinity_system.start()

if __name__ == "__main__":
    main()