
import os
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from clients.nas_client import NASClient
from clients.gdrive_client import GoogleDriveClient
from models.memory import DatabaseManager
from config.settings import settings
from utils.logger import logger

def setup_directories():
    """Create all necessary local directories"""
    directories = [
        "data",
        "data/inbox",
        "data/manual_input",
        "data/archive",
        "data/uploads",
        "logs",
        "temp",
        "credentials"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    # Create watch directories from settings
    for directory in settings.watch_dirs_list:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created watch directory: {directory}")

def setup_database():
    """Initialize the database"""
    try:
        db_manager = DatabaseManager(settings.memory_db_path)
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False

def setup_nas():
    """Setup NAS connection and folder structure"""
    try:
        nas_client = NASClient()
        if nas_client.fs:
            nas_client.create_folder_structure()
            logger.info("NAS setup completed successfully")
            return True
        else:
            logger.error("Failed to connect to NAS")
            return False
    except Exception as e:
        logger.error(f"NAS setup failed: {e}")
        return False

def setup_google_drive():
    """Setup Google Drive connection and folder structure"""
    try:
        gdrive_client = GoogleDriveClient()
        if gdrive_client.service:
            gdrive_client.create_folder_structure()
            logger.info("Google Drive setup completed successfully")
            return True
        else:
            logger.error("Failed to connect to Google Drive")
            return False
    except Exception as e:
        logger.error(f"Google Drive setup failed: {e}")
        return False

def create_test_files():
    """Create test files for validation"""
    test_files = [
        {
            "path": "data/inbox/test_memory.txt",
            "content": """Save this: This is a test memory for the Trinity system.
Tags: test, demo, health
Category: health
Priority: 1

This memory should be processed automatically by the file watcher.
"""
        },
        {
            "path": "data/manual_input/idea_note.txt",
            "content": """Remember this idea: Build an AI assistant that can remember everything.

Tags: ai, idea, innovation
Category: ideas

The assistant should have persistent memory across sessions and be able to recall specific information when needed.
"""
        }
    ]
    
    for test_file in test_files:
        with open(test_file["path"], 'w', encoding='utf-8') as f:
            f.write(test_file["content"])
        logger.info(f"Created test file: {test_file['path']}")

def main():
    """Main setup function"""
    logger.info("Starting Trinity Memory System setup...")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        logger.error(".env file not found. Please create one based on .env.example")
        return False
    
    success = True
    
    # Setup local directories
    setup_directories()
    
    # Setup database
    if not setup_database():
        success = False
    
    # Setup NAS
    if not setup_nas():
        logger.warning("NAS setup failed - check your NAS configuration")
        success = False
    
    # Setup Google Drive
    if not setup_google_drive():
        logger.warning("Google Drive setup failed - check your Google credentials")
        success = False
    
    # Create test files
    create_test_files()
    
    if success:
        logger.info("Trinity Memory System setup completed successfully!")
        logger.info("You can now start the system with: python scripts/run_system.py")
    else:
        logger.info("Setup completed with some warnings. Check the logs above.")
    
    return success

if __name__ == "__main__":
    main()