import os
import sys
import subprocess
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

def check_dependencies():
    """Check if required external services are available"""
    print("🔍 Checking dependencies...")
    from config.settings import settings
    # Check Redis
    try:
        import redis
        r = redis.Redis.from_url(settings.redis_url)
        r.ping()
        print("Redis connection successful")
    except Exception as e:
        print(f"Redis connection failed: {e}")
        print("Install Redis: brew install redis (macOS) or apt install redis-server (Ubuntu)")
    
    # Check Pinecone
    pinecone_key = settings.pinecone_api_key
    if not pinecone_key:
        print("PINECONE_API_KEY not set")
        print("Get your API key from https://app.pinecone.io/")
    else:
        print("Pinecone API key configured")
    
    # Check OpenAI
    openai_key = settings.openai_api_key
    if not openai_key:
        print("OPENAI_API_KEY not set")
        print("Get your API key from https://platform.openai.com/api-keys")
    else:
        print("OpenAI API key configured")

def setup_vector_database():
    """Initialize Pinecone vector database"""
    print("Setting up vector database...")
    
    try:
        from pinecone import Pinecone
        from config.settings import settings
        
        # Initialize Pinecone
        pinecone = Pinecone(api_key=settings.pinecone_api_key)

        # Create or connect to index
        if settings.pinecone_index_name not in pinecone.list_indexes().names():
            pinecone.create_index_for_model(
                name=settings.pinecone_index_name,
                cloud="aws",
                region=settings.pinecone_environment,
                embed={
                    "model": "llama-text-embed-v2",
                    "field_map": {"text": "chunk_text"}
                }
            )
            print(f"Created Pinecone index: {settings.pinecone_index_name}")
        else:
            print(f"Pinecone index already exists: {settings.pinecone_index_name}")
          
    except Exception as e:
        print(f"Vector database setup failed: {e}")


def create_systemd_service():
    """Create systemd service for production deployment"""
    service_content = """
[Unit]
Description=Trinity Enhanced Memory System
After=network.target

[Service]
Type=simple
User=trinity
WorkingDirectory=/opt/trinity
Environment=PATH=/opt/trinity/venv/bin
ExecStart=/opt/trinity/venv/bin/python scripts/run_enhanced_system.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    
    service_path = "/etc/systemd/system/trinity-memory.service"
    print(f"Systemd service template created")
    print(f"To install: sudo cp {service_path} /etc/systemd/system/")
    print("Then: sudo systemctl enable trinity-memory && sudo systemctl start trinity-memory")


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
    
     # Check dependencies
    check_dependencies()
    
    # Setup vector database
    setup_vector_database()
    
    # Create directories
    directories = [
        "data", "data/sessions", "data/inbox", "data/manual_input",
        "data/archive", "logs", "temp", "credentials"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create systemd service template
    create_systemd_service()
    
    # Create test files
    create_test_files()
    
    if success:
        logger.info("Trinity Memory System setup completed successfully!")
        logger.info("You can now start the system with: python scripts/run_system.py")
    else:
        logger.info("Setup completed with some warnings. Check the logs above.")
    
    print("\nSetup complete!")
    print("Next steps:")
    print("1. Configure your .env file with API keys")
    print("2. Start Redis: redis-server")
    print("3. Run the system: python scripts/run_system.py")
    print("4. Test the API: curl http://localhost:5000/api/health")
    
    return success

if __name__ == "__main__":
    main()