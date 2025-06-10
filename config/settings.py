import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # APP Configuration
    api_host: str = "http://localhost"
    api_port: int = 8000
    # NAS Configuration
    nas_host: str = "192.168.1.100"
    nas_port: int = 5000
    nas_username: str = "admin"
    nas_password: str = ""
    nas_base_path: str = "/home"
    
    # Google Drive Configuration
    google_credentials_path: str = "credentials/google_credentials.json"
    google_drive_folder_id: str = ""
    
    # Vector Database Configuration
    pinecone_api_key: str = ""
    pinecone_environment: str = "us-west1-gcp"
    pinecone_index_name: str = "trinity-memory"
    
    # OpenAI Configuration
    openai_api_key: str = ""
    
    # Memory Configuration
    memory_db_path: str = "data/trinity_memory.db"
    watch_directories: str = "data/inbox,data/manual_input"
    sync_interval: int = 300
    
    # Auto-save Configuration
    auto_save_threshold: int = 10
    vector_cleanup_days: int = 30
    session_archive_days: int = 90
    
    # Background Tasks
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/trinity.log"
    
    # Additional fields causing the validation error
    secret_key: str = "your_secret_key_here"  # Add default values or ensure they are set in your .env file
    allowed_hosts: str = "localhost,127.0.0.1"
    max_vector_results: int = 50
    max_nas_results: int = 20
    session_timeout_hours: int = 24
    cleanup_interval_hours: int = 6
    
    @property
    def watch_dirs_list(self) -> List[str]:
        return [d.strip() for d in self.watch_directories.split(",")]
    
    class Config:
        env_file = ".env"  # Ensure that your .env file has values for the new fields
        extra = "allow"

settings = Settings()
