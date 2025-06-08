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
    nas_base_path: str = "/trinity_memory"
    
    # Google Drive Configuration
    google_credentials_path: str = "credentials/google_credentials.json"
    google_drive_folder_id: str = ""
    
    # API Configuration
    openai_api_key: str = ""
    
    # Memory Configuration
    memory_db_path: str = "data/trinity_memory.db"
    watch_directories: str = "data/inbox,data/manual_input"
    sync_interval: int = 300
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/trinity.log"
    
    @property
    def watch_dirs_list(self) -> List[str]:
        return [d.strip() for d in self.watch_directories.split(",")]
    
    class Config:
        env_file = ".env"

settings = Settings()