import requests
import json
from typing import Dict, Optional, List
from synology_api import filestation, core_sys_info
from config.settings import settings
from utils.logger import logger
import os

class NASClient:
    def __init__(self):
        self.host = settings.nas_host
        self.port = settings.nas_port
        self.username = settings.nas_username
        self.password = settings.nas_password
        self.base_path = "/home/trinityai/memory" #settings.nas_base_path
        try:
            self.fs = filestation.FileStation(
                ip_address=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                secure=True
            )
            logger.info("Successfully connected to NAS")
        except Exception as e:
            logger.error(f"Failed to connect to NAS: {e}")
            self.fs = None
    
    def create_folder_structure(self) -> bool:
        """Create the basic folder structure on NAS"""
        folders = [
            "agents",
            "agents/freelance",
            "agents/personal",
            "categories",
            "categories/health",
            "categories/finance",
            "categories/ideas",
            "categories/tasks",
            "categories/jobs",
            "categories/test",
            "logs",
            "metadata",
            "backups",
            "inbox"
        ]
        for folder in folders:
            try:
                response = self.fs.create_folder(self.base_path, folder)
                logger.info(f"Create folder response for {folder}: {response}")
            except Exception as e:
                # Folder might already exist
                logger.debug(f"Folder creation info for {folder}: {e}")
        
        return True
    
    def upload_file(self, local_path: str, remote_path: str) -> bool:
        """Upload file to NAS"""
        if not self.fs:
            logger.error("NAS not connected")
            return False
        
        try:
            self.fs.upload_file(remote_path, local_path)
            logger.info(f"Uploaded file: {local_path} -> {remote_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
            return False
    
    def download_file(self, remote_path: str, local_path: str) -> bool:
        """Download file from NAS"""
        if not self.fs:
            logger.error("NAS not connected")
            return False
        
        try:
            self.fs.download_file(remote_path, local_path)
            logger.info(f"Downloaded file: {remote_path} -> {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download file: {e}")
            return False
    
    def list_files(self, path: str) -> List[Dict]:
        """List files in directory"""
        if not self.fs:
            logger.error("NAS not connected")
            return []
        
        try:
            files = self.fs.get_list(path)
            return files.get('data', {}).get('files', [])
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []
    
    def save_memory_content(self, content: str, category: str, filename: str) -> Optional[str]:
        """Save memory content to appropriate NAS folder"""
        folder_path = f"{self.base_path}/categories/{category}"
        file_path = f"{folder_path}/{filename}"
        
        try:
            response = self.fs.create_folder(self.base_path, f"/categories/{category}")
            # logger.info(f"Create folder response for {category}: {response}")
        except Exception as e:
            # logger.debug(f"Folder creation info for {category}: {e}")
            None
        # Create temporary local file
        temp_path = f"temp/{filename}"
        os.makedirs("temp", exist_ok=True)
        
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            if self.upload_file(temp_path, file_path):
                os.remove(temp_path)
                return file_path
            else:
                os.remove(temp_path)
                return None
        except Exception as e:
            logger.error(f"Failed to save memory content: {e}")
            return None