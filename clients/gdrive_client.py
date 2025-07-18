import os
import json
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from typing import Optional, List, Dict
import io
from config.settings import settings
from utils.logger import logger

class GoogleDriveClient:
    SCOPES = ['https://www.googleapis.com/auth/drive']
    
    def __init__(self):
        self.service = self._authenticate()
        self.folder_id = None #settings.google_drive_folder_id
    
    def _authenticate(self):
        """Authenticate with Google Drive API"""
        creds = None
        token_path = 'credentials/token.json'
        
        # Load existing token
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, self.SCOPES)
        # If no valid credentials, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(settings.google_credentials_path):
                    logger.error(f"Google credentials file not found: {settings.google_credentials_path}")
                    return None
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    settings.google_credentials_path, self.SCOPES)
                creds = flow.run_local_server(port=5000)
            
            # Save credentials for next run
            os.makedirs('credentials', exist_ok=True)
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
        
        try:
            service = build('drive', 'v3', credentials=creds, cache_discovery=False)
            logger.info("Successfully authenticated with Google Drive")
            return service
        except Exception as e:
            logger.error(f"Failed to build Google Drive service: {e}")
            return None
    
    def create_folder_structure(self) -> bool:
        if not self.service:
            return False

        folders = [
            "trinity_memory/agents/freelance",
            "trinity_memory/agents/personal",
            "trinity_memory/categories/health",
            "trinity_memory/categories/finance",
            "trinity_memory/categories/ideas",
            "trinity_memory/categories/tasks",
            "trinity_memory/categories/jobs",
            "trinity_memory/categories/sessions",
            "trinity_memory/logs",
            "trinity_memory/metadata",
            "trinity_memory/backups"
        ]

        # Set root folder ID
        self.folder_id = self._get_folder_id("trinity_memory")

        for path in folders:
            folder_id = self._get_folder_id(path)
            if folder_id:
                logger.info(f"Ensured folder: {path}")
            else:
                logger.error(f"Failed to create or locate folder: {path}")

        return True

    
    def _create_folder(self, name: str, parent_id: Optional[str]) -> Optional[Dict]:
        """Create a folder in Google Drive"""
        if not self.service:
            return None
        
        folder_metadata = {
            'name': name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        
        if parent_id:
            folder_metadata['parents'] = [parent_id]
        
        try:
            folder = self.service.files().create(body=folder_metadata, fields='id,name').execute()
            logger.info(f"Created Google Drive folder: {name}")
            return folder
        except Exception as e:
            logger.error(f"Failed to create folder {name}: {e}")
            return None
    
    def _get_folder_id(self, path: str) -> Optional[str]:
        if not self.service:
            return None

        folder_names = path.strip("/").split("/")
        parent_id = "root"

        for folder_name in folder_names:
            query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and '{parent_id}' in parents and trashed=false"
            response = self.service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name)',
                pageSize=1
            ).execute()
            files = response.get('files', [])

            if files:
                parent_id = files[0]['id']
            else:
                metadata = {
                    'name': folder_name,
                    'mimeType': 'application/vnd.google-apps.folder',
                    'parents': [parent_id]
                }
                folder = self.service.files().create(body=metadata, fields='id').execute()
                logger.info(f"Created folder: {folder_name}")
                parent_id = folder['id']

        return parent_id
    
    def upload_file(self, local_path: str, remote_name: str, folder_id: Optional[str] = None) -> Optional[str]:
        """Upload file to Google Drive"""
        if not self.service:
            return None
        
        if not folder_id:
            folder_id = self.folder_id
        
        file_metadata = {
            'name': remote_name,
            'parents': [folder_id] if folder_id else []
        }
        
        try:
            media = MediaFileUpload(local_path, resumable=True)
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id,name'
            ).execute()
            
            logger.info(f"Uploaded file to Google Drive: {remote_name}")
            return file.get('id')
        except Exception as e:
            logger.error(f"Failed to upload file to Google Drive: {e}")
            return None
    
    def sync_folder(self, local_folder: str, remote_folder_id: str) -> bool:
        """Sync local folder with Google Drive folder"""
        # This is a simplified sync - in production, use rclone or implement proper sync logic
        if not os.path.exists(local_folder):
            return False
        
        for root, dirs, files in os.walk(local_folder):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, local_folder)
                
                # Upload file
                self.upload_file(local_file_path, relative_path, remote_folder_id)
        
        return True