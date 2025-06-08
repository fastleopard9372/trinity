from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
import os
import asyncio
from datetime import datetime
import json

from clients.nas_client import NASClient
from clients.gdrive_client import GoogleDriveClient
from core.command_parser import CommandParser
from core.memory_listener import MemoryListener, MemoryProcessor
from models.memory import DatabaseManager
from utils.logger import logger
from config.settings import settings

app = FastAPI(title="Trinity Memory System API", version="1.0.0")

# Initialize components
nas_client = NASClient()
gdrive_client = GoogleDriveClient()
command_parser = CommandParser()
db_manager = DatabaseManager(settings.memory_db_path)
memory_processor = MemoryProcessor()

# Pydantic models for API
class MemoryInput(BaseModel):
    content: str
    tags: Optional[List[str]] = []
    category: Optional[str] = None
    priority: Optional[int] = 1

class MemoryResponse(BaseModel):
    id: int
    content: str
    title: str
    tags: List[str]
    category: str
    timestamp: str
    file_path: Optional[str]
    nas_path: Optional[str]
    gdrive_path: Optional[str]

class SystemStatus(BaseModel):
    nas_connected: bool
    gdrive_connected: bool
    database_accessible: bool
    watch_directories: List[str]
    total_memories: int

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize system on startup"""
    logger.info("Starting Trinity Memory System API")
    
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    os.makedirs("credentials", exist_ok=True)
    
    for directory in settings.watch_dirs_list:
        os.makedirs(directory, exist_ok=True)
    
    # Initialize folder structures
    nas_client.create_folder_structure()
    gdrive_client.create_folder_structure()
    
    logger.info("Trinity Memory System API started successfully")

@app.get("/init")
async def init():
    """Initialize system on startup"""
    logger.info("Starting Trinity Memory System API")
    
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    os.makedirs("credentials", exist_ok=True)
    
    for directory in settings.watch_dirs_list:
        os.makedirs(directory, exist_ok=True)
    
    # Initialize folder structures
    nas_client.create_folder_structure()
    gdrive_client.create_folder_structure()
    
    logger.info("Trinity Memory System API started successfully")
    return JSONResponse(content={"message": "System initialized successfully"})


@app.get("/")
async def root():
    return {"message": "Trinity Memory System API", "version": "1.0.0"}

@app.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get system status and health check"""
    
    # Check NAS connection
    nas_connected = nas_client.fs is not None
    
    # Check Google Drive connection
    gdrive_connected = gdrive_client.service is not None
          
    # Check database
    # logger.info(total_memories, "")
    try:
        total_memories = len(db_manager.get_memories(limit=1))
        db_accessible = True
    except:
        total_memories = 0
        db_accessible = False
    
    return SystemStatus(
        nas_connected=nas_connected,
        gdrive_connected=gdrive_connected,
        database_accessible=db_accessible,
        watch_directories=settings.watch_dirs_list,
        total_memories=total_memories
    )

@app.post("/api/memory/save", response_model=MemoryResponse)
async def save_memory(memory_input: MemoryInput):
    """Save a new memory through API"""
    try:
        # Parse the content
        command = command_parser.parse(memory_input.content)
        
        # Override with API inputs if provided
        if memory_input.tags:
            command.tags = memory_input.tags
        if memory_input.category:
            command.category = memory_input.category
        if memory_input.priority:
            command.priority = memory_input.priority
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"memory_api_{timestamp}_{command.category}.txt"
        
        # Save to NAS
        nas_path = nas_client.save_memory_content(
            command.content, 
            command.category, 
            filename
        )
        
        # Save to Google Drive
        gdrive_file_id = None
        temp_file = f"temp/{filename}"
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(command.content)
            
            gdrive_file_id = gdrive_client.upload_file(
                temp_file, 
                filename, 
                gdrive_client.folder_id
            )
            
            os.remove(temp_file)
        except Exception as e:
            logger.error(f"Failed to upload to Google Drive: {e}")
        
        # Save to database
        memory_data = {
            "content": command.content,
            "title": memory_processor._generate_title(command.content),
            "tags": command.tags,
            "category": command.category,
            "nas_path": nas_path,
            "gdrive_path": gdrive_file_id,
            "metadata": command.metadata,
            "source": "api"
        }
        
        memory_record = db_manager.add_memory(memory_data)
        
        return MemoryResponse(
            id=memory_record.id,
            content=memory_record.content,
            title=memory_record.title,
            tags=json.loads(memory_record.tags) if memory_record.tags else [],
            category=memory_record.category,
            timestamp=memory_record.timestamp.isoformat(),
            file_path=memory_record.file_path,
            nas_path=memory_record.nas_path,
            gdrive_path=memory_record.gdrive_path
        )
        
    except Exception as e:
        logger.error(f"Failed to save memory: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save memory: {str(e)}")

@app.post("/api/memory/upload")
async def upload_memory_file(file: UploadFile = File(...), category: str = Form("general")):
    """Upload a file as memory content"""
    try:
        # Read file content
        content = await file.read()
        
        # Handle different file types
        if file.content_type.startswith('text/'):
            text_content = content.decode('utf-8')
        else:
            # For non-text files, save the file and store reference
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{file.filename}"
            file_path = f"data/uploads/{filename}"
            
            os.makedirs("data/uploads", exist_ok=True)
            
            with open(file_path, 'wb') as f:
                f.write(content)
            
            text_content = f"Uploaded file: {file.filename}\nFile path: {file_path}\nFile type: {file.content_type}"
        
        # Process as memory
        memory_input = MemoryInput(
            content=text_content,
            category=category,
            tags=[file.content_type, "upload"]
        )
        
        return await save_memory(memory_input)
        
    except Exception as e:
        logger.error(f"Failed to upload file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

@app.get("/api/memory/list", response_model=List[MemoryResponse])
async def list_memories(
    limit: int = 50, 
    category: Optional[str] = None,
    skip: int = 0
):
    """List memories with optional filtering"""
    try:
        memories = db_manager.get_memories(limit=limit, category=category)
        
        return [
            MemoryResponse(
                id=memory.id,
                content=memory.content,
                title=memory.title or "",
                tags=json.loads(memory.tags) if memory.tags else [],
                category=memory.category or "",
                timestamp=memory.timestamp.isoformat() if memory.timestamp else "",
                file_path=memory.file_path,
                nas_path=memory.nas_path,
                gdrive_path=memory.gdrive_path
            )
            for memory in memories[skip:skip+limit]
        ]
        
    except Exception as e:
        logger.error(f"Failed to list memories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list memories: {str(e)}")

@app.get("/api/memory/search", response_model=List[MemoryResponse])
async def search_memories(
    q: str,
    limit: int = 50
):
    """Search memories by keyword"""
    try:
        memories = db_manager.search_memories(q, limit=limit)
        
        return [
            MemoryResponse(
                id=memory.id,
                content=memory.content,
                title=memory.title or "",
                tags=json.loads(memory.tags) if memory.tags else [],
                category=memory.category or "",
                timestamp=memory.timestamp.isoformat() if memory.timestamp else "",
                file_path=memory.file_path,
                nas_path=memory.nas_path,
                gdrive_path=memory.gdrive_path
            )
            for memory in memories
        ]
        
    except Exception as e:
        logger.error(f"Failed to search memories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search memories: {str(e)}")

@app.post("/api/system/sync")
async def trigger_sync():
    """Manually trigger sync between NAS and Google Drive"""
    try:
        # Sync from NAS to Google Drive
        success = gdrive_client.sync_folder("data", gdrive_client.folder_id)
        
        if success:
            return {"message": "Sync completed successfully"}
        else:
            raise HTTPException(status_code=500, detail="Sync failed")
            
    except Exception as e:
        logger.error(f"Sync failed: {e}")
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")

@app.post("/api/system/test-command")
async def test_command_parsing(content: str):
    """Test command parsing functionality"""
    try:
        command = command_parser.parse(content)
        
        return {
            "action": command.action,
            "content": command.content,
            "tags": command.tags,
            "category": command.category,
            "priority": command.priority,
            "metadata": command.metadata
        }
        
    except Exception as e:
        logger.error(f"Command parsing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Command parsing failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )