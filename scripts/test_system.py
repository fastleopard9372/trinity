#!/usr/bin/env python3
"""
Test script for Trinity Memory System
"""

import os
import sys
import time
import requests
import json
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from clients.nas_client import NASClient
from clients.gdrive_client import GoogleDriveClient
from core.command_parser import CommandParser
from models.memory import DatabaseManager
from config.settings import settings
from utils.logger import logger

def test_command_parser():
    """Test the command parser functionality"""
    logger.info("Testing command parser...")
    
    parser = CommandParser()
    
    test_cases = [
        "Save this: I need to remember to take my vitamins daily. Tags: health, reminder",
        "Remember this idea: Build a voice-controlled home automation system. Category: ideas",
        "Log this task: Complete the Trinity project by next Friday. Priority: 5",
        "Keep this note: Meeting with client at 3 PM tomorrow about the new website.",
    ]
    
    for test_case in test_cases:
        command = parser.parse(test_case)
        logger.info(f"Input: {test_case[:50]}...")
        logger.info(f"Parsed - Action: {command.action}, Category: {command.category}, Tags: {command.tags}")
    
    logger.info("Command parser tests completed")

def test_database():
    """Test database functionality"""
    logger.info("Testing database...")
    
    db_manager = DatabaseManager(settings.memory_db_path)
    
    # Add test memory
    test_memory = {
        "content": "This is a test memory for database validation",
        "title": "Test Memory",
        "tags": ["test", "database"],
        "category": "test",
        "source": "test_script"
    }
    
    memory_record = db_manager.add_memory(test_memory)
    logger.info(f"Added memory with ID: {memory_record.id}")
    
    # Search for the memory
    memories = db_manager.search_memories("test memory")
    logger.info(f"Found {len(memories)} memories matching 'test memory'")
    
    # List memories
    all_memories = db_manager.get_memories(limit=10)
    logger.info(f"Total memories in database: {len(all_memories)}")
    
    logger.info("Database tests completed")

def test_nas_connection():
    """Test NAS connection"""
    logger.info("Testing NAS connection...")
    
    nas_client = NASClient()
    
    if nas_client.fs:
        logger.info("NAS connection successful")
        
        # Test file upload
        test_content = "This is a test file for NAS validation"
        test_file = "temp/nas_test.txt"
        
        os.makedirs("temp", exist_ok=True)
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        nas_path = nas_client.save_memory_content(test_content, "test", "nas_test.txt")
        
        if nas_path:
            logger.info(f"NAS file upload successful: {nas_path}")
        else:
            logger.error( f"NAS file upload failed")
        
        os.remove(test_file)
    else:
        logger.error("NAS connection failed")

def test_google_drive():
    """Test Google Drive connection"""
    logger.info("Testing Google Drive connection...")
    
    gdrive_client = GoogleDriveClient()
    
    if gdrive_client.service:
        logger.info("Google Drive connection successful")
        
        # Test file upload
        test_content = "This is a test file for Google Drive validation"
        test_file = "temp/gdrive_test.txt"
        
        os.makedirs("temp", exist_ok=True)
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        file_id = gdrive_client.upload_file(test_file, "gdrive_test.txt")
        
        if file_id:
            logger.info(f"Google Drive file upload successful: {file_id}")
        else:
            logger.error("Google Drive file upload failed")
        
        os.remove(test_file)
    else:
        logger.error("Google Drive connection failed")

def test_api_endpoints():
    """Test API endpoints"""
    logger.info("Testing API endpoints...")
    
    base_url = f"http://{settings.api_host}:{settings.api_port}"
    
    # Test status endpoint
    try:
        response = requests.get(f"{base_url}/status")
        if response.status_code == 200:
            status = response.json()
            logger.info(f"API status endpoint working: {status}")
        else:
            logger.error(f"API status endpoint failed: {response.status_code}")
    except requests.exceptions.ConnectionError:
        logger.error("API server not running or not accessible")
        return
    
    # Test save memory endpoint
    try:
        test_memory = {
            "content": "This is a test memory via API",
            "tags": ["test", "api"],
            "category": "test"
        }
        
        response = requests.post(
            f"{base_url}/api/memory/save",
            json=test_memory
        )
        
        if response.status_code == 200:
            memory = response.json()
            logger.info(f"API save memory successful: ID {memory['id']}")
        else:
            logger.error(f"API save memory failed: {response.status_code}")
    except Exception as e:
        logger.error(f"API save memory error: {e}")

def test_file_watcher():
    """Test file watcher functionality"""
    logger.info("Testing file watcher...")
    
    # Create test file in watch directory
    test_content = """Save this: This is a test for the file watcher system.
Tags: test, filewatcher
Category: test

The file watcher should detect this file and process it automatically.
"""
    
    test_file = f"{settings.watch_dirs_list[0]}/test_watcher.txt"
    
    with open(test_file, 'w') as f:
        f.write(test_content)
    
    logger.info(f"Created test file: {test_file}")
    logger.info("File should be processed automatically by the file watcher")
    logger.info("Check the logs and database for processing confirmation")

def main():
    """Run all tests"""
    logger.info("🧪 Starting Trinity Memory System tests...")
    
    # Run tests
    test_command_parser()
    test_database()
    test_nas_connection()
    test_google_drive()
    test_api_endpoints()
    test_file_watcher()
    
    logger.info("🏁 All tests completed! Check the logs for results.")

if __name__ == "__main__":
    main()