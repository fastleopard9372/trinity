from sqlalchemy import Column, Integer, String, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from datetime import datetime
from typing import List, Dict, Optional
import json

Base = declarative_base()

class MemoryRecord(Base):
    __tablename__ = "memories"
    
    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    title = Column(String(255), nullable=True)
    tags = Column(Text, nullable=True)  # JSON string
    category = Column(String(100), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    file_path = Column(String(500), nullable=True)
    nas_path = Column(String(500), nullable=True)
    gdrive_path = Column(String(500), nullable=True)
    meta_data = Column(Text, nullable=True)  # JSON string
    source = Column(String(100), default="manual")
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "content": self.content,
            "title": self.title,
            "tags": json.loads(self.tags) if self.tags else [],
            "category": self.category,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "file_path": self.file_path,
            "nas_path": self.nas_path,
            "gdrive_path": self.gdrive_path,
            "meta_data": json.loads(self.meta_data) if self.meta_data else {},
            "source": self.source
        }

class DatabaseManager:
    def __init__(self, db_path: str):
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(bind=self.engine)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.session = SessionLocal()
    
    def add_memory(self, memory_data: Dict) -> MemoryRecord:
        memory = MemoryRecord(
            content=memory_data.get("content", ""),
            title=memory_data.get("title"),
            tags=json.dumps(memory_data.get("tags", [])),
            category=memory_data.get("category"),
            file_path=memory_data.get("file_path"),
            nas_path=memory_data.get("nas_path"),
            gdrive_path=memory_data.get("gdrive_path"),
            meta_data=json.dumps(memory_data.get("meta_data", {})),
            source=memory_data.get("source", "manual")
        )
        
        self.session.add(memory)
        self.session.commit()
        self.session.refresh(memory)
        return memory
    
    def get_memories(self, limit: int = 100, category: str = None) -> List[MemoryRecord]:
        query = self.session.query(MemoryRecord)
        
        if category:
            query = query.filter(MemoryRecord.category == category)
        
        return query.order_by(MemoryRecord.timestamp.desc()).limit(limit).all()
    
    def search_memories(self, keyword: str, limit: int = 50) -> List[MemoryRecord]:
        return self.session.query(MemoryRecord).filter(
            MemoryRecord.content.contains(keyword)
        ).order_by(MemoryRecord.timestamp.desc()).limit(limit).all()