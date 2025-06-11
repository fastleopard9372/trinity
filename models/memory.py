from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import re
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta

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

class SmartQueryEngine:
    """Enhanced query engine with natural language processing and FTS"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.time_patterns = {
            'today': lambda: datetime.now().date(),
            'yesterday': lambda: (datetime.now() - timedelta(days=1)).date(),
            'this week': lambda: self._get_week_range(),
            'last week': lambda: self._get_week_range(-1),
            'this month': lambda: self._get_month_range(),
            'last month': lambda: self._get_month_range(-1),
            r'(\d+) days? ago': lambda m: (datetime.now() - timedelta(days=int(m.group(1)))).date(),
            r'last (\w+)': self._parse_last_weekday,
        }
    
    def _get_week_range(self, weeks_ago=0):
        """Get start and end date for a week"""
        today = datetime.now().date()
        start_of_week = today - timedelta(days=today.weekday()) - timedelta(weeks=weeks_ago)
        end_of_week = start_of_week + timedelta(days=6)
        return start_of_week, end_of_week
    
    def _get_month_range(self, months_ago=0):
        """Get start and end date for a month"""
        today = datetime.now().date()
        if months_ago == 0:
            start_of_month = today.replace(day=1)
            if today.month == 12:
                end_of_month = today.replace(year=today.year + 1, month=1, day=1) - timedelta(days=1)
            else:
                end_of_month = today.replace(month=today.month + 1, day=1) - timedelta(days=1)
        else:
            target_date = today - relativedelta(months=abs(months_ago))
            start_of_month = target_date.replace(day=1)
            if target_date.month == 12:
                end_of_month = target_date.replace(year=target_date.year + 1, month=1, day=1) - timedelta(days=1)
            else:
                end_of_month = target_date.replace(month=target_date.month + 1, day=1) - timedelta(days=1)
        return start_of_month, end_of_month
    
    def _parse_last_weekday(self, match):
        """Parse 'last Tuesday' type queries"""
        weekday_name = match.group(1).lower()
        weekdays = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        
        if weekday_name in weekdays:
            target_weekday = weekdays.index(weekday_name)
            today = datetime.now().date()
            days_back = (today.weekday() - target_weekday) % 7
            if days_back == 0:  # If today is the target day, go back 7 days
                days_back = 7
            target_date = today - timedelta(days=days_back)
            return target_date
        return None
    
    def parse_time_query(self, query: str) -> Optional[tuple]:
        """Parse natural language time expressions"""
        query_lower = query.lower()
        
        for pattern, handler in self.time_patterns.items():
            if isinstance(pattern, str):
                if pattern in query_lower:
                    result = handler()
                    if isinstance(result, tuple):
                        return result
                    else:
                        return result, result
            else:  # regex pattern
                match = re.search(pattern, query_lower)
                if match:
                    result = handler(match)
                    if result:
                        return result, result
        
        # Try to parse specific dates
        date_patterns = [
            r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b',
            r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, query)
            if matches:
                try:
                    parsed_date = date_parser.parse(matches[0]).date()
                    return parsed_date, parsed_date
                except:
                    continue
        
        return None
    
    def smart_search(self, query: str, limit: int = 50) -> List[MemoryRecord]:
        """Perform smart search with natural language processing"""
        results = []
        
        # Parse time constraints
        time_range = self.parse_time_query(query)
        
        # Extract category hints
        category = self._extract_category(query)
        
        # Extract keywords (remove time and category expressions)
        keywords = self._extract_keywords(query)
        
        # Build SQL query
        sql_query = """
        SELECT * FROM memories 
        WHERE 1=1
        """
        params = {}
        
        # Add keyword search using FTS if available, otherwise use LIKE
        if keywords:
            try:
                # Try FTS first
                sql_query += " AND memories MATCH :keywords"
                params['keywords'] = ' '.join(keywords)
            except:
                # Fallback to LIKE search
                keyword_conditions = []
                for i, keyword in enumerate(keywords):
                    keyword_conditions.append(f"(content LIKE :keyword_{i} OR title LIKE :keyword_{i})")
                    params[f'keyword_{i}'] = f'%{keyword}%'
                
                if keyword_conditions:
                    sql_query += f" AND ({' OR '.join(keyword_conditions)})"
        
        # Add time constraints
        if time_range:
            start_date, end_date = time_range
            sql_query += " AND DATE(timestamp) BETWEEN :start_date AND :end_date"
            params['start_date'] = start_date.isoformat()
            params['end_date'] = end_date.isoformat()
        
        # Add category filter
        if category:
            sql_query += " AND category = :category"
            params['category'] = category
        
        sql_query += " ORDER BY timestamp DESC LIMIT :limit"
        params['limit'] = limit
        
        # Execute query
        try:
            result = self.db_manager.session.execute(text(sql_query), params)
            records = result.fetchall()
            
            # Convert to MemoryRecord objects
            for record in records:
                memory = MemoryRecord()
                for key, value in record._asdict().items():
                    setattr(memory, key, value)
                results.append(memory)
                
        except Exception as e:
            print(f"Smart search error: {e}")
            # Fallback to basic search
            results = self.db_manager.search_memories(' '.join(keywords) if keywords else query, limit)
        
        return results
    
    def _extract_category(self, query: str) -> Optional[str]:
        """Extract category hints from query"""
        query_lower = query.lower()
        
        category_keywords = {
            'health': ['health', 'medical', 'fitness', 'exercise', 'doctor', 'medicine'],
            'finance': ['money', 'finance', 'budget', 'investment', 'bank', 'expense'],
            'ideas': ['idea', 'thought', 'brainstorm', 'concept', 'innovation'],
            'tasks': ['task', 'todo', 'work', 'assignment', 'deadline'],
            'jobs': ['job', 'freelance', 'client', 'project', 'proposal'],
            'personal': ['personal', 'diary', 'journal', 'private']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        
        return None
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query"""
        # Remove common question words and time expressions
        stop_words = {
            'what', 'when', 'where', 'who', 'why', 'how', 'did', 'i', 'say', 'about',
            'tell', 'me', 'show', 'find', 'search', 'last', 'this', 'week', 'month',
            'day', 'today', 'yesterday', 'ago', 'the', 'a', 'an', 'and', 'or', 'but'
        }
        
        # Split into words and filter
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords

class DatabaseManager:
    """Database manager with FTS support"""
    
    def __init__(self, db_path: str):
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(bind=self.engine)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.session = SessionLocal()
        self.query_engine = SmartQueryEngine(self)
        
        # Setup FTS if possible
        self._setup_fts()
    
    def _setup_fts(self):
        """Setup Full-Text Search virtual table"""
        try:
            # Create FTS virtual table
            fts_sql = """
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content, title, tags, category, content=memories, content_rowid=id
            );
            """
            self.session.execute(text(fts_sql))
            
            # Create triggers to keep FTS in sync
            trigger_insert = """
            CREATE TRIGGER IF NOT EXISTS memories_fts_insert AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content, title, tags, category) 
                VALUES (new.id, new.content, new.title, new.tags, new.category);
            END;
            """
            
            trigger_delete = """
            CREATE TRIGGER IF NOT EXISTS memories_fts_delete AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, title, tags, category) 
                VALUES('delete', old.id, old.content, old.title, old.tags, old.category);
            END;
            """
            
            trigger_update = """
            CREATE TRIGGER IF NOT EXISTS memories_fts_update AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, title, tags, category) 
                VALUES('delete', old.id, old.content, old.title, old.tags, old.category);
                INSERT INTO memories_fts(rowid, content, title, tags, category) 
                VALUES (new.id, new.content, new.title, new.tags, new.category);
            END;
            """
            
            self.session.execute(text(trigger_insert))
            self.session.execute(text(trigger_delete))
            self.session.execute(text(trigger_update))
            self.session.commit()
            
            print("FTS setup completed successfully")
            
        except Exception as e:
            print(f"FTS setup failed, using fallback search: {e}")
    
    def add_memory(self, memory_data: Dict) -> MemoryRecord:
        memory = MemoryRecord(
            content=memory_data.get("content", ""),
            title=memory_data.get("title"),
            tags=json.dumps(memory_data.get("tags", [])),
            category=memory_data.get("category"),
            file_path=memory_data.get("file_path"),
            nas_path=memory_data.get("nas_path"),
            gdrive_path=memory_data.get("gdrive_path"),
            meta_data=json.dumps(memory_data.get("metadata", {})),
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
    
    def search_memories(self, keyword: str, limit: int = 0) -> List[MemoryRecord]:
        return self.session.query(MemoryRecord).filter(
            MemoryRecord.content.contains(keyword)
        ).order_by(MemoryRecord.timestamp.desc()).limit(limit).all()
    
    def smart_query(self, query: str, limit: int = 50) -> List[MemoryRecord]:
        """Use the smart query engine for natural language queries"""
        return self.query_engine.smart_search(query, limit)
    
    def get_memories_by_date_range(self, start_date: datetime, end_date: datetime, 
                                   category: str = None, limit: int = 100) -> List[MemoryRecord]:
        """Get memories within a specific date range"""
        query = self.session.query(MemoryRecord).filter(
            MemoryRecord.timestamp >= start_date,
            MemoryRecord.timestamp <= end_date
        )
        
        if category:
            query = query.filter(MemoryRecord.category == category)
        
        return query.order_by(MemoryRecord.timestamp.desc()).limit(limit).all()