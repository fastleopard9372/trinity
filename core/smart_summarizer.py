import os
import json
import hashlib
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pickle
from openai import OpenAI
from config.settings import settings
from utils.logger import logger

class MemorySummarizer:
    """Smart memory summarization with caching and OpenAI integration"""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        self.cache_dir = "data/summaries"
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _get_cache_key(self, memories: List, query: str) -> str:
        """Generate cache key based on memories and query"""
        content_hash = hashlib.md5()
        
        # Hash memory IDs and timestamps
        for memory in memories:
            content_hash.update(f"{memory.id}:{memory.timestamp}".encode())
        
        # Add query to hash
        content_hash.update(query.encode())
        
        return content_hash.hexdigest()
    
    def _get_cached_summary(self, cache_key: str) -> Optional[Dict]:
        """Get cached summary if available and recent"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Check if cache is still valid (less than 1 hour old)
                cache_time = cached_data.get('timestamp')
                if cache_time and datetime.now() - cache_time < timedelta(hours=1):
                    logger.info("Using cached summary")
                    return cached_data
                
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, summary_data: Dict):
        """Save summary to cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        try:
            summary_data['timestamp'] = datetime.now()
            with open(cache_file, 'wb') as f:
                pickle.dump(summary_data, f)
            logger.info("Summary cached successfully")
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
    
    def _prepare_memory_text(self, memories: List) -> str:
        """Prepare memory content for summarization"""
        memory_texts = []
        
        for memory in memories:
            # Format each memory with metadata
            timestamp = memory.timestamp.strftime("%Y-%m-%d %H:%M") if memory.timestamp else "Unknown"
            category = memory.category or "general"
            tags = json.loads(memory.tags) if memory.tags else []
            
            memory_text = f"""
[{timestamp}] [{category}] {', '.join(tags) if tags else 'no tags'}
Title: {memory.title or 'No title'}
Content: {memory.content[:500]}{'...' if len(memory.content) > 500 else ''}
---
"""
            memory_texts.append(memory_text)
        
        return '\n'.join(memory_texts)
    
    def _create_summary_prompt(self, memories_text: str, query: str, summary_type: str = "general") -> str:
        """Create appropriate prompt for different summary types"""
        
        base_context = f"""
You are analyzing memory entries to answer the following query: "{query}"

Here are the relevant memory entries:
{memories_text}

"""
        
        if summary_type == "general":
            prompt = base_context + """
Please provide a concise summary that:
1. Directly answers the user's question
2. Highlights the most important information
3. Organizes findings by relevance
4. Mentions dates when relevant
5. Keeps the summary under 200 words

Summary:"""
        
        elif summary_type == "timeline":
            prompt = base_context + """
Please create a chronological timeline summary that:
1. Orders events by date
2. Shows progression or changes over time
3. Highlights key developments
4. Mentions specific dates
5. Keeps it under 300 words

Timeline Summary:"""
        
        elif summary_type == "insights":
            prompt = base_context + """
Please provide insights and patterns that:
1. Identify trends or recurring themes
2. Connect related information
3. Highlight important relationships
4. Suggest actionable insights
5. Keep analysis under 250 words

Insights:"""
        
        else:  # categorical
            prompt = base_context + """
Please organize the information by categories and provide:
1. A summary for each category found
2. Key points within each category
3. Cross-category connections if any
4. Most important items per category
5. Keep each category summary under 100 words

Categorical Summary:"""
        
        return prompt
    
    def summarize_memories(self, memories: List, query: str, 
                          summary_type: str = "general", use_cache: bool = True) -> Dict:
        """Summarize memories with optional caching"""
        
        if not memories:
            return {
                "summary": "No memories found matching your query.",
                "memory_count": 0,
                "date_range": None,
                "categories": [],
                "cached": False
            }
        
        # Check cache first
        cache_key = self._get_cache_key(memories, f"{query}:{summary_type}")
        if use_cache:
            cached_result = self._get_cached_summary(cache_key)
            if cached_result:
                return cached_result
        
        # Prepare data for summary
        memory_count = len(memories)
        memories_text = self._prepare_memory_text(memories)
        
        # Extract metadata
        timestamps = [m.timestamp for m in memories if m.timestamp]
        date_range = {
            "start": min(timestamps).isoformat() if timestamps else None,
            "end": max(timestamps).isoformat() if timestamps else None
        }
        
        categories = list(set([m.category for m in memories if m.category]))
        
        # Generate summary
        summary_text = self._generate_ai_summary(memories_text, query, summary_type)
        
        # Prepare result
        result = {
            "summary": summary_text,
            "memory_count": memory_count,
            "date_range": date_range,
            "categories": categories,
            "query": query,
            "summary_type": summary_type,
            "cached": False,
            "generated_at": datetime.now().isoformat()
        }
        
        # Cache the result
        if use_cache:
            self._save_to_cache(cache_key, result)
        
        return result
    
    def _generate_ai_summary(self, memories_text: str, query: str, summary_type: str) -> str:
        """Generate AI summary using OpenAI or fallback to rule-based"""
        
        if self.client:
            try:
                prompt = self._create_summary_prompt(memories_text, query, summary_type)
                
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that summarizes personal memory entries. Be concise, accurate, and focus on what the user is asking for."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.3
                )
                
                summary = response.choices[0].message.content.strip()
                logger.info("AI summary generated successfully")
                return summary
                
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                return self._generate_fallback_summary(memories_text, query)
        
        else:
            logger.info("No OpenAI API key, using fallback summary")
            return self._generate_fallback_summary(memories_text, query)
    
    def _generate_fallback_summary(self, memories_text: str, query: str) -> str:
        """Generate rule-based summary when AI is not available"""
        
        lines = memories_text.split('\n')
        memory_blocks = []
        current_block = []
        
        for line in lines:
            if line.strip() == '---':
                if current_block:
                    memory_blocks.append('\n'.join(current_block))
                    current_block = []
            else:
                current_block.append(line)
        
        if current_block:
            memory_blocks.append('\n'.join(current_block))
        
        # Extract key information
        summary_parts = []
        summary_parts.append(f"Found {len(memory_blocks)} relevant memories for your query: '{query}'")
        
        # Get date range
        dates = []
        for block in memory_blocks:
            if '[' in block and ']' in block:
                date_part = block.split('[')[1].split(']')[0]
                dates.append(date_part)
        
        if dates:
            summary_parts.append(f"Date range: {min(dates)} to {max(dates)}")
        
        # Extract categories
        categories = []
        for block in memory_blocks:
            if '][' in block:
                try:
                    category = block.split('][')[1].split(']')[0]
                    categories.append(category)
                except:
                    pass
        
        if categories:
            unique_categories = list(set(categories))
            summary_parts.append(f"Categories: {', '.join(unique_categories)}")
        
        # Add key content snippets
        content_snippets = []
        for i, block in enumerate(memory_blocks[:3]):  # First 3 memories
            lines = block.split('\n')
            for line in lines:
                if line.startswith('Content:'):
                    content = line.replace('Content:', '').strip()
                    content_snippets.append(f"{i+1}. {content[:100]}...")
                    break
        
        if content_snippets:
            summary_parts.append("Key content:")
            summary_parts.extend(content_snippets)
        
        return '\n'.join(summary_parts)
    
    def get_weekly_summary(self, category: str = None) -> Dict:
        """Get summary of this week's memories"""
        from models.memory import DatabaseManager
        
        db_manager = DatabaseManager(settings.memory_db_path)
        
        # Get this week's date range
        today = datetime.now()
        start_of_week = today - timedelta(days=today.weekday())
        end_of_week = start_of_week + timedelta(days=6)
        
        memories = db_manager.get_memories_by_date_range(
            start_of_week, end_of_week, category=category
        )
        
        query = f"What happened this week" + (f" in {category}" if category else "")
        return self.summarize_memories(memories, query, "timeline")
    
    def get_category_insights(self, category: str, days: int = 30) -> Dict:
        """Get insights for a specific category over time"""
        from models.memory import DatabaseManager
        
        db_manager = DatabaseManager(settings.memory_db_path)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        memories = db_manager.get_memories_by_date_range(
            start_date, end_date, category=category
        )
        
        query = f"What insights can you provide about my {category} over the last {days} days?"
        return self.summarize_memories(memories, query, "insights")