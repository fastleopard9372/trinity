import re
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from utils.logger import logger

@dataclass
class MemoryCommand:
    action: str  # save, log, tag, query, delete
    content: str
    tags: List[str]
    category: str
    priority: int = 1
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class CommandParser:
    def __init__(self):
        self.command_patterns = {
            'save': [
                r'save this',
                r'remember this',
                r'store this',
                r'keep this',
                r'log this'
            ],
            'tag': [
                r'tag this as (.+)',
                r'categorize as (.+)',
                r'category: (.+)',
                r'tags?: (.+)'
            ],
            'category': [
                r'category: (.+)',
                r'cat: (.+)',
                r'type: (.+)'
            ]
        }
        
        self.category_mapping = {
            'health': ['health', 'medical', 'fitness', 'wellness', 'exercise'],
            'finance': ['money', 'finance', 'budget', 'investment', 'bank'],
            'ideas': ['idea', 'thought', 'brainstorm', 'concept', 'innovation'],
            'tasks': ['task', 'todo', 'work', 'job', 'assignment'],
            'jobs': ['freelance', 'project', 'client', 'proposal', 'contract'],
            'personal': ['personal', 'private', 'diary', 'journal']
        }
    
    def parse(self, text: str) -> MemoryCommand:
        """Parse text and extract memory command"""
        text_lower = text.lower().strip()
        
        # Detect action
        action = self._detect_action(text_lower)
        
        # Extract tags
        tags = self._extract_tags(text_lower)
        
        # Determine category
        category = self._determine_category(text_lower, tags)
        
        # Extract priority
        priority = self._extract_priority(text_lower)
        
        # Clean content (remove command indicators)
        content = self._clean_content(text)
        
        # Extract metadata
        metadata = self._extract_metadata(text_lower)
        
        command = MemoryCommand(
            action=action,
            content=content,
            tags=tags,
            category=category,
            priority=priority,
            metadata=metadata
        )
        
        logger.info(f"Parsed command: {action} - {category} - {len(tags)} tags")
        return command
    
    def _detect_action(self, text: str) -> str:
        """Detect the primary action from text"""
        for action, patterns in self.command_patterns.items():
            if action == 'tag' or action == 'category':
                continue
            
            for pattern in patterns:
                if re.search(pattern, text):
                    return action
        
        return 'save'  # default action
    
    def _extract_tags(self, text: str) -> List[str]:
        """Extract tags from text"""
        tags = []
        
        # Look for explicit tag indicators
        tag_patterns = [
            r'tags?: (.+?)(?:\n|$)',
            r'tag this as (.+?)(?:\n|$)',
            r'#(\w+)',
            r'@(\w+)'
        ]
        
        for pattern in tag_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, str):
                    # Split on common separators
                    tag_parts = re.split(r'[,;|]', match)
                    tags.extend([tag.strip() for tag in tag_parts if tag.strip()])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tags = []
        for tag in tags:
            if tag.lower() not in seen:
                seen.add(tag.lower())
                unique_tags.append(tag)
        
        return unique_tags
    
    def _determine_category(self, text: str, tags: List[str]) -> str:
        """Determine category based on content and tags"""
        # Check explicit category commands
        category_patterns = self.command_patterns.get('category', [])
        for pattern in category_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        # Check tags for category hints
        for tag in tags:
            for category, keywords in self.category_mapping.items():
                if tag.lower() in keywords:
                    return category
        
        # Check content for category keywords
        for category, keywords in self.category_mapping.items():
            for keyword in keywords:
                if keyword in text:
                    return category
        
        return 'general'  # default category
    
    def _extract_priority(self, text: str) -> int:
        """Extract priority level (1-5, default 1)"""
        priority_patterns = [
            r'priority: (\d+)',
            r'pri: (\d+)',
            r'p(\d+)',
            r'urgent' # maps to priority 5
        ]
        
        for pattern in priority_patterns:
            match = re.search(pattern, text)
            if match:
                if pattern == r'urgent':
                    return 5
                else:
                    priority = int(match.group(1))
                    return max(1, min(5, priority))  # clamp between 1-5
        
        return 1  # default priority
    
    def _clean_content(self, original_text: str) -> str:
        """Remove command indicators from content"""
        text = original_text
        
        # Remove common command prefixes
        prefixes_to_remove = [
            r'^save this:?\s*',
            r'^remember this:?\s*',
            r'^store this:?\s*',
            r'^log this:?\s*',
            r'^keep this:?\s*'
        ]
        
        for prefix in prefixes_to_remove:
            text = re.sub(prefix, '', text, flags=re.IGNORECASE)
        
        # Remove tag and category lines
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line_lower = line.lower().strip()
            if not (
                line_lower.startswith('tags:') or
                line_lower.startswith('tag:') or
                line_lower.startswith('category:') or
                line_lower.startswith('cat:') or
                line_lower.startswith('priority:') or
                line_lower.startswith('pri:')
            ):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def _extract_metadata(self, text: str) -> Dict:
        """Extract additional metadata from text"""
        metadata = {
            'parsed_at': datetime.utcnow().isoformat(),
            'original_length': len(text),
            'word_count': len(text.split())
        }
        
        # Extract URLs
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, text)
        if urls:
            metadata['urls'] = urls
        
        # Extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            metadata['emails'] = emails
        
        # Extract dates (simple pattern)
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        dates = re.findall(date_pattern, text)
        if dates:
            metadata['mentioned_dates'] = dates
        
        return metadata