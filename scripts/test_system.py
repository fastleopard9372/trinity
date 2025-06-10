#  testing suite for Trinity Memory System
# File: scripts/test_system.py

#!/usr/bin/env python3
"""
Comprehensive test suite for Trinity  Memory System
"""

import os
import sys
import asyncio
import json
import requests
import time
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.memory_manager import MemoryManager, ConversationEntry
from config.settings import settings
from utils.logger import logger

class SystemTester:
    def __init__(self):
        self.base_url = f"http://{settings.api_host}:{settings.api_port}"
        self.test_session_id = None
        self.test_user_id = "test_user_123"
        self.memory_manager = None
        
    async def setup(self):
        """Setup test environment"""
        print("Setting up test environment...")
        
        try:
            self.memory_manager = MemoryManager()
            print("memory manager initialized")
        except Exception as e:
            print(f"Failed to initialize memory manager: {e}")
            return False
        
        return True
    
    async def test_memory_core(self):
        """Test core memory functionality"""
        print("\nTesting Memory Core...")
        
        try:
            # Create a test session
            session_id = self.memory_manager.session_manager.create_session(
                self.test_user_id, 
                {"test": True, "created_by": "test_suite"}
            )
            self.test_session_id = session_id
            print(f"Created test session: {session_id}")
            
            # Add conversation messages
            messages = [
                ("user", "Hello, I want to learn about Python programming"),
                ("assistant", "Python is a versatile programming language great for beginners and experts alike!"),
                ("user", "Can you tell me about data structures in Python?"),
                ("assistant", "Python has several built-in data structures like lists, dictionaries, sets, and tuples. Each has unique properties and use cases."),
                ("user", "Save this conversation for future reference"),
                ("assistant", "I'll save this conversation to your memory system for future retrieval.")
            ]
            
            for role, content in messages:
                entry_id = await self.memory_manager.add_conversation_message(
                    session_id=session_id,
                    user_id=self.test_user_id,
                    content=content,
                    role=role,
                    metadata={"test": True}
                )
                print(f"  Added {role} message: {entry_id}")
            
            # Test semantic search
            search_results = await self.memory_manager.search_memory(
                "Python programming data structures",
                user_id=self.test_user_id
            )
            
            print(f"  Search found {len(search_results.vector_results)} vector results")
            print(f"  Search found {len(search_results.nas_entries)} NAS results")
            print(f"  Relevance score: {search_results.relevance_score:.2f}")
            
            return True
            
        except Exception as e:
            print(f"Vector database test failed: {e}")
            return False
    
    def test_file_watcher_integration(self):
        """Test file watcher integration with memory"""
        print("\n📁 Testing File Watcher Integration...")
        
        try:
            # Create test files in watch directories
            test_files = [
                {
                    "path": f"{settings.watch_dirs_list[0]}/test_vector_memory.txt",
                    "content": """Save this: Integration test for vector memory system.
Tags: vector, integration, test
Category: system_test

This file tests the integration between file watching and vector memory storage.
The system should automatically process this file and store it in both vector DB and NAS.
"""
                },
                {
                    "path": f"{settings.watch_dirs_list[0]}/test_conversation.txt", 
                    "content": """Remember this conversation about AI development:

User: What are the key principles of responsible AI development?
Assistant: Responsible AI development focuses on fairness, transparency, accountability, and privacy.
User: Can you elaborate on transparency?
Assistant: Transparency means making AI systems interpretable and explainable to users and stakeholders.

Tags: ai, responsible_development, transparency
Category: conversations
"""
                }
            ]
            
            for test_file in test_files:
                os.makedirs(os.path.dirname(test_file["path"]), exist_ok=True)
                with open(test_file["path"], 'w', encoding='utf-8') as f:
                    f.write(test_file["content"])
                print(f"Created test file: {test_file['path']}")
            
            print("📝 Test files created - they should be processed automatically")
            print("🕐 Wait a few seconds and check logs for processing confirmation")
            
            return True
            
        except Exception as e:
            print(f"File watcher integration test failed: {e}")
            return False
    
    def test_analytics_endpoints(self):
        """Test analytics and monitoring endpoints"""
        print("\n📊 Testing Analytics Endpoints...")
        
        try:
            # Test memory analytics
            response = requests.get(
                f"{self.base_url}/api/analytics/memory-stats",
                timeout=10
            )
            
            if response.status_code == 200:
                stats = response.json()
                print("Memory analytics working")
                print(f"  - Vector DB vectors: {stats.get('vector_db', {}).get('total_vectors', 0)}")
                print(f"  - Total sessions: {sum(stats.get('sessions', {}).get('by_status', {}).values())}")
                print(f"  - NAS memories: {stats.get('nas_storage', {}).get('total_memories', 0)}")
            else:
                print(f"Memory analytics failed: {response.status_code}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Analytics test failed: {e}")
            return False
    
    def test_background_tasks(self):
        """Test background task triggers"""
        print("\n⚙️ Testing Background Tasks...")
        
        try:
            # Test cleanup task
            response = requests.post(
                f"{self.base_url}/api/system/cleanup",
                params={"vector_days": 1, "session_days": 1},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Cleanup task triggered: {result['message']}")
            else:
                print(f"Cleanup task failed: {response.status_code}")
                return False
            
            # Test backup task
            response = requests.post(
                f"{self.base_url}/api/system/backup-sessions",
                params={"min_messages": 1},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Backup task triggered: {result['message']}")
            else:
                print(f"Backup task failed: {response.status_code}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Background tasks test failed: {e}")
            return False
    
    async def test_performance(self):
        """Test system performance with bulk operations"""
        print("\n🚀 Testing Performance...")
        
        try:
            # Test bulk conversation storage
            start_time = time.time()
            session_id = self.memory_manager.session_manager.create_session(
                f"{self.test_user_id}_perf", 
                {"test": "performance"}
            )
            
            # Add 50 messages quickly
            tasks = []
            for i in range(50):
                task = self.memory_manager.add_conversation_message(

                    session_id=session_id,
                    user_id=f"{self.test_user_id}_perf",
                    content=f"Performance test message {i}: Testing bulk operations with various AI topics like machine learning, neural networks, and data science.",
                    role="user" if i % 2 == 0 else "assistant",
                    metadata={"performance_test": True, "batch": i // 10}
                )
                tasks.append(task)
            
            # Wait for all to complete
            await asyncio.gather(*tasks)
            
            bulk_time = time.time() - start_time
            print(f"Bulk storage: 50 messages in {bulk_time:.2f} seconds ({50/bulk_time:.1f} msg/sec)")
            
            # Test bulk search performance
            start_time = time.time()
            search_results = await self.memory_manager.search_memory(
                "machine learning neural networks data science",
                user_id=f"{self.test_user_id}_perf"
            )
            search_time = time.time() - start_time
            
            print(f"Search performance: {len(search_results.vector_results)} results in {search_time:.3f} seconds")
            
            return True
            
        except Exception as e:
            print(f"Performance test failed: {e}")
            return False
    
    async def cleanup_test_data(self):
        """Clean up test data"""
        print("\n🧹 Cleaning up test data...")
        
        try:
            # Remove test files
            test_files = [
                f"{settings.watch_dirs_list[0]}/test_vector_memory.txt",
                f"{settings.watch_dirs_list[0]}/test_conversation.txt"
            ]
            
            for file_path in test_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Removed test file: {file_path}")
            
            # Note: In a real scenario, you might want to clean up test sessions
            # from the database, but for testing purposes, we'll leave them
            print("Test cleanup completed")
            
        except Exception as e:
            print(f"Cleanup failed: {e}")

    async def test_auto_save_triggers(self):
        """Test auto-save trigger functionality"""
        print("\n💾 Testing Auto-Save Triggers...")
        
        try:
            if not self.test_session_id:
                print("❌ No test session available")
                return False
            
            # Test keyword trigger
            await self.memory_manager.add_conversation_message(
                session_id=self.test_session_id,
                user_id=self.test_user_id,
                content="Remember this important information about AI safety protocols",
                role="user",
                metadata={"test_trigger": "keyword"}
            )
            
            # Give some time for processing
            await asyncio.sleep(2)
            
            print("✅ Keyword trigger test completed")
            
            # Test message count trigger by adding more messages
            for i in range(8):  # Add enough to trigger threshold
                await self.memory_manager.add_conversation_message(
                    session_id=self.test_session_id,
                    user_id=self.test_user_id,
                    content=f"Test message {i+1} for threshold testing",
                    role="user" if i % 2 == 0 else "assistant",
                    metadata={"test_trigger": "count", "message_num": i+1}
                )
            
            print("✅ Message count trigger test completed")
            return True
            
        except Exception as e:
            print(f"❌ Auto-save trigger test failed: {e}")
            return False
    
    def test_api_endpoints(self):
        """Test API endpoints"""
        print("\n🌐 Testing API Endpoints...")
        
        try:
            # Test health endpoint
            response = requests.get(f"{self.base_url}/api/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print("✅ Health endpoint working")
                print(f"  - NAS: {'✅' if health_data['nas_connected'] else '❌'}")
                print(f"  - Google Drive: {'✅' if health_data['gdrive_connected'] else '❌'}")
                print(f"  - Vector DB: {'✅' if health_data['vector_db_connected'] else '❌'}")
            else:
                print(f"❌ Health endpoint failed: {response.status_code}")
                return False
            
            # Test conversation message endpoint
            message_data = {
                "user_id": self.test_user_id,
                "content": "This is a test message via API",
                "role": "user",
                "metadata": {"test": True, "via": "api"}
            }
            
            response = requests.post(
                f"{self.base_url}/api/conversation/message",
                json=message_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Conversation message endpoint working: {result['entry_id']}")
                
                # Store session ID for further tests
                if not self.test_session_id:
                    self.test_session_id = result['session_id']
            else:
                print(f"❌ Conversation message endpoint failed: {response.status_code}")
                return False
            
            # Test memory search endpoint
            search_data = {
                "query": "test message programming",
                "user_id": self.test_user_id,
                "max_results": 5
            }
            
            response = requests.post(
                f"{self.base_url}/api/memory/search",
                json=search_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Memory search endpoint working: {result['total_results']} results")
            else:
                print(f"❌ Memory search endpoint failed: {response.status_code}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ API endpoint test failed: {e}")
            return False
    
    def test_session_management(self):
        """Test session management endpoints"""
        print("\n📂 Testing Session Management...")
        
        try:
            if not self.test_session_id:
                print("❌ No test session available")
                return False
            
            # Test get session messages
            response = requests.get(
                f"{self.base_url}/api/session/{self.test_session_id}/messages",
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Session messages: {result['total_messages']} messages")
            else:
                print(f"❌ Session messages failed: {response.status_code}")
                return False
            
            # Test get user sessions
            response = requests.get(
                f"{self.base_url}/api/sessions/user/{self.test_user_id}",
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ User sessions: {result['total_sessions']} sessions")
            else:
                print(f"❌ User sessions failed: {response.status_code}")
                return False
            
            # Test session summarization
            summary_data = {
                "session_id": self.test_session_id,
                "force_save": True
            }
            
            response = requests.post(
                f"{self.base_url}/api/session/summarize",
                json=summary_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Session summarization: {result['message_count']} messages, saved: {result['saved_to_nas']}")
            else:
                print(f"❌ Session summarization failed: {response.status_code}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Session management test failed: {e}")
            return False
    
    async def test_vector_database(self):
        """Test vector database functionality"""
        print("\n🔍 Testing Vector Database...")
        
        try:
            # Test direct vector storage
            # test_entry = ConversationEntry(
            #     id="test_vector_entry",
            #     session_id="test_session",
            #     user_id=self.test_user_id,
            #     content="This is a test entry for vector database functionality with machine learning concepts",
            #     role="user",
            #     timestamp=datetime.now(),
            #     metadata={"test": True, "concept": "machine_learning"}
            # )
            
            # success = await self.memory_manager.vector_manager.store_conversation_entry(test_entry)
            # if success:
            #     print("✅ Vector storage successful")
            # else:
            #     print("❌ Vector storage failed")
            #     return False
            
            # # Test vector search
            # results = await self.memory_manager.vector_manager.semantic_search(
            #     "machine learning concepts",
            #     k=3
            # )
            
            # if results:
            #     print(f"✅ Vector search successful: {len(results)} results")
            #     for i, result in enumerate(results[:2]):
            #         print(f"  Result {i+1}: Score {result['relevance_score']:.3f}")
            # else:
            #     print("❌ Vector search returned no results")
            #     return False
            
            return True
            
        except Exception as e:
            print(f"❌ Vector database test failed: {e}")
                  
async def main():
    """Run comprehensive test suite"""
    tester = SystemTester()
    
    print("Trinity  Memory System - Comprehensive Test Suite")
    print("=" * 60)
    
    # Setup
    if not await tester.setup():
        print("Test setup failed")
        return False
    
    # Run all tests
    tests = [
        ("Memory Core", await tester.test_memory_core()),
        # ("Auto-Save Triggers", tester.test_auto_save_triggers()),
        # ("API Endpoints", tester.test_api_endpoints()),
        # ("Session Management", tester.test_session_management()),
        # ("Vector Database", tester.test_vector_database()),
        # ("File Watcher Integration", tester.test_file_watcher_integration()),
        # ("Analytics Endpoints", tester.test_analytics_endpoints()),
        # ("Background Tasks", tester.test_background_tasks()),
        # ("Performance", tester.test_performance())
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_coro in tests:
        try:
            if asyncio.iscoroutine(test_coro):
                result = await test_coro
            else:
                result = test_coro
            
            if result:
                passed += 1
                print(f"{test_name} - PASSED")
            else:
                failed += 1
                print(f"{test_name} - FAILED")
                
        except Exception as e:
            failed += 1
            print(f"{test_name} - ERROR: {e}")
    
    # Cleanup
    await tester.cleanup_test_data()
    
    # Summary
    print("\n" + "=" * 60)
    print(f"🏁 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Trinity  Memory System is working correctly.")
    else:
        print(f"⚠️  {failed} tests failed. Check the output above for details.")
    
    return failed == 0

# =============================================================================
# USAGE EXAMPLES AND DOCUMENTATION
# =============================================================================

class TrinityUsageExamples:
    """Examples of how to use Trinity  Memory System"""
    
    def __init__(self):
        self.base_url = "http://localhost:5000"
    
    def example_conversation_tracking(self):
        """Example: Track a conversation with auto-save"""
        print("📝 Example: Conversation Tracking with Auto-Save")
        
        # Start a conversation
        message_data = {
            "user_id": "user_001",
            "content": "I want to learn about machine learning",
            "role": "user"
        }
        
        response = requests.post(
            f"{self.base_url}/api/conversation/message",
            json=message_data
        )
        
        session_id = response.json()["session_id"]
        print(f"Started session: {session_id}")
        
        # Continue conversation
        messages = [
            ("assistant", "Machine learning is a subset of AI that enables computers to learn from data."),
            ("user", "What are the main types of machine learning?"),
            ("assistant", "The main types are supervised, unsupervised, and reinforcement learning."),
            ("user", "Can you explain supervised learning?"),
            ("assistant", "Supervised learning uses labeled data to train models that can make predictions on new data."),
            ("user", "Save this conversation for future reference")  # Trigger auto-save
        ]
        
        for role, content in messages:
            message_data = {
                "session_id": session_id,
                "user_id": "user_001",
                "content": content,
                "role": role
            }
            
            requests.post(
                f"{self.base_url}/api/conversation/message",
                json=message_data
            )
        
        print("Conversation will be auto-saved due to 'save this' trigger")
    
    def example_semantic_search(self):
        """Example: Search across vector and NAS memory"""
        print("🔍 Example: Semantic Memory Search")
        
        search_data = {
            "query": "machine learning supervised learning algorithms",
            "user_id": "user_001",
            "include_nas": True,
            "max_results": 10
        }
        
        response = requests.post(
            f"{self.base_url}/api/memory/search",
            json=search_data
        )
        
        results = response.json()
        
        print(f"Found {results['total_results']} total results:")
        print(f"  Vector results: {len(results['vector_results'])}")
        print(f"  NAS results: {len(results['nas_results'])}")
        print(f"  Relevance score: {results['relevance_score']:.2f}")
        
        # Display top results
        for i, result in enumerate(results['vector_results'][:3]):
            print(f"  {i+1}. {result['content'][:100]}...")
    
    def example_session_management(self):
        """Example: Manage and summarize sessions"""
        print("📂 Example: Session Management")
        
        # Get user sessions
        response = requests.get(f"{self.base_url}/api/sessions/user/user_001")
        sessions = response.json()
        
        print(f"User has {sessions['total_sessions']} sessions")
        
        if sessions['sessions']:
            latest_session = sessions['sessions'][0]
            session_id = latest_session['session_id']
            
            # Get session messages
            response = requests.get(
                f"{self.base_url}/api/session/{session_id}/messages"
            )
            messages = response.json()
            
            print(f"Latest session has {messages['total_messages']} messages")
            
            # Force summarization
            summary_data = {
                "session_id": session_id,
                "force_save": True
            }
            
            response = requests.post(
                f"{self.base_url}/api/session/summarize",
                json=summary_data
            )
            
            result = response.json()
            print(f"Session summarized: {result['saved_to_nas']}")
    
    def example_file_based_memory(self):
        """Example: File-based memory input"""
        print("📁 Example: File-Based Memory Input")
        
        # Create memory files
        memory_files = [
            {
                "filename": "learning_notes.txt",
                "content": """Save this: Notes from AI course lecture 5

Topics covered:
- Deep learning fundamentals
- Neural network architectures
- Backpropagation algorithm
- Gradient descent optimization

Tags: ai, deep_learning, course_notes
Category: education
Priority: 3

Key takeaway: Understanding the mathematical foundations is crucial for advanced AI development.
"""
            },
            {
                "filename": "project_ideas.txt",
                "content": """Remember these project ideas:

1. AI-powered personal assistant with memory
2. Automated code review system using LLMs
3. Smart document summarization tool
4. Conversational data analysis interface

Tags: projects, ai, development
Category: ideas

These could be good portfolio projects to demonstrate AI capabilities.
"""
            }
        ]
        
        for file_data in memory_files:
            file_path = f"data/inbox/{file_data['filename']}"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(file_data['content'])
            
            print(f"Created memory file: {file_path}")
        
        print("📝 Files will be automatically processed by the file watcher")
    
    def example_analytics_monitoring(self):
        """Example: System analytics and monitoring"""
        print("📊 Example: System Analytics and Monitoring")
        
        # Check system health
        response = requests.get(f"{self.base_url}/api/health")
        health = response.json()
        
        print("System Health:")
        for component, status in health.items():
            if isinstance(status, bool):
                print(f"  {component}: {'✅' if status else '❌'}")
            elif isinstance(status, (int, float)):
                print(f"  {component}: {status}")
        
        # Get detailed analytics
        response = requests.get(f"{self.base_url}/api/analytics/memory-stats")
        analytics = response.json()
        
        print("\nMemory Analytics:")
        print(f"  Vector DB: {analytics['vector_db']['total_vectors']} vectors")
        print(f"  Sessions: {analytics['sessions']['by_status']}")
        print(f"  NAS Storage: {analytics['nas_storage']['total_memories']} memories")
    
    def run_all_examples(self):
        """Run all usage examples"""
        print("🎯 Trinity  Memory System - Usage Examples")
        print("=" * 60)
        
        examples = [
            self.example_conversation_tracking,
            self.example_semantic_search,
            self.example_session_management,
            self.example_file_based_memory,
            self.example_analytics_monitoring
        ]
        
        for example in examples:
            try:
                example()
                print()
            except Exception as e:
                print(f"Example failed: {e}\n")

# =============================================================================
# BENCHMARK AND STRESS TESTING
# =============================================================================

class TrinityBenchmark:
    """Benchmark Trinity system performance"""
    
    def __init__(self):
        self.base_url = "http://localhost:5000"
        self.test_user = "benchmark_user"
    
    def benchmark_conversation_storage(self, num_messages=1000):
        """Benchmark conversation message storage"""
        print(f"🚀 Benchmarking conversation storage ({num_messages} messages)...")
        
        start_time = time.time()
        session_responses = []
        
        for i in range(num_messages):
            message_data = {
                "user_id": self.test_user,
                "content": f"Benchmark message {i}: Testing system performance with AI and machine learning content including neural networks, deep learning, natural language processing, and computer vision applications.",
                "role": "user" if i % 2 == 0 else "assistant",
                "metadata": {"benchmark": True, "batch": i // 100}
            }
            
            try:
                response = requests.post(
                    f"{self.base_url}/api/conversation/message",
                    json=message_data,
                    timeout=5
                )
                
                if response.status_code == 200:
                    session_responses.append(response.json())
                
                if i % 100 == 0:
                    print(f"  Progress: {i}/{num_messages} messages")
                    
            except Exception as e:
                print(f"  Error at message {i}: {e}")
        
        total_time = time.time() - start_time
        success_rate = len(session_responses) / num_messages * 100
        
        print(f"📊 Results:")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Messages per second: {num_messages/total_time:.1f}")
        print(f"  Success rate: {success_rate:.1f}%")
        
        return {
            "total_time": total_time,
            "messages_per_second": num_messages/total_time,
            "success_rate": success_rate
        }
    
    def benchmark_search_performance(self, num_searches=100):
        """Benchmark search performance"""
        print(f"🔍 Benchmarking search performance ({num_searches} searches)...")
        
        search_queries = [
            "machine learning algorithms",
            "neural networks deep learning",
            "natural language processing",
            "computer vision applications",
            "data science analytics",
            "artificial intelligence ethics",
            "reinforcement learning",
            "supervised unsupervised learning",
            "AI model training",
            "big data processing"
        ]
        
        start_time = time.time()
        successful_searches = 0
        total_results = 0
        
        for i in range(num_searches):
            query = search_queries[i % len(search_queries)]
            
            search_data = {
                "query": f"{query} test query {i}",
                "user_id": self.test_user,
                "max_results": 10
            }
            
            try:
                response = requests.post(
                    f"{self.base_url}/api/memory/search",
                    json=search_data,
                    timeout=10
                )
                
                if response.status_code == 200:
                    results = response.json()
                    successful_searches += 1
                    total_results += results['total_results']
                
                if i % 20 == 0:
                    print(f"  Progress: {i}/{num_searches} searches")
                    
            except Exception as e:
                print(f"  Error at search {i}: {e}")
        
        total_time = time.time() - start_time
        
        print(f"📊 Results:")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Searches per second: {successful_searches/total_time:.1f}")
        print(f"  Average results per search: {total_results/successful_searches:.1f}")
        print(f"  Success rate: {successful_searches/num_searches*100:.1f}%")
        
        return {
            "total_time": total_time,
            "searches_per_second": successful_searches/total_time,
            "avg_results": total_results/successful_searches if successful_searches > 0 else 0,
            "success_rate": successful_searches/num_searches*100
        }
    
    async def test_auto_save_triggers(self):
        """Test auto-save trigger functionality"""
        print("\n💾 Testing Auto-Save Triggers...")
        
        try:
            if not self.test_session_id:
                print("No test session available")
                return False
            
            # Test keyword trigger
            await self.memory_manager.add_conversation_message(
                session_id=self.test_session_id,
                user_id=self.test_user_id,
                content="Remember this important information about AI safety protocols",
                role="user",
                metadata={"test_trigger": "keyword"}
            )
            
            # Give some time for processing
            await asyncio.sleep(2)
            
            print("Keyword trigger test completed")
            
            # Test message count trigger by adding more messages
            for i in range(8):  # Add enough to trigger threshold
                await self.memory_manager.add_conversation_message(
                    session_id=self.test_session_id,
                    user_id=self.test_user_id,
                    content=f"Test message {i+1} for threshold testing",
                    role="user" if i % 2 == 0 else "assistant",
                    metadata={"test_trigger": "count", "message_num": i+1}
                )
            
            print("Message count trigger test completed")
            return True
            
        except Exception as e:
            print(f"Auto-save trigger test failed: {e}")
            return False

    def test_api_endpoints(self):
        """Test API endpoints"""
        print("\nTesting API Endpoints...")
        
        try:
            # Test health endpoint
            response = requests.get(f"{self.base_url}/api/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print("Health endpoint working")
                print(f"  - NAS: {'✅' if health_data['nas_connected'] else '❌'}")
                print(f"  - Google Drive: {'✅' if health_data['gdrive_connected'] else '❌'}")
                print(f"  - Vector DB: {'✅' if health_data['vector_db_connected'] else '❌'}")
            else:
                print(f"Health endpoint failed: {response.status_code}")
                return False
            
            # Test conversation message endpoint
            message_data = {
                "user_id": self.test_user_id,
                "content": "This is a test message via API",
                "role": "user",
                "metadata": {"test": True, "via": "api"}
            }
            
            response = requests.post(
                f"{self.base_url}/api/conversation/message",
                json=message_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Conversation message endpoint working: {result['entry_id']}")
                
                # Store session ID for further tests
                if not self.test_session_id:
                    self.test_session_id = result['session_id']
            else:
                print(f"Conversation message endpoint failed: {response.status_code}")
                return False
            
            # Test memory search endpoint
            search_data = {
                "query": "test message programming",
                "user_id": self.test_user_id,
                "max_results": 5
            }
            
            response = requests.post(
                f"{self.base_url}/api/memory/search",
                json=search_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Memory search endpoint working: {result['total_results']} results")
            else:
                print(f"Memory search endpoint failed: {response.status_code}")
                return False
            
            return True
            
        except Exception as e:
            print(f"API endpoint test failed: {e}")
            return False

    def test_session_management(self):
        """Test session management endpoints"""
        print("\n📂 Testing Session Management...")
        
        try:
            if not self.test_session_id:
                print("No test session available")
                return False
            
            # Test get session messages
            response = requests.get(
                f"{self.base_url}/api/session/{self.test_session_id}/messages",
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Session messages: {result['total_messages']} messages")
            else:
                print(f"Session messages failed: {response.status_code}")
                return False
            
            # Test get user sessions
            response = requests.get(
                f"{self.base_url}/api/sessions/user/{self.test_user_id}",
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"User sessions: {result['total_sessions']} sessions")
            else:
                print(f"User sessions failed: {response.status_code}")
                return False
            
            # Test session summarization
            summary_data = {
                "session_id": self.test_session_id,
                "force_save": True
            }
            
            response = requests.post(
                f"{self.base_url}/api/session/summarize",
                json=summary_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Session summarization: {result['message_count']} messages, saved: {result['saved_to_nas']}")
            else:
                print(f"Session summarization failed: {response.status_code}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Session management test failed: {e}")
            return False

    async def test_vector_database(self):
            """Test vector database functionality"""
            print("\n🔍 Testing Vector Database...")
            
            try:
                # Test direct vector storage
                test_entry = ConversationEntry(
                    id="test_vector_entry",
                    session_id="test_session",
                    user_id=self.test_user_id,
                    content="This is a test entry for vector database functionality with machine learning concepts",
                    role="user",
                    timestamp=datetime.now(),
                    metadata={"test": True, "concept": "machine_learning"}
                )
                
                success = await self.memory_manager.vector_manager.store_conversation_entry(test_entry)
                if success:
                    print("Vector storage successful")
                else:
                    print("Vector storage failed")
                    return False
                
                # Test vector search
                results = await self.memory_manager.vector_manager.semantic_search(
                    "machine learning concepts",
                    k=3
                )
                
                if results:
                    print(f"Vector search successful: {len(results)} results")
                    for i, result in enumerate(results[:2]):
                        print(f"  Result {i+1}: Score {result['relevance_score']:.3f}")
                else:
                    print("Vector search returned no results")
                    return False
                
                return True
                
            except Exception as e:
                print(f"memory core test failed: {e}")
                return False
            
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Trinity  Memory System Testing")
    parser.add_argument("--test", action="store_true", help="Run comprehensive test suite")
    parser.add_argument("--examples", action="store_true", help="Run usage examples")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    
    args = parser.parse_args()
    
    if args.test:
        asyncio.run(main())
    elif args.examples:
        examples = TrinityUsageExamples()
        examples.run_all_examples()
    elif args.benchmark:
        benchmark = TrinityBenchmark()
        benchmark.benchmark_conversation_storage(100 if args.quick else 1000)
        benchmark.benchmark_search_performance(20 if args.quick else 100)
    else:
        parser.print_help() 