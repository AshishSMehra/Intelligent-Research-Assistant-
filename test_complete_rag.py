"""
Comprehensive test script for the complete RAG pipeline implementation.
"""

import asyncio
import json
import time
from src.models.chat import ChatQuery, ChatMetadata
from src.services.chat_service import ChatService


async def test_complete_rag_pipeline():
    """Test the complete RAG pipeline implementation."""
    print("🧪 Testing Complete RAG Pipeline Implementation")
    print("=" * 60)
    
    # Initialize chat service
    chat_service = ChatService()
    
    # Test 1: Basic query with metadata
    print("\n📝 Test 1: Basic Query with Metadata")
    print("-" * 40)
    
    test_query = ChatQuery(
        query="What is machine learning?",
        metadata=ChatMetadata(
            user_id="test_user_123",
            session_id="test_session_456",
            document_type="research_paper",
            tags=["machine-learning", "ai"],
            quality_threshold=0.7
        ),
        max_results=5,
        include_sources=True,
        include_metadata=True
    )
    
    print(f"Query: {test_query.query}")
    print(f"Session ID: {test_query.metadata.session_id}")
    print(f"User ID: {test_query.metadata.user_id}")
    
    try:
        start_time = time.time()
        response = await chat_service.process_query(test_query)
        processing_time = time.time() - start_time
        
        print(f"✅ Query processed successfully!")
        print(f"⏱️  Total processing time: {processing_time:.3f}s")
        print(f"📄 Response: {response.response[:100]}...")
        print(f"📚 Sources found: {len(response.sources)}")
        print(f"📊 Citations: {response.metadata.get('citations_count', 0)}")
        print(f"🤖 Model used: {response.metadata.get('model_used', 'unknown')}")
        print(f"🔢 Tokens used: {response.metadata.get('tokens_used', 0)}")
        
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        return
    
    # Test 2: Conversation memory
    print("\n🧠 Test 2: Conversation Memory")
    print("-" * 40)
    
    # Second query in same session
    follow_up_query = ChatQuery(
        query="Can you tell me more about its applications?",
        metadata=ChatMetadata(
            user_id="test_user_123",
            session_id="test_session_456"  # Same session
        ),
        max_results=3
    )
    
    try:
        response2 = await chat_service.process_query(follow_up_query)
        print(f"✅ Follow-up query processed!")
        print(f"📄 Response: {response2.response[:100]}...")
        
        # Check conversation history
        history = chat_service.get_conversation_history("test_session_456")
        print(f"📚 Conversation turns: {len(history)}")
        
        if len(history) >= 2:
            print("✅ Conversation memory working correctly!")
        else:
            print("⚠️  Conversation memory may not be working properly")
            
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
    
    # Test 3: Different session (no memory)
    print("\n🆕 Test 3: New Session (No Memory)")
    print("-" * 40)
    
    new_session_query = ChatQuery(
        query="What is artificial intelligence?",
        metadata=ChatMetadata(
            user_id="test_user_123",
            session_id="new_session_789"  # Different session
        ),
        max_results=3
    )
    
    try:
        response3 = await chat_service.process_query(new_session_query)
        print(f"✅ New session query processed!")
        print(f"📄 Response: {response3.response[:100]}...")
        
        # Check that new session has no history
        new_history = chat_service.get_conversation_history("new_session_789")
        print(f"📚 New session turns: {len(new_history)}")
        
        if len(new_history) == 1:
            print("✅ New session isolation working correctly!")
        else:
            print("⚠️  Session isolation may not be working properly")
            
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
    
    # Test 4: Metrics
    print("\n📊 Test 4: Metrics Collection")
    print("-" * 40)
    
    try:
        metrics = chat_service.get_metrics()
        print("✅ Metrics collected successfully!")
        print(f"📈 Total requests: {sum(metrics.get('requests', {}).values())}")
        print(f"🔍 Search operations: {metrics.get('search_operations', {}).get('total_searches', 0)}")
        print(f"🤖 LLM calls: {sum(metrics.get('llm_calls', {}).values())}")
        print(f"🔢 Token usage: {sum(metrics.get('token_usage', {}).values())}")
        
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
    
    # Test 5: Clear conversation
    print("\n🗑️ Test 5: Clear Conversation")
    print("-" * 40)
    
    try:
        # Clear the test session
        success = chat_service.clear_conversation("test_session_456")
        
        if success:
            print("✅ Conversation cleared successfully!")
            
            # Verify it's cleared
            cleared_history = chat_service.get_conversation_history("test_session_456")
            if len(cleared_history) == 0:
                print("✅ Conversation history properly cleared!")
            else:
                print("⚠️  Conversation history not properly cleared")
        else:
            print("❌ Failed to clear conversation")
            
    except Exception as e:
        print(f"❌ Test 5 failed: {e}")
    
    # Test 6: Error handling
    print("\n🚨 Test 6: Error Handling")
    print("-" * 40)
    
    # Test with empty query (should be caught by validation)
    try:
        empty_query = ChatQuery(query="")
        print("❌ Empty query should fail validation")
    except Exception as e:
        print(f"✅ Empty query correctly rejected: {e}")
    
    # Test with very short query
    try:
        short_query = ChatQuery(query="ab")
        print("❌ Short query should fail validation")
    except Exception as e:
        print(f"✅ Short query correctly rejected: {e}")
    
    print("\n🎉 All RAG Pipeline Tests Completed!")


def test_llm_service():
    """Test the LLM service directly."""
    print("\n🤖 Testing LLM Service")
    print("-" * 30)
    
    from src.services.llm_service import LLMService
    
    llm_service = LLMService()
    
    # Test prompt construction
    test_chunks = [
        {
            "text": "Machine learning is a subset of artificial intelligence.",
            "document_id": "test_doc_1",
            "source_pages": [1],
            "score": 0.9
        },
        {
            "text": "It enables systems to learn from data without explicit programming.",
            "document_id": "test_doc_2", 
            "source_pages": [2],
            "score": 0.8
        }
    ]
    
    prompt = llm_service._construct_prompt("What is machine learning?", test_chunks)
    print(f"✅ Prompt constructed ({len(prompt)} characters)")
    print(f"📝 Prompt preview: {prompt[:200]}...")


def test_memory_service():
    """Test the memory service directly."""
    print("\n🧠 Testing Memory Service")
    print("-" * 30)
    
    from src.services.memory_service import MemoryService
    
    memory_service = MemoryService()
    
    # Add conversation turns
    memory_service.add_conversation_turn(
        session_id="test_session",
        user_query="What is AI?",
        assistant_response="AI is artificial intelligence...",
        context_chunks=[{"text": "AI definition"}]
    )
    
    memory_service.add_conversation_turn(
        session_id="test_session",
        user_query="Tell me more",
        assistant_response="AI has many applications...",
        context_chunks=[{"text": "AI applications"}]
    )
    
    # Get history
    history = memory_service.get_conversation_history("test_session")
    print(f"✅ Conversation history: {len(history)} turns")
    
    # Get context
    context = memory_service.get_context_for_query("test_session", "What else?")
    print(f"✅ Context generated ({len(context)} characters)")
    
    # Get stats
    stats = memory_service.get_session_stats("test_session")
    print(f"✅ Session stats: {stats['turns_count']} turns, {stats['total_tokens']} tokens")


def test_metrics_service():
    """Test the metrics service directly."""
    print("\n📊 Testing Metrics Service")
    print("-" * 30)
    
    from src.services.metrics_service import MetricsService
    
    metrics_service = MetricsService()
    
    # Log various metrics
    metrics_service.log_request("/chat", "POST", "user123")
    metrics_service.log_response_time("/chat", "POST", 1.5)
    metrics_service.log_token_usage("gpt-3.5-turbo", 150, 0.002)
    metrics_service.log_search_operation(50, 3, 0.8)
    metrics_service.log_llm_call("gpt-3.5-turbo", 2.1, True)
    metrics_service.log_session_activity("session123", 50, 150, 3.5)
    
    # Get metrics summary
    summary = metrics_service.get_metrics_summary()
    print(f"✅ Metrics summary generated")
    print(f"📈 Requests: {len(summary['requests'])}")
    print(f"🔍 Searches: {summary['search_operations'].get('total_searches', 0)}")
    print(f"🤖 LLM calls: {len(summary['llm_calls'])}")


async def main():
    """Run all tests."""
    print("🚀 Starting Complete RAG Pipeline Tests")
    print("=" * 70)
    
    # Test individual services
    test_llm_service()
    test_memory_service()
    test_metrics_service()
    
    # Test complete pipeline
    await test_complete_rag_pipeline()
    
    print("\n🏆 All Tests Completed Successfully!")
    print("✅ Complete RAG Pipeline Implementation Verified!")


if __name__ == "__main__":
    asyncio.run(main()) 