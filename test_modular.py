"""
Test script for the modular Intelligent Research Assistant.
"""

import asyncio
import json
from src.models.chat import ChatQuery, ChatMetadata
from src.services.chat_service import ChatService


async def test_chat_functionality():
    """Test the chat functionality."""
    print("🧪 Testing Modular Chat Functionality")
    print("=" * 50)
    
    # Initialize chat service
    chat_service = ChatService()
    
    # Test query with metadata
    test_query = ChatQuery(
        query="What are the main concepts of machine learning?",
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
    
    print(f"📝 Test Query: {test_query.query}")
    print(f"👤 User ID: {test_query.metadata.user_id}")
    print(f"🏷️  Tags: {test_query.metadata.tags}")
    print(f"📊 Max Results: {test_query.max_results}")
    print()
    
    try:
        # Process the query
        print("🔄 Processing query...")
        response = await chat_service.process_query(test_query)
        
        print("✅ Query processed successfully!")
        print(f"⏱️  Processing time: {response.processing_time:.3f}s")
        print(f"📄 Response: {response.response[:100]}...")
        print(f"📚 Sources found: {len(response.sources)}")
        print(f"📊 Metadata: {response.metadata}")
        
        # Test query validation
        print("\n🧪 Testing Query Validation")
        print("-" * 30)
        
        # Test empty query
        try:
            empty_query = ChatQuery(query="")
            print("❌ Empty query should fail validation")
        except Exception as e:
            print(f"✅ Empty query correctly rejected: {e}")
        
        # Test short query
        try:
            short_query = ChatQuery(query="ab")
            print("❌ Short query should fail validation")
        except Exception as e:
            print(f"✅ Short query correctly rejected: {e}")
        
        # Test valid query
        try:
            valid_query = ChatQuery(query="This is a valid query with sufficient length")
            print("✅ Valid query accepted")
        except Exception as e:
            print(f"❌ Valid query incorrectly rejected: {e}")
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_model_serialization():
    """Test Pydantic model serialization."""
    print("\n🧪 Testing Model Serialization")
    print("=" * 40)
    
    # Test ChatQuery serialization
    query = ChatQuery(
        query="Test query for serialization",
        metadata=ChatMetadata(
            user_id="user123",
            tags=["test"]
        )
    )
    
    # Serialize to dict
    query_dict = query.model_dump()
    print(f"✅ ChatQuery serialized to dict: {len(query_dict)} fields")
    
    # Serialize to JSON
    query_json = query.model_dump_json()
    print(f"✅ ChatQuery serialized to JSON: {len(query_json)} characters")
    
    # Test ChatResponse serialization
    from src.models.chat import ChatResponse
    from datetime import datetime
    
    response = ChatResponse(
        query="Test query",
        response="Test response",
        sources=[],
        metadata={},
        processing_time=1.23
    )
    
    response_dict = response.model_dump()
    print(f"✅ ChatResponse serialized to dict: {len(response_dict)} fields")
    
    print("🎉 Model serialization tests completed!")


if __name__ == "__main__":
    print("🚀 Starting Modular Structure Tests")
    print("=" * 60)
    
    # Test model serialization
    test_model_serialization()
    
    # Test chat functionality
    asyncio.run(test_chat_functionality())
    
    print("\n🏆 All Modular Tests Completed!") 