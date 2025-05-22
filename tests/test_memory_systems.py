"""
Tests for the memory systems.

This module contains tests for the advanced memory systems.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from src.memory_systems.config import MemoryConfig, VectorDBConfig, EmbeddingConfig, DatabaseConfig
from src.memory_systems.database import DatabaseManager, TraderProfile, MarketKnowledge, ActionPattern
from src.memory_systems.episodic import EpisodicMemory
from src.memory_systems.semantic import SemanticMemory
from src.memory_systems.procedural import ProceduralMemory
from src.memory_systems.manager import MemoryManager
from src.memory_systems.integration import EnhancedTaatAgent


@pytest.fixture
def mock_memory_config():
    """Fixture for mock memory configuration."""
    return MemoryConfig(
        vector_db=VectorDBConfig(
            api_key="mock_pinecone_key",
            index_name="test-index"
        ),
        embedding=EmbeddingConfig(
            api_key="mock_openai_key",
            model="text-embedding-ada-002"
        ),
        database=DatabaseConfig(
            provider="sqlite",
            connection_string="sqlite:///:memory:"
        ),
        max_episodic_memories=3,
        max_procedural_patterns=2
    )


@pytest.fixture
def mock_db_manager(mock_memory_config):
    """Fixture for mock database manager."""
    with patch('sqlalchemy.create_engine'), \
         patch('sqlalchemy.orm.sessionmaker'), \
         patch('sqlalchemy.orm.scoped_session'):
        db_manager = DatabaseManager(mock_memory_config.database)
        
        # Mock methods
        db_manager.get_trader_profile = AsyncMock(return_value={"id": "1", "trader_id": "trader1", "reliability": 0.8})
        db_manager.create_trader_profile = AsyncMock(return_value={"id": "1", "trader_id": "trader1", "reliability": 0.5})
        db_manager.update_trader_profile = AsyncMock(return_value={"id": "1", "trader_id": "trader1", "reliability": 0.6})
        
        db_manager.get_market_knowledge = AsyncMock(return_value={"id": "1", "symbol": "AAPL", "name": "Apple Inc."})
        db_manager.create_market_knowledge = AsyncMock(return_value={"id": "1", "symbol": "AAPL", "name": "Apple Inc."})
        db_manager.update_market_knowledge = AsyncMock(return_value={"id": "1", "symbol": "AAPL", "name": "Apple Inc."})
        
        db_manager.get_action_pattern = AsyncMock(return_value={"id": "1", "pattern_type": "trade", "effectiveness": 0.7})
        db_manager.create_action_pattern = AsyncMock(return_value={"id": "1", "pattern_type": "trade", "effectiveness": 0.7})
        db_manager.update_action_pattern = AsyncMock(return_value={"id": "1", "pattern_type": "trade", "effectiveness": 0.8})
        db_manager.get_action_patterns_by_type = AsyncMock(return_value=[
            {"id": "1", "pattern_type": "trade", "effectiveness": 0.7},
            {"id": "2", "pattern_type": "trade", "effectiveness": 0.6}
        ])
        
        yield db_manager


@pytest.fixture
def mock_episodic_memory(mock_memory_config):
    """Fixture for mock episodic memory."""
    with patch('pinecone.Pinecone'), \
         patch('openai.OpenAI'):
        episodic_memory = EpisodicMemory(
            mock_memory_config.vector_db,
            mock_memory_config.embedding
        )
        
        # Mock methods
        episodic_memory._get_embedding = AsyncMock(return_value=[0.1] * 1536)
        episodic_memory.store_experience = AsyncMock(return_value="exp_123")
        episodic_memory.retrieve_similar_experiences = AsyncMock(return_value=[
            {"input": {"content": "Test"}, "similarity_score": 0.9},
            {"input": {"content": "Another test"}, "similarity_score": 0.8}
        ])
        
        yield episodic_memory


@pytest.mark.asyncio
async def test_semantic_memory(mock_db_manager):
    """Test semantic memory functionality."""
    semantic_memory = SemanticMemory(mock_db_manager)
    
    # Test get_trader_profile
    profile = await semantic_memory.get_trader_profile("trader1")
    assert profile["trader_id"] == "trader1"
    mock_db_manager.get_trader_profile.assert_called_once_with("trader1")
    
    # Test update_trader_reliability
    updated_profile = await semantic_memory.update_trader_reliability("trader1", "success")
    assert updated_profile is not None
    mock_db_manager.update_trader_profile.assert_called_once()
    
    # Test get_market_knowledge
    knowledge = await semantic_memory.get_market_knowledge("AAPL")
    assert knowledge["symbol"] == "AAPL"
    mock_db_manager.get_market_knowledge.assert_called_once_with("AAPL")
    
    # Test cache functionality
    semantic_memory.clear_caches()
    assert semantic_memory.trader_cache == {}
    assert semantic_memory.market_cache == {}


@pytest.mark.asyncio
async def test_procedural_memory(mock_db_manager):
    """Test procedural memory functionality."""
    procedural_memory = ProceduralMemory(mock_db_manager)
    
    # Test store_pattern
    pattern = await procedural_memory.store_pattern("trade", {"action": "buy", "symbol": "AAPL"}, True)
    assert pattern is not None
    assert mock_db_manager.get_action_pattern.called
    
    # Test get_patterns_by_type
    patterns = await procedural_memory.get_patterns_by_type("trade", limit=5)
    assert len(patterns) == 2
    mock_db_manager.get_action_patterns_by_type.assert_called_once_with("trade", 5)
    
    # Test get_most_effective_patterns
    effective_patterns = await procedural_memory.get_most_effective_patterns("trade", min_effectiveness=0.6)
    assert len(effective_patterns) > 0
    
    # Test cache functionality
    procedural_memory.clear_cache()
    assert procedural_memory.pattern_cache == {}


@pytest.mark.asyncio
async def test_memory_manager(mock_memory_config, mock_db_manager, mock_episodic_memory):
    """Test memory manager functionality."""
    with patch('src.memory_systems.manager.WorkingMemory'), \
         patch('src.memory_systems.manager.EpisodicMemory', return_value=mock_episodic_memory), \
         patch('src.memory_systems.manager.SemanticMemory'), \
         patch('src.memory_systems.manager.ProceduralMemory'), \
         patch('src.memory_systems.manager.DatabaseManager', return_value=mock_db_manager):
        
        memory_manager = MemoryManager(mock_memory_config)
        
        # Mock methods
        memory_manager.working_memory.get_context = MagicMock(return_value={"conversation": []})
        memory_manager.semantic_memory.get_trader_profile = AsyncMock(return_value={"trader_id": "trader1"})
        memory_manager.semantic_memory.get_market_knowledge = AsyncMock(return_value={"symbol": "AAPL"})
        memory_manager.procedural_memory.get_relevant_patterns = AsyncMock(return_value=[{"pattern": "test"}])
        
        # Test get_context
        context = await memory_manager.get_context({"content": "Test input", "trader_id": "trader1", "symbol": "AAPL"})
        assert "working_memory" in context
        assert "episodic_memory" in context
        assert "semantic_memory" in context
        assert "procedural_memory" in context
        
        # Test update_memories
        await memory_manager.update_memories(
            {"content": "Test input", "trader_id": "trader1", "symbol": "AAPL"},
            {"content": "Test response"},
            {"success": True}
        )
        assert mock_episodic_memory.store_experience.called


@pytest.mark.asyncio
async def test_enhanced_agent_integration():
    """Test integration with the core agent architecture."""
    with patch('src.memory_systems.integration.load_enhanced_config'), \
         patch('src.memory_systems.integration.MemoryManager'), \
         patch('src.memory_systems.integration.TaatAgent.__init__', return_value=None), \
         patch('src.memory_systems.integration.TaatAgent.perception'), \
         patch('src.memory_systems.integration.TaatAgent.cognition'), \
         patch('src.memory_systems.integration.TaatAgent.action'):
        
        agent = EnhancedTaatAgent()
        
        # Mock methods
        agent.perception.process_input = AsyncMock(return_value={"content": "Processed input"})
        agent.memory_manager.get_context = AsyncMock(return_value={"mock": "context"})
        agent.cognition.process = AsyncMock(return_value={"content": "Response"})
        agent.action.execute = AsyncMock(return_value={"status": "success"})
        agent.memory_manager.update_memories = AsyncMock()
        
        # Test process_input
        result = await agent.process_input("Test input")
        
        # Verify the enhanced perception-cognition-action loop
        agent.perception.process_input.assert_called_once_with("Test input", "text")
        agent.memory_manager.get_context.assert_called_once()
        agent.cognition.process.assert_called_once_with({"content": "Processed input"}, {"mock": "context"})
        agent.action.execute.assert_called_once_with({"content": "Response"})
        agent.memory_manager.update_memories.assert_called_once()
