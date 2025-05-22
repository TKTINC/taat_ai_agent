# Agent Foundation - Phase 2: Memory Systems Implementation Prompt

## Objective
Develop advanced memory systems for the TAAT AI Agent, including episodic memory for storing past experiences, semantic memory for knowledge representation, procedural memory for learned behaviors, and memory retrieval and integration mechanisms.

## Context
In Phase 1, we established the core agent architecture with a basic working memory system that maintains conversation history and simple state tracking. Phase 2 builds upon this foundation to create more sophisticated memory systems that will enable the agent to learn from past experiences, maintain knowledge about traders and markets, and improve its decision-making over time.

## Requirements

1. **Episodic Memory**
   - Implement a vector database integration for storing past experiences
   - Create embeddings for trade signals, decisions, and outcomes
   - Develop similarity search for retrieving relevant past experiences
   - Implement time-based and relevance-based retrieval mechanisms

2. **Semantic Memory**
   - Create a knowledge base for trading terminology and concepts
   - Implement a structured representation of trader profiles and reliability
   - Develop a market knowledge component for tracking securities and trends
   - Build mechanisms for updating and refining knowledge over time

3. **Procedural Memory**
   - Implement pattern storage for successful trading strategies
   - Create a system for tracking tool usage patterns and effectiveness
   - Develop mechanisms for learning from past decision sequences
   - Build a priority-based retrieval system for action patterns

4. **Memory Integration**
   - Create a unified memory manager that coordinates all memory systems
   - Implement context-aware memory retrieval based on current situation
   - Develop mechanisms for resolving conflicts between memory systems
   - Build a memory refresh system to maintain relevant context

5. **Persistence Layer**
   - Implement database storage for long-term memory persistence
   - Create serialization/deserialization for memory objects
   - Develop backup and recovery mechanisms
   - Build memory maintenance utilities (pruning, optimization)

## Implementation Guidelines

- Use a vector database (Pinecone, Weaviate, or similar) for episodic memory
- Implement proper abstraction layers to allow switching between vector DB providers
- Use SQLite for local development and PostgreSQL for production
- Ensure all memory operations are asynchronous for performance
- Implement proper error handling and fallback mechanisms
- Create comprehensive unit tests for all memory components
- Document all memory interfaces and implementation details
- Consider memory size limitations and implement appropriate strategies

## Technical Approach

### Vector Database Integration

```python
from typing import Dict, List, Any, Optional
import numpy as np
from pinecone import Pinecone, ServerlessSpec

class EpisodicMemory:
    def __init__(self, config):
        self.config = config
        self.pc = Pinecone(api_key=config.pinecone_api_key)
        self.index_name = config.pinecone_index_name
        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)
        
    def _ensure_index_exists(self):
        # Check if index exists, create if not
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-west-2")
            )
    
    async def store_experience(self, experience: Dict[str, Any], embedding: List[float]) -> str:
        """Store an experience in episodic memory."""
        # Generate a unique ID
        experience_id = f"exp_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Store in vector DB
        self.index.upsert(
            vectors=[{
                "id": experience_id,
                "values": embedding,
                "metadata": experience
            }]
        )
        
        return experience_id
    
    async def retrieve_similar_experiences(self, 
                                          query_embedding: List[float], 
                                          limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve experiences similar to the query embedding."""
        results = self.index.query(
            vector=query_embedding,
            top_k=limit,
            include_metadata=True
        )
        
        return [match.metadata for match in results.matches]
```

### Semantic Memory

```python
class SemanticMemory:
    def __init__(self, db_connection):
        self.db = db_connection
        self.trader_profiles = {}
        self.market_knowledge = {}
        self.terminology = {}
        self._load_from_db()
    
    def _load_from_db(self):
        """Load semantic memory from database."""
        # Load trader profiles
        # Load market knowledge
        # Load terminology
        pass
    
    async def get_trader_profile(self, trader_id: str) -> Dict[str, Any]:
        """Get a trader's profile from semantic memory."""
        if trader_id in self.trader_profiles:
            return self.trader_profiles[trader_id]
        
        # Load from DB if not in memory
        profile = await self.db.get_trader_profile(trader_id)
        if profile:
            self.trader_profiles[trader_id] = profile
            return profile
        
        # Create new profile if not found
        return self._create_new_trader_profile(trader_id)
    
    async def update_trader_reliability(self, trader_id: str, outcome: str) -> None:
        """Update a trader's reliability based on trade outcome."""
        profile = await self.get_trader_profile(trader_id)
        
        # Update reliability metrics
        if outcome == "success":
            profile["successful_trades"] += 1
        elif outcome == "failure":
            profile["failed_trades"] += 1
        
        profile["reliability"] = (profile["successful_trades"] / 
                                 (profile["successful_trades"] + profile["failed_trades"])
                                 if (profile["successful_trades"] + profile["failed_trades"]) > 0
                                 else 0.5)
        
        # Save to DB
        await self.db.update_trader_profile(trader_id, profile)
```

### Memory Manager

```python
class MemoryManager:
    def __init__(self, config):
        self.config = config
        self.working_memory = WorkingMemory(max_history=config.max_history)
        self.episodic_memory = EpisodicMemory(config)
        self.semantic_memory = SemanticMemory(config.db_connection)
        self.procedural_memory = ProceduralMemory(config.db_connection)
    
    async def get_context(self, current_input: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive context for decision-making."""
        # Get basic context from working memory
        context = self.working_memory.get_context()
        
        # Get embedding for current input
        input_embedding = await self._get_embedding(current_input["content"])
        
        # Retrieve relevant episodic memories
        similar_experiences = await self.episodic_memory.retrieve_similar_experiences(
            input_embedding, limit=3
        )
        
        # Get trader information if available
        trader_info = {}
        if "trader_id" in current_input:
            trader_info = await self.semantic_memory.get_trader_profile(current_input["trader_id"])
        
        # Get relevant procedural patterns
        action_patterns = await self.procedural_memory.get_relevant_patterns(current_input)
        
        # Combine all context
        context.update({
            "similar_experiences": similar_experiences,
            "trader_info": trader_info,
            "action_patterns": action_patterns
        })
        
        return context
    
    async def update_memories(self, input_data: Dict[str, Any], 
                             response: Dict[str, Any], 
                             result: Dict[str, Any]) -> None:
        """Update all memory systems with new interaction."""
        # Update working memory
        self.working_memory.update(input_data, response, result)
        
        # Create embedding for this experience
        input_text = input_data.get("content", "")
        response_text = response.get("content", "")
        combined_text = f"{input_text} {response_text}"
        embedding = await self._get_embedding(combined_text)
        
        # Store in episodic memory
        experience = {
            "input": input_data,
            "response": response,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        await self.episodic_memory.store_experience(experience, embedding)
        
        # Update semantic memory if applicable
        if "trader_id" in input_data and "outcome" in result:
            await self.semantic_memory.update_trader_reliability(
                input_data["trader_id"], result["outcome"]
            )
        
        # Update procedural memory if applicable
        if "action_sequence" in response:
            await self.procedural_memory.store_pattern(
                response["action_sequence"], result.get("success", False)
            )
```

## Integration Considerations

- The memory systems should integrate with the existing agent architecture
- The `TaatAgent` class should be updated to use the new `MemoryManager`
- The cognition module should be enhanced to leverage the richer context
- The perception module should be updated to extract information for memory storage
- Consider the performance implications of vector database operations

## Evaluation Criteria

- Memory retrieval accuracy (relevant experiences are retrieved)
- Memory integration effectiveness (context improves decision quality)
- Performance and latency (memory operations don't significantly slow the agent)
- Persistence reliability (memories are properly stored and retrieved)
- Code quality, documentation, and test coverage

## Deliverables

1. Enhanced memory system implementation
2. Vector database integration
3. Database schema for semantic and procedural memory
4. Memory manager for coordinated memory access
5. Updated agent class to use the new memory systems
6. Comprehensive tests for all memory components
7. Documentation of memory architecture and usage

## Future Considerations

This phase focuses on the memory systems themselves. In future phases, we will:
- Enhance the cognition module to better utilize these memories
- Implement learning mechanisms that leverage the memory systems
- Develop more sophisticated knowledge representation
- Create memory visualization and inspection tools
