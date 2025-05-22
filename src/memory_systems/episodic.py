"""
Episodic memory implementation with Pinecone vector database integration.

This module provides episodic memory functionality for storing and retrieving
past experiences using vector embeddings and similarity search.
"""

import time
import uuid
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import openai
import pinecone
from pinecone import Pinecone, ServerlessSpec

from src.memory_systems.config import VectorDBConfig, EmbeddingConfig


class EpisodicMemory:
    """
    Episodic memory implementation with vector database integration.
    
    Stores and retrieves past experiences using vector embeddings and similarity search.
    """
    
    def __init__(self, vector_db_config: VectorDBConfig, embedding_config: EmbeddingConfig):
        """
        Initialize episodic memory.
        
        Args:
            vector_db_config: Vector database configuration
            embedding_config: Embedding model configuration
        """
        self.vector_db_config = vector_db_config
        self.embedding_config = embedding_config
        
        # Initialize OpenAI client for embeddings
        self.openai_client = openai.OpenAI(api_key=embedding_config.api_key)
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=vector_db_config.api_key)
        self.index_name = vector_db_config.index_name
        
        # Ensure index exists
        self._ensure_index_exists()
        
        # Get the index
        self.index = self.pc.Index(self.index_name)
    
    def _ensure_index_exists(self) -> None:
        """Ensure the Pinecone index exists, create if not."""
        # Check if index exists
        existing_indexes = self.pc.list_indexes()
        
        if self.index_name not in existing_indexes.names():
            # Create the index
            self.pc.create_index(
                name=self.index_name,
                dimension=self.vector_db_config.dimension,
                metric=self.vector_db_config.metric,
                spec=ServerlessSpec(
                    cloud="aws", 
                    region="us-west-2"
                )
            )
            # Wait for index to be ready
            while not self.index_name in self.pc.list_indexes().names():
                time.sleep(1)
    
    async def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        response = self.openai_client.embeddings.create(
            model=self.embedding_config.model,
            input=text
        )
        return response.data[0].embedding
    
    async def store_experience(self, experience: Dict[str, Any]) -> str:
        """
        Store an experience in episodic memory.
        
        Args:
            experience: Experience to store
            
        Returns:
            Experience ID
        """
        # Generate a unique ID
        experience_id = f"exp_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Add timestamp if not present
        if "timestamp" not in experience:
            experience["timestamp"] = datetime.now().isoformat()
        
        # Extract text for embedding
        text_parts = []
        if "input" in experience and isinstance(experience["input"], dict):
            if "content" in experience["input"]:
                text_parts.append(experience["input"]["content"])
        
        if "response" in experience and isinstance(experience["response"], dict):
            if "content" in experience["response"]:
                text_parts.append(experience["response"]["content"])
        
        combined_text = " ".join(text_parts)
        
        # Get embedding
        embedding = await self._get_embedding(combined_text)
        
        # Convert experience to metadata (string values only)
        metadata = {}
        for key, value in experience.items():
            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value
            elif value is None:
                metadata[key] = None
            else:
                # Convert complex objects to JSON string
                metadata[key] = json.dumps(value)
        
        # Store in vector DB
        self.index.upsert(
            vectors=[{
                "id": experience_id,
                "values": embedding,
                "metadata": metadata
            }],
            namespace=self.vector_db_config.namespace
        )
        
        return experience_id
    
    async def retrieve_similar_experiences(
        self, query: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve experiences similar to the query.
        
        Args:
            query: Query text
            limit: Maximum number of experiences to return
            
        Returns:
            List of similar experiences
        """
        # Get embedding for query
        query_embedding = await self._get_embedding(query)
        
        # Query vector DB
        results = self.index.query(
            vector=query_embedding,
            top_k=limit,
            include_metadata=True,
            namespace=self.vector_db_config.namespace
        )
        
        # Process results
        experiences = []
        for match in results.matches:
            # Convert metadata back to original format
            experience = {}
            for key, value in match.metadata.items():
                if key in ["input", "response", "result"]:
                    try:
                        experience[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        experience[key] = value
                else:
                    experience[key] = value
            
            # Add match score
            experience["similarity_score"] = match.score
            
            experiences.append(experience)
        
        return experiences
    
    async def retrieve_experiences_by_time(
        self, start_time: Optional[str] = None, end_time: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve experiences by time range.
        
        Args:
            start_time: Start time (ISO format)
            end_time: End time (ISO format)
            limit: Maximum number of experiences to return
            
        Returns:
            List of experiences
        """
        # Build filter
        filter_dict = {}
        if start_time:
            filter_dict["timestamp"] = {"$gte": start_time}
        if end_time:
            if "timestamp" not in filter_dict:
                filter_dict["timestamp"] = {}
            filter_dict["timestamp"]["$lte"] = end_time
        
        # Query vector DB
        results = self.index.query(
            vector=[0.0] * self.vector_db_config.dimension,  # Dummy vector
            top_k=limit,
            include_metadata=True,
            filter=filter_dict if filter_dict else None,
            namespace=self.vector_db_config.namespace
        )
        
        # Process results
        experiences = []
        for match in results.matches:
            # Convert metadata back to original format
            experience = {}
            for key, value in match.metadata.items():
                if key in ["input", "response", "result"]:
                    try:
                        experience[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        experience[key] = value
                else:
                    experience[key] = value
            
            experiences.append(experience)
        
        return experiences
    
    async def delete_experience(self, experience_id: str) -> bool:
        """
        Delete an experience from episodic memory.
        
        Args:
            experience_id: Experience ID
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            self.index.delete(
                ids=[experience_id],
                namespace=self.vector_db_config.namespace
            )
            return True
        except Exception:
            return False
    
    async def clear_all_experiences(self) -> bool:
        """
        Clear all experiences from episodic memory.
        
        Returns:
            True if cleared, False otherwise
        """
        try:
            self.index.delete(
                delete_all=True,
                namespace=self.vector_db_config.namespace
            )
            return True
        except Exception:
            return False
