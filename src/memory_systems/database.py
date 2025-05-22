"""
Database models and persistence layer for memory systems.

This module provides database models and persistence layer for memory systems,
supporting both SQLite and PostgreSQL.
"""

import os
import json
import uuid
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool

from src.memory_systems.config import DatabaseConfig

# Base class for all models
Base = declarative_base()


class TraderProfile(Base):
    """Trader profile model for semantic memory."""
    __tablename__ = "trader_profiles"
    
    id = sa.Column(sa.String(36), primary_key=True)
    trader_id = sa.Column(sa.String(255), unique=True, nullable=False, index=True)
    username = sa.Column(sa.String(255), nullable=True)
    successful_trades = sa.Column(sa.Integer, default=0)
    failed_trades = sa.Column(sa.Integer, default=0)
    reliability = sa.Column(sa.Float, default=0.5)
    data = sa.Column(sa.Text, nullable=True)  # JSON data
    created_at = sa.Column(sa.DateTime, default=datetime.datetime.utcnow)
    updated_at = sa.Column(sa.DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "trader_id": self.trader_id,
            "username": self.username,
            "successful_trades": self.successful_trades,
            "failed_trades": self.failed_trades,
            "reliability": self.reliability,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        
        if self.data:
            try:
                result["data"] = json.loads(self.data)
            except json.JSONDecodeError:
                result["data"] = {}
        else:
            result["data"] = {}
            
        return result


class MarketKnowledge(Base):
    """Market knowledge model for semantic memory."""
    __tablename__ = "market_knowledge"
    
    id = sa.Column(sa.String(36), primary_key=True)
    symbol = sa.Column(sa.String(20), unique=True, nullable=False, index=True)
    name = sa.Column(sa.String(255), nullable=True)
    sector = sa.Column(sa.String(100), nullable=True)
    data = sa.Column(sa.Text, nullable=True)  # JSON data
    created_at = sa.Column(sa.DateTime, default=datetime.datetime.utcnow)
    updated_at = sa.Column(sa.DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "symbol": self.symbol,
            "name": self.name,
            "sector": self.sector,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        
        if self.data:
            try:
                result["data"] = json.loads(self.data)
            except json.JSONDecodeError:
                result["data"] = {}
        else:
            result["data"] = {}
            
        return result


class ActionPattern(Base):
    """Action pattern model for procedural memory."""
    __tablename__ = "action_patterns"
    
    id = sa.Column(sa.String(36), primary_key=True)
    pattern_type = sa.Column(sa.String(50), nullable=False, index=True)
    pattern_key = sa.Column(sa.String(255), nullable=False, index=True)
    success_count = sa.Column(sa.Integer, default=0)
    failure_count = sa.Column(sa.Integer, default=0)
    effectiveness = sa.Column(sa.Float, default=0.5)
    data = sa.Column(sa.Text, nullable=True)  # JSON data
    created_at = sa.Column(sa.DateTime, default=datetime.datetime.utcnow)
    updated_at = sa.Column(sa.DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "pattern_type": self.pattern_type,
            "pattern_key": self.pattern_key,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "effectiveness": self.effectiveness,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        
        if self.data:
            try:
                result["data"] = json.loads(self.data)
            except json.JSONDecodeError:
                result["data"] = {}
        else:
            result["data"] = {}
            
        return result


class DatabaseManager:
    """Database manager for memory systems."""
    
    def __init__(self, config: DatabaseConfig):
        """
        Initialize the database manager.
        
        Args:
            config: Database configuration
        """
        self.config = config
        self.engine = self._create_engine()
        self.session_factory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(self.session_factory)
        
    def _create_engine(self) -> sa.engine.Engine:
        """
        Create database engine.
        
        Returns:
            SQLAlchemy engine
        """
        return sa.create_engine(
            self.config.connection_string,
            poolclass=QueuePool,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_timeout=self.config.pool_timeout,
            pool_recycle=self.config.pool_recycle
        )
    
    def create_tables(self) -> None:
        """Create all tables."""
        Base.metadata.create_all(self.engine)
    
    def get_session(self):
        """
        Get a database session.
        
        Returns:
            Database session
        """
        return self.Session()
    
    def close_session(self) -> None:
        """Close the session."""
        self.Session.remove()
    
    async def get_trader_profile(self, trader_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a trader profile.
        
        Args:
            trader_id: Trader ID
            
        Returns:
            Trader profile or None if not found
        """
        session = self.get_session()
        try:
            profile = session.query(TraderProfile).filter_by(trader_id=trader_id).first()
            return profile.to_dict() if profile else None
        finally:
            self.close_session()
    
    async def create_trader_profile(self, trader_id: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a trader profile.
        
        Args:
            trader_id: Trader ID
            data: Additional data
            
        Returns:
            Created trader profile
        """
        session = self.get_session()
        try:
            profile = TraderProfile(
                id=str(uuid.uuid4()),
                trader_id=trader_id,
                username=data.get("username") if data else None,
                data=json.dumps(data) if data else None
            )
            session.add(profile)
            session.commit()
            return profile.to_dict()
        finally:
            self.close_session()
    
    async def update_trader_profile(self, trader_id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update a trader profile.
        
        Args:
            trader_id: Trader ID
            data: Updated data
            
        Returns:
            Updated trader profile or None if not found
        """
        session = self.get_session()
        try:
            profile = session.query(TraderProfile).filter_by(trader_id=trader_id).first()
            if not profile:
                return None
            
            # Update fields
            if "username" in data:
                profile.username = data["username"]
            if "successful_trades" in data:
                profile.successful_trades = data["successful_trades"]
            if "failed_trades" in data:
                profile.failed_trades = data["failed_trades"]
            if "reliability" in data:
                profile.reliability = data["reliability"]
            
            # Update JSON data
            profile_data = json.loads(profile.data) if profile.data else {}
            if "data" in data:
                profile_data.update(data["data"])
                profile.data = json.dumps(profile_data)
            
            session.commit()
            return profile.to_dict()
        finally:
            self.close_session()
    
    async def get_market_knowledge(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get market knowledge.
        
        Args:
            symbol: Market symbol
            
        Returns:
            Market knowledge or None if not found
        """
        session = self.get_session()
        try:
            knowledge = session.query(MarketKnowledge).filter_by(symbol=symbol).first()
            return knowledge.to_dict() if knowledge else None
        finally:
            self.close_session()
    
    async def create_market_knowledge(self, symbol: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create market knowledge.
        
        Args:
            symbol: Market symbol
            data: Additional data
            
        Returns:
            Created market knowledge
        """
        session = self.get_session()
        try:
            knowledge = MarketKnowledge(
                id=str(uuid.uuid4()),
                symbol=symbol,
                name=data.get("name") if data else None,
                sector=data.get("sector") if data else None,
                data=json.dumps(data) if data else None
            )
            session.add(knowledge)
            session.commit()
            return knowledge.to_dict()
        finally:
            self.close_session()
    
    async def update_market_knowledge(self, symbol: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update market knowledge.
        
        Args:
            symbol: Market symbol
            data: Updated data
            
        Returns:
            Updated market knowledge or None if not found
        """
        session = self.get_session()
        try:
            knowledge = session.query(MarketKnowledge).filter_by(symbol=symbol).first()
            if not knowledge:
                return None
            
            # Update fields
            if "name" in data:
                knowledge.name = data["name"]
            if "sector" in data:
                knowledge.sector = data["sector"]
            
            # Update JSON data
            knowledge_data = json.loads(knowledge.data) if knowledge.data else {}
            if "data" in data:
                knowledge_data.update(data["data"])
                knowledge.data = json.dumps(knowledge_data)
            
            session.commit()
            return knowledge.to_dict()
        finally:
            self.close_session()
    
    async def get_action_pattern(self, pattern_type: str, pattern_key: str) -> Optional[Dict[str, Any]]:
        """
        Get an action pattern.
        
        Args:
            pattern_type: Pattern type
            pattern_key: Pattern key
            
        Returns:
            Action pattern or None if not found
        """
        session = self.get_session()
        try:
            pattern = session.query(ActionPattern).filter_by(
                pattern_type=pattern_type, pattern_key=pattern_key
            ).first()
            return pattern.to_dict() if pattern else None
        finally:
            self.close_session()
    
    async def create_action_pattern(
        self, pattern_type: str, pattern_key: str, data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Create an action pattern.
        
        Args:
            pattern_type: Pattern type
            pattern_key: Pattern key
            data: Additional data
            
        Returns:
            Created action pattern
        """
        session = self.get_session()
        try:
            pattern = ActionPattern(
                id=str(uuid.uuid4()),
                pattern_type=pattern_type,
                pattern_key=pattern_key,
                data=json.dumps(data) if data else None
            )
            session.add(pattern)
            session.commit()
            return pattern.to_dict()
        finally:
            self.close_session()
    
    async def update_action_pattern(
        self, pattern_type: str, pattern_key: str, data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Update an action pattern.
        
        Args:
            pattern_type: Pattern type
            pattern_key: Pattern key
            data: Updated data
            
        Returns:
            Updated action pattern or None if not found
        """
        session = self.get_session()
        try:
            pattern = session.query(ActionPattern).filter_by(
                pattern_type=pattern_type, pattern_key=pattern_key
            ).first()
            if not pattern:
                return None
            
            # Update fields
            if "success_count" in data:
                pattern.success_count = data["success_count"]
            if "failure_count" in data:
                pattern.failure_count = data["failure_count"]
            if "effectiveness" in data:
                pattern.effectiveness = data["effectiveness"]
            
            # Update JSON data
            pattern_data = json.loads(pattern.data) if pattern.data else {}
            if "data" in data:
                pattern_data.update(data["data"])
                pattern.data = json.dumps(pattern_data)
            
            session.commit()
            return pattern.to_dict()
        finally:
            self.close_session()
    
    async def get_action_patterns_by_type(self, pattern_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get action patterns by type.
        
        Args:
            pattern_type: Pattern type
            limit: Maximum number of patterns to return
            
        Returns:
            List of action patterns
        """
        session = self.get_session()
        try:
            patterns = session.query(ActionPattern).filter_by(
                pattern_type=pattern_type
            ).order_by(ActionPattern.effectiveness.desc()).limit(limit).all()
            return [pattern.to_dict() for pattern in patterns]
        finally:
            self.close_session()
