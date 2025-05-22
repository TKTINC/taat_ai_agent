"""
Performance tracking and metrics for the TAAT AI Agent.

This module provides performance tracking functionality for monitoring and
analyzing the agent's performance over time.
"""

import json
import uuid
import datetime
from typing import Dict, List, Any, Optional, Tuple


class PerformanceTracker:
    """
    Performance tracking system for the TAAT AI Agent.
    
    Monitors and analyzes the agent's performance over time.
    """
    
    def __init__(self, config, db_manager):
        """
        Initialize the performance tracker.
        
        Args:
            config: Performance configuration
            db_manager: Database manager
        """
        self.config = config
        self.db_manager = db_manager
        self.metrics_window_size = config.metrics_window_size
        self.min_sample_size = config.min_sample_size
        self.confidence_threshold = config.confidence_threshold
        self.report_frequency = config.report_frequency
        
        # Performance metrics history
        self.metrics_history = []
        self.last_report_time = None
    
    async def calculate_metrics(self, timeframe: str = "all") -> Dict[str, Any]:
        """
        Calculate performance metrics.
        
        Args:
            timeframe: Time frame for metrics (all, day, week, month)
            
        Returns:
            Performance metrics
        """
        # Get trade outcomes for the specified timeframe
        outcomes = await self._get_trade_outcomes(timeframe)
        
        # Calculate basic metrics
        total_trades = len(outcomes)
        
        if total_trades < self.min_sample_size:
            # Not enough data for reliable metrics
            return {
                "timeframe": timeframe,
                "total_trades": total_trades,
                "status": "insufficient_data",
                "min_sample_size": self.min_sample_size
            }
        
        successful_trades = sum(1 for o in outcomes if o.get("outcome") == "success")
        failed_trades = sum(1 for o in outcomes if o.get("outcome") == "failure")
        neutral_trades = total_trades - successful_trades - failed_trades
        
        success_rate = successful_trades / total_trades if total_trades > 0 else 0
        
        # Calculate profit/loss metrics
        total_profit = sum(o.get("profit_loss", 0) for o in outcomes if o.get("profit_loss", 0) > 0)
        total_loss = sum(o.get("profit_loss", 0) for o in outcomes if o.get("profit_loss", 0) < 0)
        net_profit = total_profit + total_loss
        
        # Calculate advanced metrics
        avg_profit_per_trade = total_profit / successful_trades if successful_trades > 0 else 0
        avg_loss_per_trade = total_loss / failed_trades if failed_trades > 0 else 0
        
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        
        # Calculate confidence
        confidence = min(1.0, total_trades / self.metrics_window_size)
        
        # Create metrics record
        metrics = {
            "id": str(uuid.uuid4()),
            "timeframe": timeframe,
            "total_trades": total_trades,
            "successful_trades": successful_trades,
            "failed_trades": failed_trades,
            "neutral_trades": neutral_trades,
            "success_rate": success_rate,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "net_profit": net_profit,
            "avg_profit_per_trade": avg_profit_per_trade,
            "avg_loss_per_trade": avg_loss_per_trade,
            "profit_factor": profit_factor,
            "confidence": confidence,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        
        # Add to history
        self.metrics_history.append(metrics)
        
        # Limit history size
        if len(self.metrics_history) > self.metrics_window_size:
            self.metrics_history = self.metrics_history[-self.metrics_window_size:]
        
        # Store metrics in database
        await self._store_metrics(metrics)
        
        return metrics
    
    async def calculate_trader_metrics(self, trader_id: str, timeframe: str = "all") -> Dict[str, Any]:
        """
        Calculate performance metrics for a specific trader.
        
        Args:
            trader_id: Trader ID
            timeframe: Time frame for metrics
            
        Returns:
            Trader performance metrics
        """
        # Get trade outcomes for the specified trader and timeframe
        outcomes = await self._get_trader_outcomes(trader_id, timeframe)
        
        # Calculate basic metrics
        total_trades = len(outcomes)
        
        if total_trades < self.min_sample_size:
            # Not enough data for reliable metrics
            return {
                "trader_id": trader_id,
                "timeframe": timeframe,
                "total_trades": total_trades,
                "status": "insufficient_data",
                "min_sample_size": self.min_sample_size
            }
        
        successful_trades = sum(1 for o in outcomes if o.get("outcome") == "success")
        failed_trades = sum(1 for o in outcomes if o.get("outcome") == "failure")
        neutral_trades = total_trades - successful_trades - failed_trades
        
        success_rate = successful_trades / total_trades if total_trades > 0 else 0
        
        # Calculate profit/loss metrics
        total_profit = sum(o.get("profit_loss", 0) for o in outcomes if o.get("profit_loss", 0) > 0)
        total_loss = sum(o.get("profit_loss", 0) for o in outcomes if o.get("profit_loss", 0) < 0)
        net_profit = total_profit + total_loss
        
        # Calculate confidence
        confidence = min(1.0, total_trades / self.metrics_window_size)
        
        # Create metrics record
        metrics = {
            "id": str(uuid.uuid4()),
            "trader_id": trader_id,
            "timeframe": timeframe,
            "total_trades": total_trades,
            "successful_trades": successful_trades,
            "failed_trades": failed_trades,
            "neutral_trades": neutral_trades,
            "success_rate": success_rate,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "net_profit": net_profit,
            "confidence": confidence,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        
        # Store metrics in database
        await self._store_trader_metrics(metrics)
        
        return metrics
    
    async def calculate_symbol_metrics(self, symbol: str, timeframe: str = "all") -> Dict[str, Any]:
        """
        Calculate performance metrics for a specific symbol.
        
        Args:
            symbol: Market symbol
            timeframe: Time frame for metrics
            
        Returns:
            Symbol performance metrics
        """
        # Get trade outcomes for the specified symbol and timeframe
        outcomes = await self._get_symbol_outcomes(symbol, timeframe)
        
        # Calculate basic metrics
        total_trades = len(outcomes)
        
        if total_trades < self.min_sample_size:
            # Not enough data for reliable metrics
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "total_trades": total_trades,
                "status": "insufficient_data",
                "min_sample_size": self.min_sample_size
            }
        
        successful_trades = sum(1 for o in outcomes if o.get("outcome") == "success")
        failed_trades = sum(1 for o in outcomes if o.get("outcome") == "failure")
        neutral_trades = total_trades - successful_trades - failed_trades
        
        success_rate = successful_trades / total_trades if total_trades > 0 else 0
        
        # Calculate profit/loss metrics
        total_profit = sum(o.get("profit_loss", 0) for o in outcomes if o.get("profit_loss", 0) > 0)
        total_loss = sum(o.get("profit_loss", 0) for o in outcomes if o.get("profit_loss", 0) < 0)
        net_profit = total_profit + total_loss
        
        # Calculate confidence
        confidence = min(1.0, total_trades / self.metrics_window_size)
        
        # Create metrics record
        metrics = {
            "id": str(uuid.uuid4()),
            "symbol": symbol,
            "timeframe": timeframe,
            "total_trades": total_trades,
            "successful_trades": successful_trades,
            "failed_trades": failed_trades,
            "neutral_trades": neutral_trades,
            "success_rate": success_rate,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "net_profit": net_profit,
            "confidence": confidence,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        
        # Store metrics in database
        await self._store_symbol_metrics(metrics)
        
        return metrics
    
    async def generate_performance_report(self, timeframe: str = "all") -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Args:
            timeframe: Time frame for the report
            
        Returns:
            Performance report
        """
        # Calculate overall metrics
        overall_metrics = await self.calculate_metrics(timeframe)
        
        # Calculate metrics by trader
        trader_ids = await self._get_active_trader_ids(timeframe)
        trader_metrics = {}
        
        for trader_id in trader_ids:
            trader_metrics[trader_id] = await self.calculate_trader_metrics(trader_id, timeframe)
        
        # Calculate metrics by symbol
        symbols = await self._get_active_symbols(timeframe)
        symbol_metrics = {}
        
        for symbol in symbols:
            symbol_metrics[symbol] = await self.calculate_symbol_metrics(symbol, timeframe)
        
        # Calculate trend metrics
        trend_metrics = await self._calculate_trend_metrics(timeframe)
        
        # Create report
        report = {
            "id": str(uuid.uuid4()),
            "timeframe": timeframe,
            "overall": overall_metrics,
            "by_trader": trader_metrics,
            "by_symbol": symbol_metrics,
            "trends": trend_metrics,
            "generated_at": datetime.datetime.utcnow().isoformat()
        }
        
        # Update last report time
        self.last_report_time = datetime.datetime.utcnow()
        
        # Store report in database
        await self._store_performance_report(report)
        
        return report
    
    async def should_generate_report(self) -> bool:
        """
        Check if a new performance report should be generated.
        
        Returns:
            True if a new report should be generated, False otherwise
        """
        if not self.last_report_time:
            return True
        
        # Check if enough time has passed since last report
        now = datetime.datetime.utcnow()
        hours_since_last_report = (now - self.last_report_time).total_seconds() / 3600
        
        return hours_since_last_report >= self.report_frequency
    
    async def _get_trade_outcomes(self, timeframe: str) -> List[Dict[str, Any]]:
        """
        Get trade outcomes for the specified timeframe.
        
        Args:
            timeframe: Time frame for outcomes
            
        Returns:
            List of trade outcomes
        """
        try:
            return await self.db_manager.get_trade_outcomes(timeframe)
        except Exception as e:
            # Log error and return empty list
            print(f"Error getting trade outcomes: {e}")
            return []
    
    async def _get_trader_outcomes(self, trader_id: str, timeframe: str) -> List[Dict[str, Any]]:
        """
        Get trade outcomes for the specified trader and timeframe.
        
        Args:
            trader_id: Trader ID
            timeframe: Time frame for outcomes
            
        Returns:
            List of trade outcomes
        """
        try:
            return await self.db_manager.get_trader_outcomes(trader_id, timeframe)
        except Exception as e:
            # Log error and return empty list
            print(f"Error getting trader outcomes: {e}")
            return []
    
    async def _get_symbol_outcomes(self, symbol: str, timeframe: str) -> List[Dict[str, Any]]:
        """
        Get trade outcomes for the specified symbol and timeframe.
        
        Args:
            symbol: Market symbol
            timeframe: Time frame for outcomes
            
        Returns:
            List of trade outcomes
        """
        try:
            return await self.db_manager.get_symbol_outcomes(symbol, timeframe)
        except Exception as e:
            # Log error and return empty list
            print(f"Error getting symbol outcomes: {e}")
            return []
    
    async def _get_active_trader_ids(self, timeframe: str) -> List[str]:
        """
        Get active trader IDs for the specified timeframe.
        
        Args:
            timeframe: Time frame for active traders
            
        Returns:
            List of active trader IDs
        """
        try:
            return await self.db_manager.get_active_trader_ids(timeframe)
        except Exception as e:
            # Log error and return empty list
            print(f"Error getting active trader IDs: {e}")
            return []
    
    async def _get_active_symbols(self, timeframe: str) -> List[str]:
        """
        Get active symbols for the specified timeframe.
        
        Args:
            timeframe: Time frame for active symbols
            
        Returns:
            List of active symbols
        """
        try:
            return await self.db_manager.get_active_symbols(timeframe)
        except Exception as e:
            # Log error and return empty list
            print(f"Error getting active symbols: {e}")
            return []
    
    async def _calculate_trend_metrics(self, timeframe: str) -> Dict[str, Any]:
        """
        Calculate trend metrics for the specified timeframe.
        
        Args:
            timeframe: Time frame for trend metrics
            
        Returns:
            Trend metrics
        """
        # Get historical metrics
        historical_metrics = await self._get_historical_metrics(timeframe)
        
        if len(historical_metrics) < 2:
            # Not enough data for trend analysis
            return {
                "status": "insufficient_data",
                "min_data_points": 2
            }
        
        # Sort by timestamp
        historical_metrics.sort(key=lambda x: x.get("timestamp", ""))
        
        # Calculate trends
        success_rate_trend = self._calculate_trend(
            [m.get("success_rate", 0) for m in historical_metrics]
        )
        
        net_profit_trend = self._calculate_trend(
            [m.get("net_profit", 0) for m in historical_metrics]
        )
        
        # Calculate volatility
        success_rate_volatility = self._calculate_volatility(
            [m.get("success_rate", 0) for m in historical_metrics]
        )
        
        net_profit_volatility = self._calculate_volatility(
            [m.get("net_profit", 0) for m in historical_metrics]
        )
        
        return {
            "success_rate_trend": success_rate_trend,
            "net_profit_trend": net_profit_trend,
            "success_rate_volatility": success_rate_volatility,
            "net_profit_volatility": net_profit_volatility,
            "data_points": len(historical_metrics)
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """
        Calculate trend for a list of values.
        
        Args:
            values: List of values
            
        Returns:
            Trend value (positive for upward trend, negative for downward trend)
        """
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression
        n = len(values)
        x = list(range(n))
        
        # Calculate means
        mean_x = sum(x) / n
        mean_y = sum(values) / n
        
        # Calculate slope
        numerator = sum((x[i] - mean_x) * (values[i] - mean_y) for i in range(n))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        
        # Normalize slope to [-1, 1]
        max_possible_slope = (values[-1] - values[0]) / (n - 1)
        if max_possible_slope == 0:
            return 0.0
        
        normalized_slope = slope / abs(max_possible_slope)
        
        return max(-1.0, min(1.0, normalized_slope))
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """
        Calculate volatility for a list of values.
        
        Args:
            values: List of values
            
        Returns:
            Volatility value (0 to 1, higher means more volatile)
        """
        if len(values) < 2:
            return 0.0
        
        # Calculate mean
        mean = sum(values) / len(values)
        
        # Calculate variance
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        
        # Calculate standard deviation
        std_dev = variance ** 0.5
        
        # Normalize to [0, 1]
        max_value = max(values)
        min_value = min(values)
        value_range = max_value - min_value
        
        if value_range == 0:
            return 0.0
        
        normalized_volatility = std_dev / value_range
        
        return min(1.0, normalized_volatility)
    
    async def _get_historical_metrics(self, timeframe: str) -> List[Dict[str, Any]]:
        """
        Get historical metrics for the specified timeframe.
        
        Args:
            timeframe: Time frame for historical metrics
            
        Returns:
            List of historical metrics
        """
        try:
            return await self.db_manager.get_historical_metrics(timeframe)
        except Exception as e:
            # Log error and return metrics history
            print(f"Error getting historical metrics: {e}")
            return self.metrics_history
    
    async def _store_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Store metrics in database.
        
        Args:
            metrics: Metrics to store
        """
        try:
            await self.db_manager.store_metrics(metrics)
        except Exception as e:
            # Log error
            print(f"Error storing metrics: {e}")
    
    async def _store_trader_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Store trader metrics in database.
        
        Args:
            metrics: Trader metrics to store
        """
        try:
            await self.db_manager.store_trader_metrics(metrics)
        except Exception as e:
            # Log error
            print(f"Error storing trader metrics: {e}")
    
    async def _store_symbol_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Store symbol metrics in database.
        
        Args:
            metrics: Symbol metrics to store
        """
        try:
            await self.db_manager.store_symbol_metrics(metrics)
        except Exception as e:
            # Log error
            print(f"Error storing symbol metrics: {e}")
    
    async def _store_performance_report(self, report: Dict[str, Any]) -> None:
        """
        Store performance report in database.
        
        Args:
            report: Performance report to store
        """
        try:
            await self.db_manager.store_performance_report(report)
        except Exception as e:
            # Log error
            print(f"Error storing performance report: {e}")
