"""
Main entry point for the TAAT AI Agent.

This module provides the main entry point for running the agent.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

from src.agent_core.agent import TaatAgent


def main():
    """Main entry point for the TAAT AI Agent."""
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    # Create and run the agent
    agent = TaatAgent()
    
    try:
        asyncio.run(agent.run_loop())
    except KeyboardInterrupt:
        print("\nShutting down TAAT Agent.")
    except Exception as e:
        print(f"Error running TAAT Agent: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
