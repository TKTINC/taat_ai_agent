# Agent Foundation - Phase 1: Core Agent Architecture Implementation Prompt

## Objective
Design and implement the basic agent structure with LLM integration, establish the perception-cognition-action loop, create the agent's system prompt and personality, and implement basic working memory and context management.

## Context
This is the first phase of the Agent Foundation workstream, which establishes the core architecture that will serve as the foundation for all other capabilities of the TAAT AI Agent. The focus is on creating a minimal but functional agent structure that can be progressively enhanced in later phases.

## Requirements

1. **Basic Agent Structure**
   - Design a modular Python-based agent architecture
   - Implement Claude API integration for the cognitive core
   - Create a tool registry system for function calling
   - Establish a simple event loop for agent operation

2. **Perception-Cognition-Action Loop**
   - Implement the basic perception module for input processing
   - Create the cognition module for decision-making using the LLM
   - Develop the action module for executing decisions
   - Establish the flow between these components

3. **System Prompt and Personality**
   - Design a comprehensive system prompt that defines the agent's purpose, constraints, and operational parameters
   - Create a consistent agent personality focused on trading assistance
   - Implement prompt templates for different agent functions
   - Ensure the agent maintains appropriate tone and style in communications

4. **Working Memory and Context Management**
   - Implement a basic working memory system to maintain conversation context
   - Create a simple state management system for tracking agent status
   - Develop mechanisms for context window management
   - Implement basic conversation history tracking

5. **Local Development Environment**
   - Create a Docker-based local development environment
   - Set up configuration management for API keys and settings
   - Implement logging and debugging capabilities
   - Create a simple CLI interface for agent interaction during development

## Implementation Guidelines

- Use Python 3.10+ as the primary development language
- Implement modular design with clear separation of concerns
- Follow clean architecture principles with dependency injection
- Use type hints throughout the codebase
- Create comprehensive unit tests for all components
- Document all major functions, classes, and modules
- Implement error handling and graceful degradation
- Use environment variables for configuration
- Follow PEP 8 style guidelines

## Technical Approach

### Agent Core Structure

```python
class TaatAgent:
    def __init__(self, config):
        self.config = config
        self.memory = WorkingMemory()
        self.perception = PerceptionModule()
        self.cognition = CognitionModule(self.config.llm_settings)
        self.action = ActionModule()
        self.tools = ToolRegistry()
        self.register_tools()
        
    def register_tools(self):
        # Register available tools
        pass
        
    async def run_loop(self):
        # Main agent loop
        while True:
            # 1. Perception: Get input
            input_data = await self.perception.process_input()
            
            # 2. Cognition: Process and decide
            context = self.memory.get_context()
            response = await self.cognition.process(input_data, context)
            
            # 3. Action: Execute decision
            result = await self.action.execute(response)
            
            # 4. Update memory
            self.memory.update(input_data, response, result)
```

### LLM Integration

```python
class CognitionModule:
    def __init__(self, llm_settings):
        self.llm_settings = llm_settings
        self.client = anthropic.Anthropic(api_key=llm_settings.api_key)
        
    async def process(self, input_data, context):
        system_prompt = self.get_system_prompt()
        messages = self.format_messages(system_prompt, context, input_data)
        
        response = await self.client.messages.create(
            model=self.llm_settings.model,
            system=system_prompt,
            messages=messages,
            max_tokens=self.llm_settings.max_tokens
        )
        
        return response
```

### Working Memory

```python
class WorkingMemory:
    def __init__(self, max_history=10):
        self.conversation_history = []
        self.max_history = max_history
        self.state = {}
        
    def get_context(self):
        return {
            "conversation": self.conversation_history,
            "state": self.state
        }
        
    def update(self, input_data, response, result):
        self.conversation_history.append({
            "input": input_data,
            "response": response,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
        # Trim history if needed
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
```

## Integration Considerations

- The Agent Foundation must provide clear interfaces for other workstreams to build upon
- The perception module should be designed to accommodate future social media monitoring capabilities
- The action module should be extensible for future trade execution capabilities
- The memory system should be designed with future expansion to vector databases in mind

## Evaluation Criteria

- Agent successfully processes input and generates appropriate responses
- Perception-cognition-action loop functions correctly
- System prompt effectively guides agent behavior
- Working memory maintains appropriate context
- Docker environment runs consistently across different machines
- Code is well-documented and follows best practices
- Unit tests cover core functionality

## Deliverables

1. Python codebase implementing the core agent architecture
2. Docker configuration for local development
3. System prompt and personality definition
4. Working memory implementation
5. Basic CLI for agent interaction
6. Documentation of the architecture and components
7. Unit tests for all major components
