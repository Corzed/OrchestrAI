# 🎭 OrchestrAI

[![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)

**OrchestrAI** is a powerful framework for orchestrating collaborative AI agents. Build sophisticated multi-agent systems where specialized agents work together, use tools, and solve complex problems through coordinated actions.

## 🔍 Overview

OrchestrAI enables you to create and manage a network of AI agents that can:
- Communicate and collaborate with each other
- Use specialized tools with structured inputs and outputs  
- Delegate tasks based on agent specialization
- Follow a hierarchical or peer-to-peer organization
- Execute complex workflows through agent coordination

## ✨ Key Features

- **Hierarchical Agent Architecture**: Create parent-child agent relationships with clear delegation paths
- **Dynamic Tool Integration**: Define and attach tools to agents with proper parameter validation
- **Structured Communication**: Agents communicate via a well-defined JSON schema for reliable interaction
- **Flexible Deployment**: Use different models for different agents based on task complexity
- **Comprehensive Logging**: Rich console output for debugging and monitoring agent activities
- **Error Resilience**: Robust error handling for API calls, tool execution, and inter-agent communication

## 📦 Installation

```bash
# Install from GitHub
pip install git+https://github.com/Corzed/OrchestrAI.git

# Alternative: Clone and install locally
git clone https://github.com/Corzed/OrchestrAI.git
cd OrchestrAI
pip install -e .
```

## 🚀 Quick Start

### Basic Example: Single Agent with Tool

```python
from OrchestrAI import AgentManager, Agent, AgentTool
import os

# Set up your OpenAI API key (or use .env file)
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Create an agent manager
manager = AgentManager()

# Define a calculator tool
def calculator(a: str, b: str, operator: str) -> str:
    try:
        a, b = int(a), int(b)
        if operator == "+": return str(a + b)
        elif operator == "-": return str(a - b)
        elif operator == "*": return str(a * b)
        elif operator == "/": 
            return str(a / b) if b != 0 else "Error: Division by zero"
        else:
            return f"Error: Unsupported operator '{operator}'"
    except Exception as e:
        return f"Error: {e}"

# Create a tool object
calc_tool = AgentTool(
    name="calculator",
    description="Performs basic arithmetic operations on two numbers",
    func=calculator
)

# Create an agent with the calculator tool
math_agent = Agent(
    name="Math_Assistant",
    role="You are a helpful math assistant that can perform calculations",
    description="Performs mathematical calculations",
    manager=manager,
    tools={"calculator": calc_tool},
    verbose=True,
    model="gpt-4o" # Or any model you prefer
)

# Run a conversation with the agent
response = math_agent.run_conversation("What is 125 * 37?")
print(f"Final answer: {response}")
```

### Advanced Example: Multi-Agent Collaboration

```python
from OrchestrAI import AgentManager, Agent, AgentTool

# Create manager
manager = AgentManager()

# Create main orchestrator agent
orchestrator = Agent(
    name="Orchestrator",
    role="You coordinate specialized agents to solve complex problems",
    description="Coordinates specialized agents",
    manager=manager,
    verbose=True,
    model="gpt-4o"
)

# Create specialized math agent
math_agent = Agent(
    name="Mathematician",
    role="You solve complex mathematical problems",
    description="Expert at mathematics",
    manager=manager,
    tools={"calculator": calc_tool},  # Using the calc_tool from earlier
    parent=orchestrator,  # Set parent relationship
    verbose=True,
    model="gpt-4o-mini"  # Can use a smaller model for specialized tasks
)

# Create specialized writer agent
writer_agent = Agent(
    name="Writer",
    role="You write clear explanations of complex topics",
    description="Expert at communication",
    manager=manager,
    parent=orchestrator,  # Set parent relationship
    verbose=True,
    model="gpt-4o-mini"
)

# Run a conversation that requires cooperation
response = orchestrator.run_conversation(
    "Calculate the compound interest on $10,000 invested for 5 years at 8% APR, "
    "and explain the result in simple terms for a non-financial person."
)
print(f"Final response: {response}")
```

## 📋 Core Components

### `AgentManager`

The central registry for all agents in your system:

```python
manager = AgentManager()
manager.register(agent)  # Usually handled automatically
agent = manager.get("agent_name")
manager.unregister("agent_name")
all_agents = manager.all_agents()
```

### `Agent`

The primary actor in OrchestrAI:

```python
agent = Agent(
    name="unique_name",          # Required: Unique identifier
    role="agent_instructions",   # Required: Instructions for the AI
    description="agent_summary", # Required: Short description
    manager=manager,             # Required: AgentManager
    tools={},                    # Optional: Dictionary of AgentTools
    parent=None,                 # Optional: Parent agent
    verbose=False,               # Optional: Enable detailed logging
    model="gpt-4o",              # Optional: OpenAI model to use
    api_key=None                 # Optional: API key (defaults to env)
)
```

### `AgentTool`

Function with metadata that agents can call:

```python
tool = AgentTool(
    name="tool_name",           # Required: Name for the tool
    description="tool_summary", # Required: Description of what the tool does
    func=my_function            # Required: The function to execute
)
```

Tool functions should have well-defined parameters and return strings for best results:

```python
def my_tool(param1: str, param2: str) -> str:
    # Process inputs and return a string result
    return f"Processed {param1} and {param2}"
```

## 🔄 Conversation Flow

1. **User Input**: Start with `agent.run_conversation(user_message)`
2. **Agent Processing**:
   - Agent receives the message
   - Agent decides on actions based on the message
   - Agent may call tools or delegate to other agents
3. **Final Response**: When ready, agent returns a final response

## 🛠️ Advanced Usage

### Custom System Messages

Tailor agent behaviors with custom system messages:

```python
agent.history.update_system(
    "You are a financial expert specializing in cryptocurrency market analysis. "
    "Always explain your reasoning and cite sources."
)
```

### Conversation History

Access and manipulate conversation history:

```python
# Add a system message
agent.history.add_system("New system instructions")

# Add a user message
agent.history.add_user("User input")

# Add an assistant message
agent.history.add_assistant("Assistant response")

# Get all messages
messages = agent.history.get_messages()
```

### Error Handling

OrchestrAI provides detailed error handling for various scenarios:

```python
try:
    response = agent.run_conversation("Complex query")
    print(f"Success: {response}")
except Exception as e:
    print(f"Error: {e}")
    # You can inspect agent.history for debugging
```

## 📚 Examples

### Research Assistant

```python
# Web search tool
def web_search(query: str) -> str:
    # Implement web search functionality here
    return f"Results for: {query}"

search_tool = AgentTool("web_search", "Search the web", web_search)

# Create research agent
researcher = Agent(
    name="Researcher",
    role="You are a research assistant who finds information",
    description="Conducts research on topics",
    manager=manager,
    tools={"web_search": search_tool},
    model="gpt-4o"
)

# Run a research task
response = researcher.run_conversation("Find information about quantum computing")
```

### Customer Support System

```python
# Create a multi-agent customer support system
support_manager = Agent(
    name="SupportManager",
    role="You manage customer support requests and delegate to specialists",
    description="Support request coordinator",
    manager=manager,
    model="gpt-4o"
)

tech_agent = Agent(
    name="TechSupport",
    role="You solve technical problems with products",
    description="Technical specialist",
    parent=support_manager,
    manager=manager,
    model="gpt-4o-mini"
)

billing_agent = Agent(
    name="BillingSupport",
    role="You handle billing and payment issues",
    description="Billing specialist",
    parent=support_manager,
    manager=manager,
    model="gpt-4o-mini"
)

# Process a support request
response = support_manager.run_conversation(
    "I'm having trouble connecting my device to Wi-Fi and I also need to update my payment method"
)
```

## 🔧 Debugging Tips

- Use `verbose=True` to enable detailed console logging
- Inspect `agent.history.get_messages()` to view the full conversation
- Use `AgentTool` return values for debugging tool execution
- Add intermediate `log_message()` calls in complex tool functions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
