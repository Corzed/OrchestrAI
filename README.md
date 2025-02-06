# 🤖 OrchestrAI

A modern AI agentic orchestration framework. This framework supports multi-agent collaboration, tool usage, etc.

---

## ✨ Features

- **AI Response Processing**: Uses OpenAI API with strict JSON schema validation.
- **Multi-Agent Collaboration**: Agents can communicate and delegate tasks.
- **Tool Execution**: Agents can invoke tools with structured parameters.
- **Verbose Logging**: Uses `Rich` for detailed logs and progress spinners.
- **Error Handling**: Provides detailed error handling and debugging.

---

## 📦 Installation

```bash
pip install git+https://github.com/Corzed/OrchestrAI.git
```

---

## 🚀 Usage

### 2️⃣ Create an Agent Manager

```python
manager = AgentManager()
```

### 3️⃣ Define an AI Agent

```python
agent = Agent(
    name="Orchestrator",
    role="You are an Orchestrator of agents like yourself, delegate tasks to them if necessary.",
    description="Orchestrates Agents",
    manager=manager,
    verbose=True,
    model="gpt-4o",
    api_key="optional if you have it set in .env under OPENAI_API_KEY"
)
```

### 4️⃣ Add Tools to the Agent

```python
def calculator(a: int, b: int, operator: str) -> str:
    try:
        a, b = int(a), int(b)
        if operator == "+":
            result = a + b
        elif operator == "-":
            result = a - b
        elif operator == "*":
            result = a * b
        elif operator == "/":
            if b == 0:
                return "Error: Division by zero"
            result = a / b
        else:
            return "Error: Unsupported operator. Use one of '+', '-', '*', '/'"
        return str(result)
    except Exception as e:
        return f"Error: {e}"

calc_tool = AgentTool(
    name="calculator",
    description=(
        "A simple calculator tool that accepts three parameters: "
        "a (first integer), b (second integer), and operator (one of '+', '-', '*', '/')."
    ),
    func=calculator
)
```

### 5️⃣ Add Child Agents

Agents can collaborate by delegating tasks to child agents.

```python
child_agent = Agent(
    name="Mathematician",
    role="Solves mathematical problems",
    description="A specialized AI for math calculations.",
    manager=manager,
    verbose=True,
    tools={"calculator": calc_tool},
    model="gpt-4o-mini",
    parent=agent
)
```

Now, the main agent can delegate tasks to the child agent.

### 6️⃣ Run a Conversation

```python
response = agent.run_conversation("What's 2 + 2?")
print("Final Response:", response)
```

---

## 🛠 API Documentation

### **AgentManager**
- `register(agent: Agent)`: Registers an agent.
- `get(name: str) -> Optional[Agent]`: Retrieves an agent.
- `unregister(name: str)`: Unregisters an agent.

### **Agent**
- `send(message: str) -> AIResponseModel`: Sends a message to OpenAI API.
- `process_actions(ai_response: AIResponseModel) -> bool`: Processes AI actions.
- `run_conversation(initial_message: str) -> str`: Runs the conversation loop.

### **AgentTool**
- Represents a callable tool.
- Example: `tool = AgentTool("sum", "Adds numbers", lambda a, b: a + b)`

---

## 📄 License

This project is open-source and available under the **MIT License**.

---

## 🏆 Contributions

Contributions are welcome! Feel free to submit a pull request or report issues.
