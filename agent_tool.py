from typing import Callable, Any

class AgentTool:
    """
    Represents a tool that an agent can use.
    Each tool has a name, description, and a function that is called when the tool is used.
    """
    def __init__(self, name: str, description: str, func: Callable[..., Any]):
        self.name = name
        self.description = description
        self.func = func

    def __call__(self, **kwargs) -> Any:
        """
        Makes the tool callable so that it can execute its function with the given parameters.
        """
        return self.func(**kwargs)
