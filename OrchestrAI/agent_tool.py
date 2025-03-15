from typing import Callable, Any

class AgentTool:
    """Tool that an agent can use."""
    
    def __init__(self, name: str, description: str, func: Callable[..., Any]):
        self.name = name
        self.description = description
        self.func = func
    
    def __call__(self, **kwargs) -> Any:
        """Execute the tool function with given parameters."""
        return self.func(**kwargs)
