from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent

class AgentManager:
    """Manager for tracking registered agents."""
    
    def __init__(self):
        self._agents = {}
    
    def register(self, agent: "Agent") -> None:
        """Register an agent."""
        if agent.name in self._agents:
            raise ValueError(f"Agent '{agent.name}' already exists")
        self._agents[agent.name] = agent
    
    def get(self, name: str) -> Optional["Agent"]:
        """Get agent by name."""
        return self._agents.get(name)
    
    def unregister(self, name: str) -> None:
        """Unregister an agent."""
        if name in self._agents:
            del self._agents[name]
    
    def all_agents(self) -> List["Agent"]:
        """Get all registered agents."""
        return list(self._agents.values())
