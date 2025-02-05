from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent import Agent  # Forward reference for type checking

class AgentManager:
    """
    Manager class that keeps track of all registered agents.
    It provides methods to register, retrieve, unregister, and list agents.
    """
    def __init__(self):
        self._agents: Dict[str, "Agent"] = {}

    def register(self, agent: "Agent") -> None:
        """
        Registers an agent with the manager.
        Raises an error if an agent with the same name already exists.
        """
        if agent.name in self._agents:
            raise ValueError(f"Agent with name '{agent.name}' already exists.")
        self._agents[agent.name] = agent

    def get(self, name: str) -> Optional["Agent"]:
        """
        Retrieves an agent by name.
        """
        return self._agents.get(name)

    def unregister(self, name: str) -> None:
        """
        Unregisters an agent by its name.
        """
        if name in self._agents:
            del self._agents[name]

    def all_agents(self) -> List["Agent"]:
        """
        Returns a list of all registered agents.
        """
        return list(self._agents.values())
