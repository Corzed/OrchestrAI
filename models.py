import json
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, field_validator  # Using Pydantic V2 style

class ToolModel(BaseModel):
    """
    Pydantic model representing a tool with an optional name and a parameters dictionary.
    """
    name: Optional[str] = None
    params: Dict[str, Any] = {}  # Default to an empty dictionary.

    class Config:
        extra = "forbid"  # Forbid any extra fields.

class ActionModel(BaseModel):
    """
    Pydantic model representing an action from the AI. Actions can be one of:
    - respond: Provide a final response.
    - use_tool: Indicate that a tool should be used.
    - call_agent: Request that another agent is called.
    """
    type: str
    agent: Optional[str] = None  # Name of target agent if the action is 'call_agent'.
    tool: Optional[Union[ToolModel, None]] = None  # Details of the tool if 'use_tool' is chosen.
    message: str  # Message content associated with the action.

    @field_validator("type")
    def check_type(cls, v):
        """
        Validates that the action type is one of the accepted values.
        """
        if v not in ["respond", "use_tool", "call_agent"]:
            raise ValueError("Invalid action type; must be 'respond', 'use_tool', or 'call_agent'.")
        return v

    class Config:
        extra = "forbid"  # Forbid any extra fields.

class AIResponseModel(BaseModel):
    """
    Pydantic model representing the AI's overall response.
    Contains the reasoning behind the decision and a list of actions.
    """
    reasoning: str
    actions: List[ActionModel]

    class Config:
        extra = "forbid"  # Forbid any extra fields.

# JSON schema used to instruct the OpenAI API on the desired response format.
AGENT_RESPONSE_SCHEMA = {
    "name": "ai_action_schema",
    "schema": {
        "type": "object",
        "required": ["reasoning", "actions"],
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "The reasoning behind the actions."
            },
            "actions": {
                "type": "array",
                "description": "List of actions.",
                "items": {
                    "type": "object",
                    "required": ["type", "agent", "tool", "message"],
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["respond", "use_tool", "call_agent"],
                            "description": "Action type."
                        },
                        "agent": {
                            "type": ["string", "null"],
                            "description": "Target agent (if call_agent)."
                        },
                        "tool": {
                            "anyOf": [
                                {
                                    "type": "object",
                                    "required": ["name", "params"],
                                    "properties": {
                                        "name": {
                                            "type": ["string", "null"],
                                            "description": "Tool name."
                                        },
                                        "params": {
                                            "type": "object",
                                            "properties": {},
                                            "additionalProperties": False,
                                            "description": "Tool parameters."
                                        }
                                    },
                                    "additionalProperties": False
                                },
                                {"type": "null"}
                            ],
                            "description": "Tool details."
                        },
                        "message": {
                            "type": "string",
                            "description": "Message content."
                        }
                    },
                    "additionalProperties": False
                }
            }
        },
        "additionalProperties": False
    },
    "strict": True  # Enforce strict schema validation.
}
