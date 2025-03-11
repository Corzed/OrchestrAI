import json
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, field_validator

class ToolModel(BaseModel):
    name: Optional[str] = None
    params: str = ""  # JSON string representing parameters

    class Config:
        extra = "forbid"

class ActionModel(BaseModel):
    type: str
    agent: Optional[str] = None
    tool: Optional[ToolModel] = None
    message: str

    @field_validator("type")
    def validate_type(cls, v):
        if v not in {"respond", "use_tool", "call_agent"}:
            raise ValueError("Invalid action type. Must be 'respond', 'use_tool', or 'call_agent'.")
        return v

    class Config:
        extra = "forbid"

class AIResponseModel(BaseModel):
    reasoning: str
    actions: List[ActionModel]

    class Config:
        extra = "forbid"

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
                                            "type": "string",
                                            "description": "Tool parameters as a JSON-encoded string."
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
    "strict": True
}
