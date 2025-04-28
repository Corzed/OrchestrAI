from typing import List, Optional
from pydantic import BaseModel, field_validator

class ToolModel(BaseModel):
    name: str
    params: str = "{}"  # Default empty JSON object
    
    class Config:
        extra = "forbid"

class ActionModel(BaseModel):
    type: str
    agent: Optional[str] = None
    tool: Optional[ToolModel] = None
    message: str = ""
    
    @field_validator("type")
    def validate_type(cls, v):
        if v not in {"respond", "use_tool", "call_agent"}:
            raise ValueError("Invalid action type: must be 'respond', 'use_tool', or 'call_agent'")
        return v
    
    class Config:
        extra = "forbid"

class AIResponseModel(BaseModel):
    reasoning: Optional[str] = None  # Make reasoning optional
    actions: List[ActionModel]
    
    class Config:
        extra = "forbid"

# Dynamic schema generation for response format
def get_response_schema(include_reasoning=True):
    properties = {
        "actions": {
            "type": "array",
            "description": "List of actions to take",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["respond", "use_tool", "call_agent"],
                        "description": "The type of action"
                    },
                    "agent": {
                        "type": "string",
                        "description": "Optional agent name for delegation"
                    },
                    "tool": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "params": {"type": "string"}
                        },
                        "required": ["name"]
                    },
                    "message": {"type": "string"}
                },
                "required": ["type"]
            }
        }
    }
    
    # Add reasoning field if enabled
    if include_reasoning:
        properties["reasoning"] = {
            "type": "string",
            "description": "Reasoning behind actions"
        }
    
    required_fields = ["actions"]
    if include_reasoning:
        required_fields.append("reasoning")
    
    return {
        "name": "agent_response",
        "schema": {
            "type": "object",
            "properties": properties,
            "required": required_fields,
            "additionalProperties": False
        },
        "strict": True
    }

AGENT_RESPONSE_SCHEMA = get_response_schema()["schema"]
