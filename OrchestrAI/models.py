from typing import List, Optional
from pydantic import BaseModel, field_validator

class ToolModel(BaseModel):
    name: Optional[str] = None
    params: str = ""  # JSON string of parameters
    
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
    reasoning: str
    actions: List[ActionModel]
    
    class Config:
        extra = "forbid"

# Simplified schema with only essential fields
AGENT_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["reasoning", "actions"],
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Reasoning behind actions"
        },
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["type"],
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["respond", "use_tool", "call_agent"]
                    },
                    "agent": {"type": "string"},
                    "tool": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "params": {"type": "string"}
                        }
                    },
                    "message": {"type": "string"}
                }
            }
        }
    }
}
