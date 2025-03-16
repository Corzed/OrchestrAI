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
  "name": "agent_response",
  "schema": {
    "type": "object",
    "properties": {
      "reasoning": {
        "type": "string",
        "description": "Reasoning behind actions."
      },
      "actions": {
        "type": "array",
        "description": "List of actions taken based on the reasoning.",
        "items": {
          "type": "object",
          "properties": {
            "type": {
              "type": "string",
              "enum": [
                "respond",
                "use_tool",
                "call_agent"
              ],
              "description": "The type of action."
            },
            "agent": {
              "type": "string",
              "description": "Optional agent associated with the action."
            },
            "tool": {
              "type": "object",
              "properties": {
                "name": {
                  "type": "string",
                  "description": "Name of the tool."
                },
                "params": {
                  "type": "string",
                  "description": "Parameters for the tool in JSON string format."
                }
              },
              "required": [
                "name",
                "params"
              ],
              "additionalProperties": False
            },
            "message": {
              "type": "string",
              "description": "Message related to the action."
            }
          },
          "required": [
            "type",
            "agent",
            "tool",
            "message"
          ],
          "additionalProperties": False
        }
      }
    },
    "required": [
      "reasoning",
      "actions"
    ],
    "additionalProperties": False
  },
  "strict": True
}
