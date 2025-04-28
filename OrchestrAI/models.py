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
    reasoning: str
    actions: List[ActionModel]

    class Config:
        extra = "forbid"

# -----------------------------------------------------------------------------
# Structured Outputs JSON Schema for Agent responses
# -----------------------------------------------------------------------------
AGENT_RESPONSE_JSON_SCHEMA = {
    "name": "agent_response",
    "schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Chain-of-thought reasoning the agent used to pick its actions"
            },
            "actions": {
                "type": "array",
                "description": "A sequence of actions to perform",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["respond", "use_tool", "call_agent"],
                            "description": "What kind of action to take"
                        },
                        "agent": {
                            "type": ["string", "null"],
                            "description": "Name of a peer agent to delegate to, or null"
                        },
                        "tool": {
                            "type": ["object", "null"],
                            "description": "Tool invocation, if using use_tool; null otherwise",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The exact name of the tool to call"
                                },
                                "params": {
                                    "type": "string",
                                    "description": "Stringified JSON of parameters for the tool"
                                }
                            },
                            "required": ["name", "params"],
                            "additionalProperties": False
                        },
                        "message": {
                            "type": "string",
                            "description": "Text to send back for respond, or instructions for call_agent"
                        }
                    },
                    "required": ["type", "agent", "tool", "message"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["reasoning", "actions"],
        "additionalProperties": False
    },
    "strict": True
}
