import os
import json
from typing import Any, Dict, List, Optional
import inspect
import openai

from .logging_utils import log_message, spinner
from .models import AIResponseModel, AGENT_RESPONSE_SCHEMA
from .agent_manager import AgentManager
from .agent_tool import AgentTool

# -----------------------------------------------------------------------------
# ConversationHistory
# -----------------------------------------------------------------------------
class ConversationHistory:
    """Manages conversation messages in a format compatible with OpenAI Chat API."""
    
    def __init__(self, system_message: Optional[str] = None):
        self.messages = []
        if system_message:
            self.add_system(system_message)
    
    def add_system(self, content: str):
        for msg in self.messages:
            if msg["role"] == "system":
                msg["content"] = content
                return
        self.messages.insert(0, {"role": "system", "content": content})
    
    def add_message(self, role: str, content: str, **kwargs):
        """Generic method to add any type of message with optional additional fields"""
        message = {"role": role, "content": content, **kwargs}
        self.messages.append({k: v for k, v in message.items() if v is not None})
    
    def add_user(self, content: str):
        self.add_message("user", content)
    
    def add_assistant(self, content: str):
        self.add_message("assistant", content)
    
    def add_function(self, content: str, name: str):
        self.add_message("function", content, name=name)
    
    def get_messages(self):
        return self.messages

# -----------------------------------------------------------------------------
# Agent
# -----------------------------------------------------------------------------
class Agent:
    def __init__(
        self,
        name: str,
        role: str,
        description: str,
        manager: AgentManager,
        tools: Optional[Dict[str, AgentTool]] = None,
        parent: Optional["Agent"] = None,
        verbose: bool = False,
        model: str = None,
        api_key: Optional[str] = None
    ):
        openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        self.name = name
        self.role = role
        self.description = description
        self.tools = tools or {}
        self.parent = parent
        self.children = []
        self.verbose = verbose
        self.manager = manager
        self.model = model
        self.last_response = None
        
        # Build system message
        system_message = self._build_system_message()
        self.history = ConversationHistory(system_message)
        
        self.manager.register(self)
        if self.parent:
            self.parent.register_child(self)
    
    def _build_system_message(self) -> str:
        """Generates an efficient system message with available tools and peer agents."""
        # Tool information
        tool_info = "none"
        if self.tools:
            tool_descriptions = []
            for tool in self.tools.values():
                sig = inspect.signature(tool.func)
                params = list(sig.parameters.keys())
                tool_descriptions.append(f"{tool.name} (params: {', '.join(params)})")
            tool_info = "; ".join(tool_descriptions)
        
        # Child agent information
        child_info = "none"
        if self.children:
            child_info = ", ".join(f"{c.name} ({c.description})" for c in self.children)
        
        return (
            f"Role: {self.role}. Tools: {tool_info}. "
            "For tool use, include 'tool' object with 'name' and 'params' (JSON string). "
            f"Peer Agents: {child_info}."
        )
    
    def register_child(self, child: "Agent"):
        self.children.append(child)
        if self.verbose:
            log_message(self.name, f"Registered child agent '{child.name}'.", level="DEBUG")
        
        # Update system message with new child info
        self.history.add_system(self._build_system_message())
    
    def call_api(self, message: str) -> Any:
        """Calls the OpenAI API with the current conversation history."""
        self.history.add_user(message)
        
        api_params = {
            "model": self.model,
            "messages": self.history.get_messages(),
            "response_format": {"type": "json_schema", "json_schema": AGENT_RESPONSE_SCHEMA},
        }
        
        try:
            if self.verbose:
                with spinner("Waiting for response..."):
                    response = openai.chat.completions.create(**api_params)
            else:
                response = openai.chat.completions.create(**api_params)
        except Exception as e:
            log_message(self.name, f"API error: {e}", level="ERROR")
            raise
            
        return response
    
    def parse_response(self, reply: Any) -> AIResponseModel:
        """Parses API response to AIResponseModel."""
        if not isinstance(reply, str):
            reply = str(reply)
            
        try:
            parsed = json.loads(reply)
            return AIResponseModel.model_validate(parsed)
        except Exception as e:
            log_message(self.name, f"Parse error: {e}", level="ERROR")
            self.history.add_assistant(f"Error parsing response: {e}")
    
    def send(self, message: str) -> AIResponseModel:
        """Sends a message and processes the response."""
        response = self.call_api(message)
        reply = response.choices[0].message.content
        ai_response = self.parse_response(reply)
        self.history.add_assistant(reply)
        
        if self.verbose:
            log_message(self.name, f"Reasoning: {ai_response.reasoning}", level="INFO")
            
        return ai_response
    
    def process_actions(self, ai_response: AIResponseModel) -> bool:
        """Processes AI actions and returns True if final 'respond' action encountered."""
        self.last_response = None
        final = False
        
        for action in ai_response.actions:
            if action.type == "respond":
                log_message(self.name, action.message, level="INFO")
                self.last_response = action.message
                final = True
                
            elif action.type == "use_tool":
                if not (action.tool and action.tool.name in self.tools):
                    log_message(self.name, f"Tool not available: {action.tool.name if action.tool else None}", level="ERROR")
                    continue
                    
                tool_func = self.tools[action.tool.name].func
                
                try:
                    # Parse parameters
                    params = self._parse_tool_params(action.tool.params)
                    if params is None:
                        continue
                        
                    # Validate parameters
                    if not self._validate_tool_params(tool_func, params):
                        continue
                        
                    # Execute tool
                    log_message(self.name, f"Using {action.tool.name} with params {params}", level="INFO")
                    result = tool_func(**params)
                    tool_result = "Tool executed successfully with no output." if result is None else result
                    log_message(self.name, f"Tool result: {tool_result}", level="INFO")
                    self.history.add_function(f"{tool_result}", name=action.tool.name)
                    
                except Exception as e:
                    error_msg = f"Tool error ({action.tool.name}): {e}"
                    log_message(self.name, error_msg, level="ERROR")
                    self.history.add_assistant(error_msg)
                    
            elif action.type == "call_agent":
                self._handle_agent_call(action)
                
        return final
    
    def _parse_tool_params(self, params_input):
        """Parse tool parameters from string or object."""
        try:
            if isinstance(params_input, str):
                params = json.loads(params_input)
            else:
                params = params_input
                
            # Convert to dict if it's a list
            if isinstance(params, list):
                return None
            
            return {k: str(v) for k, v in params.items()}
            
        except Exception as e:
            log_message(self.name, f"Invalid params: {e}", level="ERROR")
            return None
    
    def _validate_tool_params(self, tool_func, params):
        """Validate tool parameters against the function signature."""
        sig = inspect.signature(tool_func)
        expected_params = set(sig.parameters.keys())
        provided_params = set(params.keys())
        
        if provided_params != expected_params:
            log_message(
                self.name, 
                f"Param mismatch: expected {expected_params}, got {provided_params}", 
                level="ERROR"
            )
            return False
            
        return True
    
    def _handle_agent_call(self, action):
        """Handle delegation to another agent."""
        allowed = {self.parent.name} if self.parent else set()
        allowed.update(c.name for c in self.children)
        
        if action.agent not in allowed:
            log_message(self.name, f"Agent call not allowed: {action.agent}", level="ERROR")
            return
            
        target = self.manager.get(action.agent)
        if not target:
            log_message(self.name, f"Agent {action.agent} not found", level="ERROR")
            return
            
        log_message(self.name, f"Delegating to {action.agent}: {action.message}", level="INFO")
        final_response = target.run_conversation(action.message)
        
        if final_response:
            self.history.add_assistant(f"{action.agent}: {final_response}")
            log_message(self.name, f"Received from {action.agent}: {final_response}", level="INFO")
        else:
            log_message(self.name, f"No response from {action.agent}", level="ERROR")
    
    def run_conversation(self, initial_message: str) -> str:
        """Run conversation until a final 'respond' action is produced."""
        try:
            ai_response = self.send(initial_message)
            while not self.process_actions(ai_response):
                ai_response = self.send("Continue.")
            return self.last_response
        except Exception as e:
            log_message(self.name, f"Conversation error: {e}", level="ERROR")
            raise
