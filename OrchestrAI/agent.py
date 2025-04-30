# agent.py
import os
import json
from typing import Any, Dict, Optional, List
import inspect
import openai

from .logging_utils import log_message, spinner
from .models import AIResponseModel, AGENT_RESPONSE_JSON_SCHEMA
from .agent_manager import AgentManager
from .agent_tool import AgentTool

class ConversationHistory:
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
        entry = {"role": role, "content": content, **kwargs}
        self.messages.append({k: v for k, v in entry.items() if v is not None})
        
    def add_user(self, content: str):
        self.add_message("user", content)
        
    def add_assistant(self, content: str):
        self.add_message("assistant", content)
        
    def add_function(self, content: str, name: str):
        self.add_message("function", content, name=name)
        
    def get_messages(self):
        return self.messages

class Agent:
    def __init__(self, name: str, role: str, description: str,
                 manager: AgentManager, tools: Optional[Dict[str, AgentTool]] = None,
                 parent: Optional["Agent"] = None, verbose: bool = False,
                 model: str = None, api_key: Optional[str] = None):
        openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.name = name
        self.role = role
        self.description = description
        self.tools = tools or {}
        self.parent = parent
        self.children: List["Agent"] = []
        self.verbose = verbose
        self.manager = manager
        self.model = model
        self.last_response = None

        system_message = self._build_system_message()
        self.history = ConversationHistory(system_message)

        self.manager.register(self)
        if self.parent:
            self.parent.register_child(self)

    def _build_system_message(self) -> str:
        # Include each tool's name, params, AND description:
        if self.tools:
            tool_descs = []
            for tool in self.tools.values():
                params = ", ".join(inspect.signature(tool.func).parameters)
                tool_descs.append(f"{tool.name} (params: {params}) — {tool.description}")
            tool_info = "; ".join(tool_descs)
        else:
            tool_info = "none"

        child_info = ", ".join(f"{c.name} ({c.description})" for c in self.children) or "none"

        return (
            f"Role: {self.role}\n"
            f"Tools:\n  {tool_info}\n\n"
            "To call a tool, emit an action of type 'use_tool' with 'tool.name' and 'tool.params'.\n"
            f"Peer Agents: {child_info}."
        )

    def register_child(self, child: "Agent"):
        self.children.append(child)
        if self.verbose:
            log_message(self.name, f"Registered child agent '{child.name}'.", level="DEBUG")
        self.history.add_system(self._build_system_message())

    def call_api(self, message: str) -> Any:
        self.history.add_user(message)
        api_params = {
            "model": self.model,
            "messages": self.history.get_messages(),
            "response_format": {
                "type": "json_schema",
                "json_schema": AGENT_RESPONSE_JSON_SCHEMA
            },
        }
        try:
            if self.verbose:
                with spinner("Waiting for response..."):
                    return openai.chat.completions.create(**api_params)
            return openai.chat.completions.create(**api_params)
        except Exception as e:
            log_message(self.name, f"API error: {e}", level="ERROR")
            raise

    def parse_response(self, reply: Any) -> AIResponseModel:
        text = reply if isinstance(reply, str) else str(reply)
        try:
            data = json.loads(text)
            return AIResponseModel.model_validate(data)
        except Exception as e:
            log_message(self.name, f"Parse error: {e}", level="ERROR")
            self.history.add_assistant(f"Error parsing response: {e}")

    def send(self, message: str) -> AIResponseModel:
        resp = self.call_api(message)
        content = resp.choices[0].message.content
        ai_resp = self.parse_response(content)
        self.history.add_assistant(content)
        if self.verbose:
            log_message(self.name, f"Reasoning: {ai_resp.reasoning}", level="INFO")
        return ai_resp

    def process_actions(self, ai_response: AIResponseModel) -> bool:
        self.last_response = None
        done = False
        for action in ai_response.actions:
            if action.type == "respond":
                log_message(self.name, action.message, level="INFO")
                self.last_response = action.message
                done = True
            elif action.type == "use_tool":
                if not (action.tool and action.tool.name in self.tools):
                    log_message(self.name, f"Tool not available: {action.tool.name if action.tool else None}", level="ERROR")
                    continue
                func = self.tools[action.tool.name].func
                try:
                    params = self._parse_tool_params(action.tool.params)
                    if params is None or not self._validate_tool_params(func, params):
                        continue
                    log_message(self.name, f"Using {action.tool.name} with {params}", level="INFO")
                    result = func(**params)
                    out = "Tool executed successfully." if result is None else result
                    log_message(self.name, f"Result: {out}", level="INFO")
                    self.history.add_function(out, name=action.tool.name)
                except Exception as e:
                    log_message(self.name, f"Tool error: {e}", level="ERROR")
                    self.history.add_assistant(f"Tool error: {e}")
            elif action.type == "call_agent":
                self._handle_agent_call(action)
        return done

    def _parse_tool_params(self, params_input):
        """Parse tool parameters, preserving native types (int, float, etc.)."""
        try:
            # Decode JSON string if necessary
            if isinstance(params_input, str):
                params = json.loads(params_input)
            else:
                params = params_input

            # Only dicts are valid params
            if not isinstance(params, dict):
                return None

            # Return as-is (no str coercion)
            return params

        except Exception as e:
            log_message(self.name, f"Invalid params: {e}", level="ERROR")
            return None


    def _validate_tool_params(self, func, params):
        sig = inspect.signature(func)
        if set(sig.parameters) != set(params):
            log_message(self.name, f"Param mismatch: want {set(sig.parameters)}, got {set(params)}", level="ERROR")
            return False
        return True

    def _handle_agent_call(self, action):
        allowed = {self.parent.name} if self.parent else set()
        allowed |= {c.name for c in self.children}
        if action.agent not in allowed:
            log_message(self.name, f"Delegation not allowed: {action.agent}", level="ERROR")
            return
        tgt = self.manager.get(action.agent)
        if not tgt:
            log_message(self.name, f"Agent not found: {action.agent}", level="ERROR")
            return
        log_message(self.name, f"Delegating to {action.agent}", level="INFO")
        resp = tgt.run_conversation(action.message)
        if resp:
            self.history.add_assistant(f"{action.agent}: {resp}")
            log_message(self.name, f"Got from {action.agent}: {resp}", level="INFO")
        else:
            log_message(self.name, "No reply from child agent", level="ERROR")

    def run_conversation(self, initial_message: str) -> str:
        try:
            ai_resp = self.send(initial_message)
            while not self.process_actions(ai_resp):
                ai_resp = self.send("Continue.")
            return self.last_response
        except Exception as e:
            log_message(self.name, f"Conversation error: {e}", level="ERROR")
            raise
