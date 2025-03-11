import os
import json
import openai
import inspect
from typing import Any, Dict, List, Optional

from .logging_utils import log_message, spinner
from .models import AIResponseModel, AGENT_RESPONSE_SCHEMA
from .agent_manager import AgentManager
from .agent_tool import AgentTool

import dotenv
dotenv.load_dotenv()

# -----------------------------------------------------------------------------
# ConversationHistory
# -----------------------------------------------------------------------------
class ConversationHistory:
    """
    Manages conversation messages in a format compatible with the OpenAI Chat API.
    Allowed roles: "system", "user", "assistant", "function".
    """
    def __init__(self, system_message: Optional[str] = None):
        self.messages: List[Dict[str, Any]] = []
        if system_message:
            self._update_or_add("system", system_message)

    def _update_or_add(self, role: str, content: str):
        """
        Updates the first message with the given role if it exists; otherwise, adds it.
        """
        for msg in self.messages:
            if msg["role"] == role:
                msg["content"] = content
                return
        self.messages.insert(0, {"role": role, "content": content})

    def add_system(self, content: str):
        self._update_or_add("system", content)

    def add_user(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def add_function(self, content: str, name: str):
        """
        Adds a function message with the specified tool's name.
        """
        self.messages.append({"role": "function", "content": content, "name": name})

    def get_messages(self) -> List[Dict[str, Any]]:
        return self.messages

    def update_system(self, content: str):
        self._update_or_add("system", content)

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
        # Set API key.
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
        self.last_response: Optional[str] = None

        # Build a detailed system message:
        # Include your role and list all available tools along with the exact expected parameter names.
        if self.tools:
            tool_info_list = []
            for tool in self.tools.values():
                sig = inspect.signature(tool.func)
                params_list = list(sig.parameters.keys())
                tool_info_list.append(f"{tool.name} (expects parameters: {params_list})")
            tool_info = "; ".join(tool_info_list)
        else:
            tool_info = "none"

        # Optionally include child agent info if available.
        child_info = "none"  # Initially no children.
        system_message = (
            f"Your role is {self.role}. You have access to the following Tools: {tool_info}. "
            "When calling a tool (action type 'use_tool'), output a JSON that includes a 'tool' object with a 'name' "
            "and a 'params' field. The 'params' field must be a JSON-encoded string containing exactly the parameters "
            "expected by that tool. The expected JSON output must follow the provided schema. "
            f"Peer Agents: {child_info}."
        )
        # Initialize conversation history with this system message.
        self.history = ConversationHistory(system_message)
        
        self.manager.register(self)
        if self.parent:
            self.parent.register_child(self)

    def register_child(self, child: "Agent"):
        self.children.append(child)
        if self.verbose:
            log_message(self.name, f"Registered child agent '{child.name}'.", level="DEBUG")
        # Update system message to include child agent info.
        child_info = ", ".join(f"{c.name} ({c.description})" for c in self.children) or "none"
        # Rebuild the system message; tool info remains unchanged.
        if self.tools:
            tool_info_list = []
            for tool in self.tools.values():
                sig = inspect.signature(tool.func)
                params_list = list(sig.parameters.keys())
                tool_info_list.append(f"{tool.name} (expects parameters: {params_list})")
            tool_info = "; ".join(tool_info_list)
        else:
            tool_info = "none"
        system_message = (
            f"Your role is {self.role}. You have access to the following Tools: {tool_info}. "
            "When calling a tool (action type 'use_tool'), output a JSON that includes a 'tool' object with a 'name' "
            "and a 'params' field. The 'params' field must be a JSON-encoded string containing exactly the parameters "
            "expected by that tool. The expected JSON output must follow the provided schema. "
            f"Peer Agents: {child_info}."
        )
        self.history.update_system(system_message)

    def call_api(self, message: str) -> Any:
        """
        Appends a user message and calls the OpenAI API using the current conversation history.
        """
        self.history.add_user(message)
        api_params = {
            "model": self.model,
            "messages": self.history.get_messages(),
            "response_format": {"type": "json_schema", "json_schema": AGENT_RESPONSE_SCHEMA},
            "temperature": 0.7,
            "max_completion_tokens": 2048,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }
        try:
            if self.verbose:
                with spinner("Waiting for response..."):
                    response = openai.chat.completions.create(**api_params)
            else:
                response = openai.chat.completions.create(**api_params)
        except Exception as e:
            log_message(self.name, f"API call error: {e}", level="ERROR")
            raise
        return response

    def parse_response(self, reply: Any) -> AIResponseModel:
        """
        Parses the raw API reply into an AIResponseModel.
        """
        if isinstance(reply, list):
            reply = reply[0].get("text", str(reply[0]))
        elif not isinstance(reply, str):
            reply = str(reply)
        try:
            parsed = json.loads(reply)
            return AIResponseModel.model_validate(parsed)
        except Exception as e:
            log_message(self.name, f"Response parsing error: {e}", level="ERROR")
            self.history.add_assistant("Error parsing response: {e}")

    def send(self, message: str) -> AIResponseModel:
        """
        Sends a user message to the API and logs the assistant's reply.
        """
        response = self.call_api(message)
        reply = response.choices[0].message.content
        ai_response = self.parse_response(reply)
        self.history.add_assistant(reply)
        if self.verbose:
            log_message(self.name, f"Reasoning: {ai_response.reasoning}", level="INFO")
            for act in ai_response.actions:
                log_message(self.name, f"Action: {act}", level="DEBUG")
        return ai_response

    def _validate_strict_params(self, params: Dict[str, Any]) -> Dict[str, str]:
        """
        Converts all provided tool parameters to strings.
        """
        return {key: str(value) for key, value in params.items()}

    def process_actions(self, ai_response: AIResponseModel) -> bool:
        """
        Processes each action from the AI.
        Returns True if a final 'respond' action is encountered.
        """
        self.last_response = None
        final = False
        if self.verbose:
            log_message(self.name, f"Processing actions with reasoning: {ai_response.reasoning}", level="DEBUG")
        for action in ai_response.actions:
            if action.type == "respond":
                log_message(self.name, action.message, level="INFO")
                self.last_response = action.message
                final = True
            elif action.type == "use_tool":
                if action.tool and action.tool.name in self.tools:
                    tool_func = self.tools[action.tool.name].func
                    try:
                        # Attempt to parse the parameters.
                        params_input = action.tool.params
                        if isinstance(params_input, str):
                            params_parsed = json.loads(params_input)
                        else:
                            params_parsed = params_input
                        # If the parsed parameters are not a dict but a list, convert them.
                        if not isinstance(params_parsed, dict):
                            if isinstance(params_parsed, list):
                                sig = inspect.signature(tool_func)
                                expected_params = list(sig.parameters.keys())
                                if len(params_parsed) == len(expected_params):
                                    params_parsed = dict(zip(expected_params, params_parsed))
                                else:
                                    error_msg = (f"Parameter mismatch for {action.tool.name}: expected a dict or a list of length "
                                                 f"{len(expected_params)}, got a list of length {len(params_parsed)}")
                                    log_message(self.name, error_msg, level="ERROR")
                                    self.history.add_assistant(error_msg)
                                    continue
                            else:
                                error_msg = f"Tool params for {action.tool.name} must be a JSON object or list, got {type(params_parsed)}"
                                log_message(self.name, error_msg, level="ERROR")
                                self.history.add_assistant(error_msg)
                                continue
                    except Exception as ex:
                        error_msg = f"Invalid JSON in tool params: {ex}"
                        log_message(self.name, error_msg, level="ERROR")
                        self.history.add_assistant(error_msg)
                        continue

                    params = self._validate_strict_params(params_parsed)
                    # Validate parameter names using the tool function's signature.
                    sig = inspect.signature(tool_func)
                    expected_params = set(sig.parameters.keys())
                    provided_params = set(params.keys())
                    if provided_params != expected_params:
                        error_msg = (f"Parameter mismatch for {action.tool.name}: expected {expected_params}, got {provided_params}")
                        log_message(self.name, error_msg, level="ERROR")
                        self.history.add_assistant(error_msg)
                        continue
                    log_message(self.name, f"Using {action.tool.name} with params {params}", level="INFO")
                    try:
                        result = tool_func(**params)
                        if result is None:
                            tool_result = "Tool executed successfully with no output."
                        else:
                            tool_result = result
                        log_message(self.name, f"Tool result: {tool_result}", level="INFO")
                        self.history.add_function(f"{tool_result}", name=action.tool.name)
                    except Exception as e:
                        error_msg = f"Error with tool {action.tool.name}: {e}"
                        log_message(self.name, error_msg, level="ERROR")
                        self.history.add_assistant(error_msg)
                else:
                    error_msg = f"Tool not available: {action.tool.name if action.tool else None}"
                    log_message(self.name, error_msg, level="ERROR")
                    self.history.add_assistant(error_msg)
            elif action.type == "call_agent":
                allowed = {self.parent.name} if self.parent else set()
                allowed.update(c.name for c in self.children)
                if action.agent not in allowed:
                    log_message(self.name, f"Agent call not allowed: {action.agent}. Allowed: {allowed}", level="ERROR")
                    continue
                target = self.manager.get(action.agent)
                if target:
                    log_message(self.name, f"Delegating to {action.agent}: {action.message}", level="INFO")
                    final_response = target.run_conversation(action.message)
                    if final_response:
                        self.history.add_assistant(f"{action.agent}: {final_response}")
                        log_message(self.name, f"Received from {action.agent}: {final_response}", level="INFO")
                    else:
                        log_message(self.name, f"No final response from {action.agent}", level="ERROR")
                else:
                    log_message(self.name, f"Agent {action.agent} not found", level="ERROR")
            else:
                log_message(self.name, f"Unknown action type: {action.type}", level="ERROR")
        return final


    def run_conversation(self, initial_message: str) -> str:
        """
        Runs the conversation loop until a final 'respond' action is produced.
        """
        try:
            ai_response = self.send(initial_message)
            while not self.process_actions(ai_response):
                ai_response = self.send("Continue decision making.")
            return self.last_response
        except Exception as e:
            log_message(self.name, f"Conversation error: {e}", level="ERROR")
            raise
