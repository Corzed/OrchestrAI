import os
import json
import time
from typing import Any, Dict, List, Optional, Union
import ast

import openai

from .logging_utils import log_message, spinner
from .models import AIResponseModel, AGENT_RESPONSE_SCHEMA  # Remove ToolParams import
from .agent_manager import AgentManager
from .agent_tool import AgentTool

class Agent:
    """
    Represents an agent that can engage in conversations and perform actions based on AI responses.
    The agent can use tools, delegate to child agents, or directly respond.
    """
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
        api_key: Optional[str] = None  # New parameter for manual API key setting
    ):
        # Use the manual API key if provided; otherwise, try the environment variable.
        if api_key:
            openai.api_key = api_key
        else:
            # If openai.api_key is not already set, attempt to load from environment variable.
            if not openai.api_key:
                openai.api_key = os.getenv("OPENAI_API_KEY")
        
        self.name = name
        self.role = role
        self.description = description
        self.tools = tools if tools else {}
        self.parent = parent
        self.children: List["Agent"] = []
        self.verbose = verbose
        self.conversation_history: List[Dict[str, Any]] = []
        self.last_response: Optional[str] = None
        self.manager = manager
        self.model = model

        self.manager.register(self)
        if self.parent:
            self.parent.register_child(self)
        self.update_system_message(initial=True)

    def update_system_message(self, initial: bool = False):
        """
        Updates the system message that holds context about available tools and agents.
        """
        tool_list = ", ".join(f"{t.name} ({t.description})" for t in self.tools.values()) if self.tools else "none"
        child_list = ", ".join(f"{c.name} ({c.description})" for c in self.children) if self.children else "none"
        system_text = f"{self.role} Tools: {tool_list}. Agents: {child_list}."
        
        if self.conversation_history:
            current_system = self.conversation_history[0]["content"][0]["text"]
            if current_system == system_text:
                return  # No change required.
            # Update the system message.
            self.conversation_history[0] = {"role": "system", "content": [{"type": "text", "text": system_text}]}
        else:
            # Insert the system message if none exists.
            self.conversation_history.insert(0, {"role": "system", "content": [{"type": "text", "text": system_text}]})
        
        # Log the update if in verbose mode (but not on initial setup).
        if self.verbose and not initial:
            log_message(f"{self.name} System", system_text, level="DEBUG")

    def register_child(self, child: "Agent"):
        """
        Registers a child agent and updates the system message.
        """
        self.children.append(child)
        if self.verbose:
            log_message(self.name, f"Registered child agent '{child.name}'.", level="DEBUG")
        self.update_system_message()

    def add_message(self, sender: str, text: str):
        """
        Appends a new message to the conversation history.
        """
        full_text = f"{sender}: {text}"
        self.conversation_history.append({"role": "system", "content": [{"type": "text", "text": full_text}]})

    def send(self, message: str, sender: str = "user") -> AIResponseModel:
        """
        Sends a message to the OpenAI API, adds the message to the conversation history,
        receives a response, and parses it into an AIResponseModel.
        """
        # Add the outgoing message to the conversation history.
        self.add_message(sender, message)
        try:
            # Use the spinner if verbose mode is enabled.
            if self.verbose:
                with spinner("Waiting for response..."):
                    response = openai.chat.completions.create(
                        model=self.model,  # Use the model selected by the user.
                        messages=self.conversation_history,
                        response_format={"type": "json_schema", "json_schema": AGENT_RESPONSE_SCHEMA},
                        temperature=1,
                        max_completion_tokens=2048,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                    )
            else:
                # Make the API call without the spinner.
                response = openai.chat.completions.create(
                    model=self.model,  # Use the model selected by the user.
                    messages=self.conversation_history,
                    response_format={"type": "json_schema", "json_schema": AGENT_RESPONSE_SCHEMA},
                    temperature=1,
                    max_completion_tokens=2048,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
        except Exception as e:
            log_message(self.name, f"Error during API call: {e}", level="ERROR")
            raise

        # Extract the content of the reply.
        reply = response.choices[0].message.content

        # Process the reply in case it is in list format.
        if isinstance(reply, list):
            if reply and isinstance(reply[0], dict) and "text" in reply[0]:
                reply = reply[0]["text"]
            else:
                reply = str(reply[0])
        elif not isinstance(reply, str):
            reply = str(reply)

        try:
            # Attempt to parse the reply as JSON and validate it against the AIResponseModel.
            parsed = json.loads(reply)
            ai_response = AIResponseModel.model_validate(parsed)
        except Exception as e:
            log_message(self.name, f"Error parsing response: {e}", level="ERROR")
            raise ValueError(f"[{self.name}] Error parsing response: {e}")

        # Add the assistant's reply to the conversation history.
        self.add_message("assistant", reply)

        # If verbose, log the reasoning and actions.
        if self.verbose:
            log_message(f"{self.name} Received", f"Reasoning: {ai_response.reasoning}", level="INFO")
            for act in ai_response.actions:
                log_message(f"{self.name} Action", str(act), level="DEBUG")
        return ai_response

    def _validate_strict_params(self, params: Dict[str, Any]) -> Dict[str, str]:
        """
        Validates and converts tool parameters to strings, enforcing strict mode.
        Each value is converted via str() if not already a string.
        """
        validated = {}
        for key, value in params.items():
            if not isinstance(value, str):
                validated[key] = str(value)
            else:
                validated[key] = value
        return validated

    def process_actions(self, ai_response: AIResponseModel) -> bool:
        """
        Processes each action returned by the AI response.
        Executes tool usage, agent delegation, or final response accordingly.
        Returns True if a final response ('respond' action) was generated.
        """
        self.last_response = None
        final = False
        if self.verbose:
            log_message(f"{self.name} Processing", f"Reasoning: {ai_response.reasoning}", level="DEBUG")
        for action in ai_response.actions:
            if action.type == "respond":
                log_message(f"{self.name} Responds", action.message, level="INFO")
                self.last_response = action.message
                final = True
            elif action.type == "use_tool":
                if action.tool and action.tool.name in self.tools:
                    # If params is a string, parse it to a dict.
                    if isinstance(action.tool.params, str):
                        try:
                            params_dict = json.loads(action.tool.params)
                        except Exception as ex:
                            raise ValueError(f"Invalid JSON in tool params: {ex}")
                    else:
                        params_dict = action.tool.params
                    # Enforce strict values: convert every value to string.
                    params = self._validate_strict_params(params_dict)
                    log_message(self.name, f"Using {action.tool.name} with strict params {params}", level="INFO")
                    try:
                        result = self.tools[action.tool.name](**params)
                        log_message(self.name, f"Tool result: {result}", level="INFO")
                        self.add_message("tool", f"Successfully used {action.tool.name} -> {result}")
                    except Exception as e:
                        error_msg = f"Error with tool {action.tool.name}: {str(e)}"
                        log_message(self.name, error_msg, level="ERROR")
                        self.add_message("tool", f"Error while using {action.tool.name} -> {e}")
                else:
                    error_msg = f"Tool not available: {action.tool.name if action.tool else None}"
                    log_message(self.name, error_msg, level="ERROR")
                    self.add_message("error", error_msg)
            elif action.type == "call_agent":
                # Action to delegate a message to another agent.
                allowed = set()
                if self.parent:
                    allowed.add(self.parent.name)
                allowed.update(c.name for c in self.children)
                if action.agent not in allowed:
                    log_message(self.name, f"Cannot call agent {action.agent}. Allowed: {allowed}", level="ERROR")
                    continue
                target = self.manager.get(action.agent)
                if target:
                    log_message(f"{self.name} Delegates", f"To {action.agent}: {action.message}", level="INFO")
                    final_response = target.run_conversation(action.message)
                    if final_response:
                        self.add_message(action.agent, final_response)
                        log_message(f"{self.name} Received", f"From {action.agent}: {final_response}", level="INFO")
                    else:
                        log_message(self.name, f"No final response from {action.agent}", level="ERROR")
                else:
                    log_message(self.name, f"Agent {action.agent} not found", level="ERROR")
            else:
                log_message(self.name, f"Unknown action type: {action.type}", level="ERROR")
        return final


    def run_conversation(self, initial_message: str) -> str:
        """
        Runs the conversation loop until a final 'respond' action is generated.
        Returns the final response message.
        """
        try:
            # Send the initial message to the API.
            current = self.send(initial_message)
            # Process the actions from the response.
            complete = self.process_actions(current)
            # Continue the loop until a final response is produced.
            while not complete:
                current = self.send("Continue decision making.", sender=self.name)
                complete = self.process_actions(current)
            return self.last_response
        except Exception as e:
            log_message(self.name, f"Conversation halted due to error: {e}", level="ERROR")
            raise
