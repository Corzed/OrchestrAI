import json
import time
from typing import Any, Dict, List, Optional, Union

import openai

from .logging_utils import log_message, spinner
from .models import AIResponseModel, AGENT_RESPONSE_SCHEMA
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
        model: str = None
    ):
        self.name = name
        self.role = role
        self.description = description
        self.tools = tools if tools else {}  # Dictionary of available tools.
        self.parent = parent
        self.children: List["Agent"] = []  # List to hold child agents.
        self.verbose = verbose
        self.conversation_history: List[Dict[str, Any]] = []  # History of messages exchanged.
        self.last_response: Optional[str] = None  # Final response from the agent.
        self.manager = manager
        self.model = model  # Store the selected model.

        # Register this agent with the manager.
        self.manager.register(self)

        # If this agent has a parent, register it as a child.
        if self.parent:
            self.parent.register_child(self)

        # Initialize the system message that describes the agent's current context.
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
                # Final response action.
                log_message(f"{self.name} Responds", action.message, level="INFO")
                self.last_response = action.message
                final = True
            elif action.type == "use_tool":
                # Action to use a specific tool.
                if action.tool and action.tool.name in self.tools:
                    log_message(self.name, f"Using {action.tool.name} with {action.tool.params}", level="INFO")
                    try:
                        # Execute the tool function with the provided parameters.
                        result = self.tools[action.tool.name](**action.tool.params)
                        log_message(self.name, f"Tool result: {result}", level="INFO")
                        self.add_message("tool", f"{action.tool.name} -> {result}")
                    except Exception as e:
                        log_message(self.name, f"Error with tool {action.tool.name}: {e}", level="ERROR")
                else:
                    log_message(self.name, f"Tool not available: {action.tool.name if action.tool else None}", level="ERROR")
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
