#!/usr/bin/env python3
"""
Simple example agent demonstrating the Health Universe A2A SDK.

This example shows:
1. Creating a basic agent by subclassing Agent
2. Implementing required methods (get_agent_name, get_agent_description, process_message)
3. Starting an HTTP server with agent.serve()

Run with:
    pip install -e .
    python examples/simple_agent.py

Then test with:
    curl http://localhost:8000/.well-known/agent-card.json
"""

from health_universe_a2a import Agent, AgentContext


class SimpleEchoAgent(Agent):
    """A simple agent that echoes messages back with a prefix."""

    def get_agent_name(self) -> str:
        return "Simple Echo Agent"

    def get_agent_description(self) -> str:
        return "A simple demonstration agent that echoes messages back to you"

    def get_agent_version(self) -> str:
        return "1.0.0"

    async def process_message(self, message: str, context: AgentContext) -> str:
        """Process a message by echoing it back with a friendly prefix."""
        # You can access context properties
        user = context.user_id or "anonymous"

        # You can send progress updates during processing
        await context.update_progress("Processing your message...", 0.5)

        # Return the final response
        return f"Hello {user}! You said: {message}"


if __name__ == "__main__":
    # Create the agent
    agent = SimpleEchoAgent()

    # Start the server (reads HOST, PORT from environment, defaults to 0.0.0.0:8000)
    # Try: PORT=8080 python examples/simple_agent.py
    agent.serve()
