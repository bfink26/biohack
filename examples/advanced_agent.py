#!/usr/bin/env python3
"""
Advanced example agent demonstrating SDK features.

This example shows:
1. Custom message validation
2. Lifecycle hooks (on_startup, on_task_start, on_task_complete, on_task_error)
3. Progress updates and artifacts
4. Error handling

Run with:
    pip install -e .
    python examples/advanced_agent.py
"""

import asyncio
import json
from typing import Any

from health_universe_a2a import (
    Agent,
    AgentContext,
    ValidationAccepted,
    ValidationRejected,
)


class DataProcessorAgent(Agent):
    """An agent that processes data with validation and progress tracking."""

    def __init__(self) -> None:
        super().__init__()
        self.processed_count = 0

    # Required methods

    def get_agent_name(self) -> str:
        return "Data Processor Agent"

    def get_agent_description(self) -> str:
        return "Processes and analyzes data with validation and progress tracking"

    def get_agent_version(self) -> str:
        return "2.0.0"

    # Custom validation

    async def validate_message(
        self, message: str, metadata: dict[str, Any]
    ) -> ValidationAccepted | ValidationRejected:
        """Validate that the message is not empty and not too long."""
        if not message or len(message.strip()) == 0:
            return ValidationRejected(reason="Message cannot be empty")

        if len(message) > 1000:
            return ValidationRejected(
                reason=f"Message too long ({len(message)} chars). Maximum 1000 characters."
            )

        # Estimate processing time based on message length
        estimated_seconds = max(1, len(message) // 100)
        return ValidationAccepted(estimated_duration_seconds=estimated_seconds)

    # Main processing

    async def process_message(self, message: str, context: AgentContext) -> str:
        """Process the message with progress updates."""
        words = message.split()

        # Simulate processing with progress updates
        await context.update_progress("Analyzing message...", 0.0)
        await asyncio.sleep(0.5)

        await context.update_progress("Counting words...", 0.25)
        word_count = len(words)
        await asyncio.sleep(0.5)

        await context.update_progress("Analyzing sentiment...", 0.5)
        # Simple sentiment based on keywords
        positive_words = {"good", "great", "excellent", "amazing", "wonderful", "love"}
        negative_words = {"bad", "terrible", "awful", "hate", "poor"}

        positive_count = sum(1 for w in words if w.lower() in positive_words)
        negative_count = sum(1 for w in words if w.lower() in negative_words)

        if positive_count > negative_count:
            sentiment = "positive"
        elif negative_count > positive_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        await asyncio.sleep(0.5)

        await context.update_progress("Generating report...", 0.75)

        # Create analysis result
        analysis: dict[str, Any] = {
            "word_count": word_count,
            "character_count": len(message),
            "sentiment": sentiment,
            "positive_words": positive_count,
            "negative_words": negative_count,
        }

        # Add as artifact
        await context.add_artifact(
            "analysis.json", json.dumps(analysis, indent=2), data_type="application/json"
        )

        await asyncio.sleep(0.5)

        await context.update_progress("Complete!", 1.0)

        # Return summary
        return f"""Analysis Complete!

📊 Statistics:
- Words: {word_count}
- Characters: {len(message)}
- Sentiment: {sentiment.upper()}
- Positive words: {positive_count}
- Negative words: {negative_count}

✅ Full analysis saved to analysis.json artifact"""

    # Lifecycle hooks

    async def on_startup(self) -> None:
        """Called when the agent server starts."""
        self.logger.info("🚀 Data Processor Agent starting up!")
        self.processed_count = 0

    async def on_task_start(self, message: str, context: AgentContext) -> None:
        """Called before processing each message."""
        self.logger.info(f"📝 Starting task for user: {context.user_id}")

    async def on_task_complete(self, message: str, result: str, context: AgentContext) -> None:
        """Called after successfully processing a message."""
        self.processed_count += 1
        self.logger.info(f"✅ Task completed! Total processed: {self.processed_count}")

    async def on_task_error(
        self, message: str, error: Exception, context: AgentContext
    ) -> str | None:
        """Called when an error occurs during processing."""
        self.logger.error(f"❌ Task failed: {error}")
        if isinstance(error, asyncio.TimeoutError):
            return "Processing timed out. Please try a shorter message."
        return None  # Use default error message


if __name__ == "__main__":
    import logging

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create and start the agent
    agent = DataProcessorAgent()
    agent.serve(port=8001)  # Use port 8001 to avoid conflicts
