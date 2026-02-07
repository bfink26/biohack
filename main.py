"""
Health Universe A2A Agent Template

This is a minimal agent template. Customize the MyAgent class to build your agent.

Run with:
    uv run python main.py

Your agent will be available at http://localhost:8000
"""

import os

import uvicorn

from health_universe_a2a import Agent, AgentContext, create_app

# Configure poppler path to enable working with PDF files
# Comment this out as well as the poppler dependencies in packages.txt if you don't need PDFs
poppler_path = os.getenv("POPPLER_PATH")
if not poppler_path:
    # Common locations for poppler-utils
    common_paths = [
        "/usr/bin",  # Linux (apt install poppler-utils)
        "/opt/homebrew/bin",  # macOS Apple Silicon (brew install poppler)
        "/usr/local/bin",  # macOS Intel (brew install poppler)
        "/app/.apt/usr/bin",  # Heroku buildpack
    ]

    for path in common_paths:
        pdftoppm_path = os.path.join(path, "pdftoppm")
        if os.path.exists(pdftoppm_path):
            poppler_path = path
            break

if poppler_path:
    print(f"Using poppler from: {poppler_path}")
    # Add to PATH so pdf2image can find it
    os.environ["PATH"] = f"{poppler_path}:{os.environ.get('PATH', '')}"
else:
    print("WARNING: poppler not found. PDF processing may fail.")
    print("Install with: apt-get install poppler-utils (Linux) or brew install poppler (macOS)")


class MyAgent(Agent):
    """A simple echo agent - replace with your own logic."""

    def get_agent_name(self) -> str:
        return "My Agent"

    def get_agent_description(self) -> str:
        return "A simple agent that echoes back the user's message"

    async def process_message(self, message: str, context: AgentContext) -> str:
        # Send a progress update
        await context.update_progress("Processing your message...", progress=0.5)

        # Your agent logic goes here
        result = f"You said: {message}"

        return result


# Create the ASGI app
app = create_app(MyAgent())

if __name__ == "__main__":
    # Configuration from environment
    port = int(os.getenv("PORT", os.getenv("AGENT_PORT", "8000")))
    host = os.getenv("HOST", "0.0.0.0")
    reload = os.getenv("RELOAD", "false").lower() == "true"

    # Run the server
    uvicorn.run(
        "main:app" if reload else app,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
