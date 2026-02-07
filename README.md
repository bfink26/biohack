# My A2A Agent

A Health Universe A2A-compliant agent.

## Quick Start

1. **Install dependencies**
   ```bash
   uv sync
   ```

2. **Run the agent**
   ```bash
   uv run python main.py
   ```

   For local development with auto-reload:
   ```bash
   RELOAD=true uv run python main.py
   ```

3. **Test it**
   ```bash
   # View agent card
   curl http://localhost:8000/.well-known/agent.json

   # Send a message
   curl -X POST http://localhost:8000/ \
     -H "Content-Type: application/json" \
     -d '{"jsonrpc": "2.0", "method": "message/send", "params": {"message": {"role": "user", "parts": [{"kind": "text", "text": "Hello!"}]}}}'
   ```

## Customizing Your Agent

Edit `main.py` to customize your agent:

- `get_agent_name()` - Your agent's display name
- `get_agent_description()` - What your agent does
- `process_message()` - Your agent's logic

## Adding Features

### Progress Updates

```python
from health_universe_a2a import UpdateImportance

# Progress updates are shown in Navigator UI by default
await context.update_progress("Analyzing data...", progress=0.5)

# Use INFO for verbose logging that shouldn't clutter the UI
await context.update_progress("Debug details...", importance=UpdateImportance.INFO)
```

> **Note:** The SDK automatically sends a terminal status when `process_message()` completes, ensuring the Navigator progress bar always finishes properly.

### Artifacts

```python
# Prefer markdown - the platform has markdown WYSIWYG support
await context.add_artifact(
    name="Analysis Report",
    content=markdown_report,
    data_type="text/markdown"
)
```

### Working with Documents

```python
# List documents in the thread
docs = await context.document_client.list_documents()

# Download a document
content = await context.document_client.download_text(docs[0].id)

# Write a new document
await context.document_client.write(
    name="Results",
    content='{"status": "complete"}',
    filename="results.json"
)
```

### Input Validation

```python
from health_universe_a2a import ValidationAccepted, ValidationRejected

async def validate_message(
    self, message: str, metadata: dict
) -> ValidationAccepted | ValidationRejected:
    if len(message) < 10:
        return ValidationRejected(reason="Message too short")
    return ValidationAccepted(estimated_duration_seconds=30)
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` or `AGENT_PORT` | Server port (set by platform) | `8000` |
| `HOST` | Server host (set by platform) | `0.0.0.0` |
| `RELOAD` | Enable auto-reload (local dev only) | `false` |

## Documentation

- [SDK Documentation](https://github.com/Health-Universe/healthuniverse-a2a-sdk-python)
- [A2A Protocol](https://a2a.ai)
