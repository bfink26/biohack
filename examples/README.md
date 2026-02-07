# Health Universe A2A SDK Examples

This directory contains example agents demonstrating various features of the SDK.

## Installation

```bash
# From the project root
pip install -e .
```

## Examples

### 1. Simple Echo Agent (`simple_agent.py`)

A minimal agent that demonstrates the basics:
- Creating an agent by subclassing `Agent`
- Implementing required methods (`get_agent_name`, `get_agent_description`, `process_message`)
- Progress updates via `context.update_progress()`
- Starting an HTTP server with `agent.serve()`

```bash
python examples/simple_agent.py
```

### 2. Advanced Data Processor Agent (`advanced_agent.py`)

A more sophisticated agent demonstrating:
- **Custom validation** via `validate_message()` with `ValidationAccepted`/`ValidationRejected`
- **Progress updates** with real-time progress reporting
- **Artifacts** via `context.add_artifact()`
- **Lifecycle hooks** - `on_startup`, `on_task_start`, `on_task_complete`, `on_task_error`
- **Error handling** with custom error messages

```bash
python examples/advanced_agent.py
```

Runs on port 8001.

### 3. Document Inventory Agent (`document_inventory.py`)

Demonstrates document operations:
- Listing documents with `context.document_client.list_documents()`
- Accessing document metadata (id, name, filename, type, version, visibility)
- Filtering by document type (`user_upload` vs `agent_output`)

```bash
python examples/document_inventory.py
```

### 4. Medical Symptom Classifier (`medical_classifier.py`)

A simple healthcare agent that classifies symptoms by urgency:
- Basic keyword-based classification pattern
- Multiple urgency levels (EMERGENCY, URGENT, ROUTINE, SELF_CARE)
- Structured JSON response formatting

```bash
python examples/medical_classifier.py
```

### 5. Protocol Analyzer (`protocol_analyzer.py`)

Analyzes clinical trial protocol documents:
- Finding documents with `context.document_client.filter_by_name()`
- Downloading content with `context.document_client.download_text()`
- Writing output documents with `context.document_client.write()`
- Regex-based extraction of structured information
- Message validation with `validate_message()`

```bash
python examples/protocol_analyzer.py
```

### 6. Physician Follow-Up Agent (`physician_followup_agent.py`)

A comprehensive real-world agent that processes clinical documents:
- Document classification by type (SOAP notes, labs, radiology, studies)
- Document download and content analysis
- External API integration (OpenAI for summarization)
- `UpdateImportance` levels for progress updates
- `on_startup()` lifecycle hook for initialization
- Writing structured output documents

```bash
OPENAI_API_KEY=sk-... python examples/physician_followup_agent.py
```

## Testing an Agent

```bash
# View agent card
curl http://localhost:8000/.well-known/agent-card.json

# Send a message via JSON-RPC
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "params": {
      "message": {
        "messageId": "test-1",
        "role": "user",
        "parts": [{"text": "Hello, agent!"}]
      }
    },
    "id": 1
  }'
```

## Common Patterns

### Environment Variables

All examples support these environment variables:

- `HOST` - Server host (default: "0.0.0.0")
- `PORT` or `AGENT_PORT` - Server port (default: 8000)
- `RELOAD` - Enable auto-reload for development (default: "false")

```bash
PORT=8080 RELOAD=true python examples/simple_agent.py
```

### Programmatic Server Control

Instead of using `agent.serve()`, you can create the app and run it manually:

```python
from health_universe_a2a import Agent, AgentContext, create_app
import uvicorn

class MyAgent(Agent):
    def get_agent_name(self) -> str:
        return "My Agent"

    def get_agent_description(self) -> str:
        return "Does something useful"

    async def process_message(self, message: str, context: AgentContext) -> str:
        return f"Processed: {message}"

agent = MyAgent()
app = create_app(agent)

uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
```

### Development vs Production

For development with auto-reload:
```bash
RELOAD=true python examples/simple_agent.py
```

For production with multiple workers:
```python
from health_universe_a2a import create_app
import uvicorn

app = create_app(MyAgent())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
```

## Next Steps

- Check out the [main README](../README.md) for SDK documentation
