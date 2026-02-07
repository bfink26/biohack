"""
Document Inventory Agent

An agent that lists all documents in the thread and provides metadata summary.
This example demonstrates:
- Using context.document_client.list_documents()
- Accessing document metadata (name, type, version)
- Simple document discovery patterns

Usage:
    python examples/document_inventory.py

Then send any message and the agent will return a list of documents in the thread.
"""

from typing import Any

from health_universe_a2a import Agent, AgentContext


class DocumentInventoryAgent(Agent):
    """
    Lists all documents in the thread with detailed metadata.

    Returns a formatted inventory of all documents available to the agent,
    including document types, versions, and visibility status.
    """

    def get_agent_name(self) -> str:
        return "Document Inventory"

    def get_agent_description(self) -> str:
        return (
            "Lists all documents in the current thread with metadata including "
            "name, type, version, and visibility status."
        )

    async def process_message(self, message: str, context: AgentContext) -> str:
        """
        List all documents in the thread.

        Args:
            message: Any message (ignored - always lists documents)
            context: Agent context with document_client property

        Returns:
            Formatted inventory of all documents
        """
        await context.update_progress("Scanning thread for documents...", 0.3)

        # List all documents (including hidden ones for completeness)
        try:
            docs = await context.document_client.list_documents(include_hidden=True)
        except Exception as e:
            return f"Error accessing documents: {e}"

        await context.update_progress("Compiling inventory...", 0.7)

        if not docs:
            return "No documents found in this thread."

        # Build inventory
        inventory: dict[str, Any] = {
            "total_documents": len(docs),
            "user_uploads": 0,
            "agent_outputs": 0,
            "documents": [],
        }

        for doc in docs:
            doc_info = {
                "id": doc.id,
                "name": doc.name,
                "filename": doc.filename,
                "type": doc.document_type,
                "version": doc.latest_version,
                "visible_to_user": doc.user_visible,
            }
            inventory["documents"].append(doc_info)

            if doc.document_type == "user_upload":
                inventory["user_uploads"] += 1
            elif doc.document_type == "agent_output":
                inventory["agent_outputs"] += 1

        await context.update_progress("Inventory complete", 1.0)

        # Format output
        output_lines = [
            "# Document Inventory",
            "",
            f"**Total Documents:** {inventory['total_documents']}",
            f"- User Uploads: {inventory['user_uploads']}",
            f"- Agent Outputs: {inventory['agent_outputs']}",
            "",
            "## Documents",
            "",
        ]

        for doc in inventory["documents"]:
            visibility = "visible" if doc["visible_to_user"] else "hidden"
            version_str = f"v{doc['version']}" if doc["version"] else "no version"
            output_lines.append(f"- **{doc['name']}** ({doc['type']}, {version_str}, {visibility})")
            output_lines.append(f"  - Filename: `{doc['filename']}`")
            output_lines.append(f"  - ID: `{doc['id']}`")
            output_lines.append("")

        return "\n".join(output_lines)


if __name__ == "__main__":
    agent = DocumentInventoryAgent()
    agent.serve(port=8002)
