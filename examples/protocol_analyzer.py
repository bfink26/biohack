"""
Clinical Protocol Analyzer Agent

An agent that finds clinical protocol documents and extracts key information.
This example demonstrates:
- Using context.document_client.filter_by_name() to find specific documents
- Using context.document_client.download_text() to read content
- Using context.document_client.write() to save analysis results
- Pattern matching on document content

Usage:
    python examples/protocol_analyzer.py

Then send a message asking about the protocol. The agent will:
1. Search for protocol documents in the thread
2. Extract key information (title, phase, objectives, endpoints)
3. Save the analysis as a new document
"""

import json
import re
from typing import Any

from health_universe_a2a import Agent, AgentContext, ValidationAccepted, ValidationRejected


class ProtocolAnalyzerAgent(Agent):
    """
    Analyzes clinical trial protocol documents.

    Searches for protocol documents in the thread, extracts key information
    like title, phase, objectives, and endpoints, and saves the analysis
    as a new document.
    """

    def get_agent_name(self) -> str:
        return "Protocol Analyzer"

    def get_agent_description(self) -> str:
        return (
            "Analyzes clinical trial protocol documents to extract key information "
            "including title, phase, objectives, inclusion/exclusion criteria, and endpoints."
        )

    def get_max_duration_seconds(self) -> int:
        return 300  # 5 minutes for protocol analysis

    async def validate_message(
        self, message: str, metadata: dict[str, Any]
    ) -> ValidationAccepted | ValidationRejected:
        """Validate that we can access documents."""
        # Simple validation - accept all messages
        return ValidationAccepted(estimated_duration_seconds=60)

    async def process_message(self, message: str, context: AgentContext) -> str:
        """
        Find and analyze protocol document.

        Args:
            message: User query (may specify document name)
            context: Agent context with document_client property

        Returns:
            Analysis summary with extracted information
        """
        await context.update_progress("Searching for protocol documents...", 0.1)

        # Search for protocol documents
        search_terms = ["protocol", "study", "trial", "clinical"]
        protocol_docs = []

        for term in search_terms:
            results = await context.document_client.filter_by_name(term)
            for doc in results:
                if doc not in protocol_docs:
                    protocol_docs.append(doc)

        if not protocol_docs:
            return (
                "No protocol documents found in this thread. "
                "Please upload a clinical trial protocol document first."
            )

        await context.update_progress(
            f"Found {len(protocol_docs)} potential protocol document(s)", 0.2
        )

        # Use the first matching document
        protocol_doc = protocol_docs[0]

        await context.update_progress(f"Reading: {protocol_doc.name}...", 0.3)

        # Download and read the protocol content
        try:
            content = await context.document_client.download_text(protocol_doc.id)
        except Exception as e:
            return f"Error reading protocol document: {e}"

        await context.update_progress("Extracting key information...", 0.5)

        # Extract key information
        analysis = self._extract_protocol_info(content)
        analysis["source_document"] = {
            "id": protocol_doc.id,
            "name": protocol_doc.name,
            "version": protocol_doc.latest_version,
        }

        await context.update_progress("Generating analysis report...", 0.7)

        # Format analysis as markdown
        report = self._format_analysis_report(analysis)

        # Save analysis as a new document
        await context.update_progress("Saving analysis results...", 0.9)

        await context.document_client.write(
            name="Protocol Analysis",
            content=report,
            filename="protocol_analysis.md",
        )

        # Also save structured JSON
        await context.document_client.write(
            name="Protocol Analysis (JSON)",
            content=json.dumps(analysis, indent=2),
            filename="protocol_analysis.json",
        )

        return f"Analysis complete! Analyzed: {protocol_doc.name}\n\n{report}"

    def _extract_protocol_info(self, content: str) -> dict[str, Any]:
        """Extract key information from protocol content."""
        analysis: dict[str, Any] = {
            "title": self._extract_title(content),
            "phase": self._extract_phase(content),
            "sponsor": self._extract_sponsor(content),
            "objectives": self._extract_objectives(content),
            "endpoints": self._extract_endpoints(content),
            "population": self._extract_population(content),
            "word_count": len(content.split()),
        }
        return analysis

    def _extract_title(self, content: str) -> str | None:
        """Extract protocol title."""
        # Look for common title patterns
        patterns = [
            r"(?:Protocol Title|Study Title|Title)[:\s]*([^\n]+)",
            r"^#\s*(.+)",  # Markdown heading
            r"(?:A|An)\s+(?:Phase\s+[IViv123]+\s+)?(?:Randomized|Double-Blind|Open-Label|Single-Arm)?[^\.]+Study[^\n]+",
        ]
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip() if match.lastindex else match.group(0).strip()
        return None

    def _extract_phase(self, content: str) -> str | None:
        """Extract clinical trial phase."""
        patterns = [
            r"Phase\s+([IViv123]+(?:/[IViv123]+)?)",
            r"Phase[:\s]+([IViv123]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return f"Phase {match.group(1).upper()}"
        return None

    def _extract_sponsor(self, content: str) -> str | None:
        """Extract sponsor name."""
        patterns = [
            r"(?:Sponsor|Sponsored by)[:\s]*([^\n]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _extract_objectives(self, content: str) -> dict[str, list[str]]:
        """Extract primary and secondary objectives."""
        objectives: dict[str, list[str]] = {"primary": [], "secondary": []}

        # Primary objectives
        primary_section = re.search(
            r"(?:Primary|Main)\s+Objective[s]?[:\s]*(.+?)(?:Secondary|$)",
            content,
            re.IGNORECASE | re.DOTALL,
        )
        if primary_section:
            text = primary_section.group(1)
            objectives["primary"] = self._extract_bullet_points(text)[:3]

        # Secondary objectives
        secondary_section = re.search(
            r"Secondary\s+Objective[s]?[:\s]*(.+?)(?:Endpoint|Inclusion|$)",
            content,
            re.IGNORECASE | re.DOTALL,
        )
        if secondary_section:
            text = secondary_section.group(1)
            objectives["secondary"] = self._extract_bullet_points(text)[:3]

        return objectives

    def _extract_endpoints(self, content: str) -> dict[str, list[str]]:
        """Extract primary and secondary endpoints."""
        endpoints: dict[str, list[str]] = {"primary": [], "secondary": []}

        # Primary endpoint
        primary_section = re.search(
            r"(?:Primary|Main)\s+Endpoint[s]?[:\s]*(.+?)(?:Secondary|$)",
            content,
            re.IGNORECASE | re.DOTALL,
        )
        if primary_section:
            text = primary_section.group(1)
            endpoints["primary"] = self._extract_bullet_points(text)[:2]

        # Secondary endpoints
        secondary_section = re.search(
            r"Secondary\s+Endpoint[s]?[:\s]*(.+?)(?:Safety|Inclusion|$)",
            content,
            re.IGNORECASE | re.DOTALL,
        )
        if secondary_section:
            text = secondary_section.group(1)
            endpoints["secondary"] = self._extract_bullet_points(text)[:3]

        return endpoints

    def _extract_population(self, content: str) -> dict[str, list[str]]:
        """Extract study population information."""
        population: dict[str, list[str]] = {"inclusion_criteria": [], "exclusion_criteria": []}

        # Inclusion criteria
        inclusion_section = re.search(
            r"Inclusion\s+Criteria[:\s]*(.+?)(?:Exclusion|$)",
            content,
            re.IGNORECASE | re.DOTALL,
        )
        if inclusion_section:
            text = inclusion_section.group(1)
            population["inclusion_criteria"] = self._extract_bullet_points(text)[:5]

        # Exclusion criteria
        exclusion_section = re.search(
            r"Exclusion\s+Criteria[:\s]*(.+?)(?:Study|Treatment|$)",
            content,
            re.IGNORECASE | re.DOTALL,
        )
        if exclusion_section:
            text = exclusion_section.group(1)
            population["exclusion_criteria"] = self._extract_bullet_points(text)[:5]

        return population

    def _extract_bullet_points(self, text: str) -> list[str]:
        """Extract bullet points or numbered items from text."""
        # Match bullet points, numbered items, or line breaks
        lines = re.split(r"[\n•\-\*]|\d+\.", text)
        items = []
        for line in lines:
            cleaned = line.strip()
            if cleaned and len(cleaned) > 10:  # Filter out short fragments
                items.append(cleaned[:200])  # Truncate long items
        return items

    def _format_analysis_report(self, analysis: dict[str, Any]) -> str:
        """Format analysis as markdown report."""
        lines = [
            "# Protocol Analysis Report",
            "",
            "## Overview",
            "",
        ]

        if analysis.get("title"):
            lines.append(f"**Title:** {analysis['title']}")
        if analysis.get("phase"):
            lines.append(f"**Phase:** {analysis['phase']}")
        if analysis.get("sponsor"):
            lines.append(f"**Sponsor:** {analysis['sponsor']}")
        lines.append(f"**Document Length:** {analysis['word_count']:,} words")
        lines.append("")

        # Objectives
        objectives = analysis.get("objectives", {})
        if objectives.get("primary") or objectives.get("secondary"):
            lines.append("## Objectives")
            lines.append("")
            if objectives.get("primary"):
                lines.append("### Primary Objectives")
                for obj in objectives["primary"]:
                    lines.append(f"- {obj}")
                lines.append("")
            if objectives.get("secondary"):
                lines.append("### Secondary Objectives")
                for obj in objectives["secondary"]:
                    lines.append(f"- {obj}")
                lines.append("")

        # Endpoints
        endpoints = analysis.get("endpoints", {})
        if endpoints.get("primary") or endpoints.get("secondary"):
            lines.append("## Endpoints")
            lines.append("")
            if endpoints.get("primary"):
                lines.append("### Primary Endpoints")
                for ep in endpoints["primary"]:
                    lines.append(f"- {ep}")
                lines.append("")
            if endpoints.get("secondary"):
                lines.append("### Secondary Endpoints")
                for ep in endpoints["secondary"]:
                    lines.append(f"- {ep}")
                lines.append("")

        # Population
        population = analysis.get("population", {})
        if population.get("inclusion_criteria") or population.get("exclusion_criteria"):
            lines.append("## Study Population")
            lines.append("")
            if population.get("inclusion_criteria"):
                lines.append("### Inclusion Criteria")
                for criterion in population["inclusion_criteria"]:
                    lines.append(f"- {criterion}")
                lines.append("")
            if population.get("exclusion_criteria"):
                lines.append("### Exclusion Criteria")
                for criterion in population["exclusion_criteria"]:
                    lines.append(f"- {criterion}")
                lines.append("")

        # Source
        source = analysis.get("source_document", {})
        if source:
            lines.append("---")
            lines.append("")
            lines.append(
                f"*Analyzed from: {source.get('name', 'Unknown')} (v{source.get('version', '?')})*"
            )

        return "\n".join(lines)


if __name__ == "__main__":
    agent = ProtocolAnalyzerAgent()
    agent.serve(port=8003)
