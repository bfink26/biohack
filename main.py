"""
Health Universe A2A Agent Template

This is a minimal agent template. Customize the MyAgent class to build your agent.

Run with:
    uv run python main.py

Your agent will be available at http://localhost:8000
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

import pandas as pd
import requests
import uvicorn
from openai import OpenAI

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


@dataclass
class KidneyIntake:
    raw_text: str
    donor_abo: str | None = None
    donor_type: str | None = None
    warm_ischemia_minutes: int | None = None
    cold_flush_time: str | None = None
    cross_clamp_time: str | None = None
    location: str | None = None
    preservation_method: str | None = None


class MyAgent(Agent):
    """Kidney transportation and recipient recommendation agent."""

    def get_agent_name(self) -> str:
        return "Kidney Transportation Option to Maximize Viability Giving Recipient Accessibility"

    def get_agent_description(self) -> str:
        return (
            "Recommends the best kidney recipient and preservation strategy based on intake data, "
            "recipient lists, and logistics constraints."
        )

    async def process_message(self, message: str, context: AgentContext) -> str:
        await context.update_progress("Parsing kidney intake data...", progress=0.1)
        kidney = self._parse_kidney_intake(message)

        await context.update_progress("Loading recipient and pump inventory data...", progress=0.25)
        recipients_df, pumps_df = await self._load_reference_data(context)

        await context.update_progress("Ranking recipients and preservation options...", progress=0.45)
        recipient_recommendation = self._rank_recipients(recipients_df, kidney)
        preservation_recommendation = self._recommend_preservation(kidney, pumps_df)

        await context.update_progress("Consulting clinical evidence sources...", progress=0.6)
        evidence = self._fetch_evidence_snippets(kidney)

        await context.update_progress("Drafting final recommendation...", progress=0.8)
        narrative = self._generate_llm_summary(
            message=message,
            kidney=kidney,
            recipient_recommendation=recipient_recommendation,
            preservation_recommendation=preservation_recommendation,
            evidence=evidence,
        )

        report_markdown = self._build_markdown_report(
            kidney=kidney,
            recipient_recommendation=recipient_recommendation,
            preservation_recommendation=preservation_recommendation,
            evidence=evidence,
            narrative=narrative,
        )

        report_pdf = self._render_pdf(report_markdown)
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        filename = f"kidney-recommendation-{timestamp}.pdf"

        await context.document_client.write(
            name="Kidney Recommendation Report",
            content=report_pdf,
            filename=filename,
        )
        await context.add_artifact(
            name="Kidney Recommendation Report",
            content=report_pdf,
            data_type="application/pdf",
        )
        await context.add_artifact(
            name="Kidney Recommendation (Markdown)",
            content=report_markdown,
            data_type="text/markdown",
        )

        await context.update_progress("Complete.", progress=1.0)
        return report_markdown

    def _parse_kidney_intake(self, message: str) -> KidneyIntake:
        raw_text = message.strip()
        donor_abo = None
        donor_type = None
        warm_ischemia_minutes = None
        location = None
        preservation_method = None
        cross_clamp_time = None
        cold_flush_time = None

        try:
            payload = json.loads(message)
            if isinstance(payload, dict):
                donor_abo = self._extract_field(payload, ["donor_abo", "abo", "blood_type"])
                donor_type = self._extract_field(payload, ["donor_type", "donor_type_label", "dcd_dbd"])
                warm_ischemia_minutes = self._parse_int(
                    self._extract_field(payload, ["warm_ischemia_minutes", "warm_ischemia", "fwi_minutes"])
                )
                location = self._extract_field(payload, ["location", "current_location", "organ_location"])
                preservation_method = self._extract_field(
                    payload, ["preservation_method", "current_preservation", "storage_method"]
                )
                cross_clamp_time = self._extract_field(payload, ["cross_clamp_time", "cross_clamp"])
                cold_flush_time = self._extract_field(payload, ["cold_flush_time", "cold_flush"])
        except json.JSONDecodeError:
            pass

        return KidneyIntake(
            raw_text=raw_text,
            donor_abo=donor_abo,
            donor_type=donor_type,
            warm_ischemia_minutes=warm_ischemia_minutes,
            cold_flush_time=cold_flush_time,
            cross_clamp_time=cross_clamp_time,
            location=location,
            preservation_method=preservation_method,
        )

    async def _load_reference_data(
        self, context: AgentContext
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        recipients_df = None
        pumps_df = None
        documents = await context.document_client.list_documents()

        for doc in documents:
            name = (doc.name or "").lower()
            filename = (doc.filename or "").lower()
            if not (name.endswith(".csv") or filename.endswith(".csv")):
                continue

            content = await context.document_client.download_text(doc.id)
            df = pd.read_csv(io.StringIO(content))

            if "recipient" in name or "recipient" in filename:
                recipients_df = df
            elif "pump" in name or "pump" in filename or "preservation" in name:
                pumps_df = df

        return recipients_df, pumps_df

    def _rank_recipients(self, recipients_df: pd.DataFrame | None, kidney: KidneyIntake) -> dict[str, Any]:
        if recipients_df is None or recipients_df.empty:
            return {
                "status": "no_recipients",
                "message": "No recipient CSV provided; unable to rank candidates.",
                "top_candidate": None,
                "scored_table": None,
            }

        df = recipients_df.copy()
        df.columns = [col.strip().lower() for col in df.columns]

        abo_column = self._first_existing_column(df, ["abo", "blood_type", "recipient_abo"])
        if abo_column and kidney.donor_abo:
            donor_abo = kidney.donor_abo.upper()
            df = df[df[abo_column].astype(str).str.upper().str.contains(donor_abo[:1], na=False)]

        score = pd.Series(0.0, index=df.index)
        urgency_col = self._first_existing_column(df, ["urgency", "status_score", "priority_score"])
        wait_col = self._first_existing_column(df, ["wait_days", "waiting_days", "wait_time"])
        distance_col = self._first_existing_column(df, ["distance_km", "distance_miles"])

        if urgency_col:
            score += df[urgency_col].fillna(0).astype(float) * 2.0
        if wait_col:
            score += df[wait_col].fillna(0).astype(float) * 0.5
        if distance_col:
            score -= df[distance_col].fillna(0).astype(float) * 0.2

        if score.empty:
            top_candidate = None
        else:
            df = df.assign(score=score)
            top_candidate = df.sort_values("score", ascending=False).head(1).to_dict(orient="records")[0]

        return {
            "status": "ranked",
            "message": "Recipient ranking completed.",
            "top_candidate": top_candidate,
            "scored_table": df.sort_values("score", ascending=False).head(10),
        }

    def _recommend_preservation(self, kidney: KidneyIntake, pumps_df: pd.DataFrame | None) -> dict[str, Any]:
        recommendation = {
            "method": kidney.preservation_method or "Static cold storage",
            "reason": "Defaulting to current preservation method.",
            "pump_available": None,
        }

        warm_ischemia = kidney.warm_ischemia_minutes or 0
        if pumps_df is not None and not pumps_df.empty:
            pump_cols = [col.strip().lower() for col in pumps_df.columns]
            pumps_df.columns = pump_cols
            available_col = self._first_existing_column(pumps_df, ["available", "status", "in_service"])
            if available_col:
                available = pumps_df[pumps_df[available_col].astype(str).str.contains("yes|available|true", case=False)]
                recommendation["pump_available"] = not available.empty

        if warm_ischemia >= 20 or (kidney.donor_type or "").lower() == "dcd":
            if recommendation["pump_available"]:
                recommendation.update(
                    {
                        "method": "Hypothermic machine perfusion",
                        "reason": (
                            "DCD graft with elevated warm ischemia time; pump perfusion can mitigate ischemic injury."
                        ),
                    }
                )
            else:
                recommendation.update(
                    {
                        "method": "Static cold storage",
                        "reason": (
                            "Elevated warm ischemia suggests perfusion is ideal, but no pump availability found."
                        ),
                    }
                )

        return recommendation

    def _fetch_evidence_snippets(self, kidney: KidneyIntake) -> list[dict[str, str]]:
        snippets: list[dict[str, str]] = []
        try:
            pubmed = self._query_pubmed("kidney cold ischemia machine perfusion outcomes")
            snippets.extend(pubmed)
        except requests.RequestException:
            pass
        return snippets

    def _generate_llm_summary(
        self,
        message: str,
        kidney: KidneyIntake,
        recipient_recommendation: dict[str, Any],
        preservation_recommendation: dict[str, Any],
        evidence: list[dict[str, str]],
    ) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "OPENAI_API_KEY not set; skipping LLM summary."

        client = OpenAI(api_key=api_key)
        model = os.getenv("OPENAI_MODEL", "gpt-4")
        evidence_summary = "\n".join(f"- {item['title']} ({item['url']})" for item in evidence) or "None"

        prompt = (
            "You are a transplant coordinator assistant. Provide a concise recommendation for kidney allocation "
            "and preservation based on the provided data. Highlight the top recipient and preservation method, "
            "and note key risks. Use clear medical language.\n\n"
            f"Kidney intake summary: {kidney.raw_text}\n\n"
            f"Recipient recommendation data: {recipient_recommendation}\n\n"
            f"Preservation recommendation data: {preservation_recommendation}\n\n"
            f"Evidence snippets: {evidence_summary}\n"
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert transplant allocation assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    def _build_markdown_report(
        self,
        kidney: KidneyIntake,
        recipient_recommendation: dict[str, Any],
        preservation_recommendation: dict[str, Any],
        evidence: list[dict[str, str]],
        narrative: str,
    ) -> str:
        recipient = recipient_recommendation.get("top_candidate")
        recipient_lines = "\n".join(
            f"- **{key.replace('_', ' ').title()}**: {value}" for key, value in (recipient or {}).items()
        ) or "No recipient data available."

        evidence_lines = "\n".join(
            f"- [{item['title']}]({item['url']})" for item in evidence
        ) or "No evidence retrieved."

        return (
            "# Kidney Allocation Recommendation\n\n"
            "## Kidney Intake Summary\n"
            f"- **Donor ABO**: {kidney.donor_abo or 'Unknown'}\n"
            f"- **Donor Type**: {kidney.donor_type or 'Unknown'}\n"
            f"- **Warm Ischemia (minutes)**: {kidney.warm_ischemia_minutes or 'Unknown'}\n"
            f"- **Cross Clamp Time**: {kidney.cross_clamp_time or 'Unknown'}\n"
            f"- **Cold Flush Time**: {kidney.cold_flush_time or 'Unknown'}\n"
            f"- **Location**: {kidney.location or 'Unknown'}\n"
            f"- **Current Preservation**: {kidney.preservation_method or 'Unknown'}\n\n"
            "## Recommended Recipient\n"
            f"{recipient_lines}\n\n"
            "## Preservation Recommendation\n"
            f"- **Suggested Method**: {preservation_recommendation['method']}\n"
            f"- **Rationale**: {preservation_recommendation['reason']}\n"
            f"- **Pump Available**: {preservation_recommendation['pump_available']}\n\n"
            "## Evidence Snapshot\n"
            f"{evidence_lines}\n\n"
            "## Narrative Summary\n"
            f"{narrative}\n"
        )

    def _render_pdf(self, markdown_text: str) -> bytes:
        lines = markdown_text.splitlines()
        content_lines = [self._escape_pdf_text(line) for line in lines if line.strip()]
        text_stream = [
            "BT",
            "/F1 11 Tf",
            "14 TL",
            "72 720 Td",
        ]
        for line in content_lines:
            text_stream.append(f"({line}) Tj")
            text_stream.append("T*")
        text_stream.append("ET")
        content = "\n".join(text_stream)
        content_bytes = content.encode("latin-1", errors="replace")

        objects: list[bytes] = []
        objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
        objects.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
        objects.append(
            b"3 0 obj\n"
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\n"
            b"endobj\n"
        )
        objects.append(
            f"4 0 obj\n<< /Length {len(content_bytes)} >>\nstream\n".encode("latin-1")
            + content_bytes
            + b"\nendstream\nendobj\n"
        )
        objects.append(b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")

        xref_positions = []
        pdf_body = b"%PDF-1.4\n"
        for obj in objects:
            xref_positions.append(len(pdf_body))
            pdf_body += obj

        xref_start = len(pdf_body)
        xref_entries = [b"0000000000 65535 f \n"]
        for pos in xref_positions:
            xref_entries.append(f"{pos:010d} 00000 n \n".encode("latin-1"))
        xref_table = b"xref\n0 6\n" + b"".join(xref_entries)
        trailer = (
            b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n"
            + str(xref_start).encode("latin-1")
            + b"\n%%EOF"
        )
        return pdf_body + xref_table + trailer

    def _escape_pdf_text(self, text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    def _query_pubmed(self, query: str) -> list[dict[str, str]]:
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": 3,
        }
        response = requests.get(base, params=params, timeout=10)
        response.raise_for_status()
        ids = response.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        summary_response = requests.get(
            summary_url,
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
            timeout=10,
        )
        summary_response.raise_for_status()
        summaries = summary_response.json().get("result", {})
        snippets = []
        for pmid in ids:
            item = summaries.get(pmid)
            if not item:
                continue
            snippets.append(
                {
                    "title": item.get("title", "PubMed Article"),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                }
            )
        return snippets

    @staticmethod
    def _extract_field(payload: dict[str, Any], keys: Iterable[str]) -> str | None:
        for key in keys:
            if key in payload and payload[key] is not None:
                return str(payload[key])
        return None

    @staticmethod
    def _parse_int(value: str | None) -> int | None:
        if value is None:
            return None
        try:
            return int(float(value))
        except ValueError:
            return None

    @staticmethod
    def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
        for col in candidates:
            if col in df.columns:
                return col
        return None


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
