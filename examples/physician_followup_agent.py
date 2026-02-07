"""
Physician Follow-Up Agent

An agent that processes SOAP notes and clinical documents to generate a physician
TODO list for the next visit, including summaries of new labs, radiology, and studies.

This example demonstrates:
- Filtering documents by type (SOAP notes, labs, radiology, studies)
- Using OpenAI for document summarization and TODO generation
- Processing clinical documents with date-based filtering
- Writing structured output documents

Environment Variables:
    OPENAI_API_KEY: OpenAI API key for GPT-4 summarization

Usage:
    OPENAI_API_KEY=sk-... python examples/physician_followup_agent.py

Then send a message with the patient context (e.g., "Generate follow-up summary")
"""

import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from openai import AsyncOpenAI

from health_universe_a2a import Agent, AgentContext, Document, UpdateImportance


@dataclass
class ClinicalDocument:
    """Parsed clinical document with metadata."""

    id: str
    name: str
    filename: str
    doc_type: str  # "soap_note", "lab", "radiology", "study", "other"
    date: datetime | None
    content: str


class PhysicianFollowUpAgent(Agent):
    """
    Analyzes SOAP notes and clinical documents to generate a follow-up TODO list.

    The agent:
    1. Identifies and retrieves all SOAP notes from the thread
    2. Extracts Assessment and Plan sections from each note
    3. Finds labs, radiology, and studies performed since the last visit
    4. Uses OpenAI to generate a consolidated TODO list and summary
    5. Writes the output as a structured document
    """

    def __init__(self) -> None:
        super().__init__()
        self.openai_client: AsyncOpenAI | None = None

    def get_agent_name(self) -> str:
        return "Physician Follow-Up Agent"

    def get_agent_description(self) -> str:
        return (
            "Analyzes SOAP notes and clinical documents to generate a physician "
            "TODO list for the next visit, including summaries of new labs, "
            "radiology reports, and studies."
        )

    async def on_startup(self) -> None:
        """Initialize OpenAI client on startup."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.logger.warning("OPENAI_API_KEY not set - agent will fail at runtime")
        self.openai_client = AsyncOpenAI(api_key=api_key)

    async def process_message(self, message: str, context: AgentContext) -> str:
        """
        Process clinical documents and generate follow-up summary.

        Args:
            message: User message (can contain instructions or patient context)
            context: Agent context with document_client for file operations

        Returns:
            Markdown-formatted follow-up summary and TODO list
        """
        if not self.openai_client:
            return "Error: OpenAI API key not configured. Set OPENAI_API_KEY environment variable."

        await context.update_progress(
            "Scanning thread for clinical documents...",
            progress=0.1,
            importance=UpdateImportance.INFO,
        )

        # Step 1: Retrieve and classify all documents
        try:
            docs = await context.document_client.list_documents()
        except Exception as e:
            return f"Error accessing documents: {e}"

        if not docs:
            return "No documents found in this thread. Please upload SOAP notes and clinical documents."

        await context.update_progress(
            f"Found {len(docs)} documents, classifying...",
            progress=0.2,
            importance=UpdateImportance.INFO,
        )

        # Step 2: Classify and download documents
        clinical_docs = await self._classify_and_download_documents(docs, context)

        soap_notes = [d for d in clinical_docs if d.doc_type == "soap_note"]
        labs = [d for d in clinical_docs if d.doc_type == "lab"]
        radiology = [d for d in clinical_docs if d.doc_type == "radiology"]
        studies = [d for d in clinical_docs if d.doc_type == "study"]

        await context.update_progress(
            f"Found {len(soap_notes)} SOAP notes, {len(labs)} labs, "
            f"{len(radiology)} radiology, {len(studies)} studies",
            progress=0.4,
            importance=UpdateImportance.INFO,
        )

        if not soap_notes:
            return (
                "No SOAP notes found in this thread. Please upload SOAP notes "
                "to generate a follow-up summary."
            )

        # Step 3: Sort by date and find the most recent visit
        soap_notes.sort(key=lambda x: x.date or datetime.min, reverse=True)
        most_recent_soap = soap_notes[0]
        last_visit_date = most_recent_soap.date

        # Filter results to those after the previous visit (if we have date info)
        if last_visit_date and len(soap_notes) > 1:
            previous_visit_date = soap_notes[1].date
            if previous_visit_date:
                labs = [d for d in labs if d.date and d.date > previous_visit_date]
                radiology = [d for d in radiology if d.date and d.date > previous_visit_date]
                studies = [d for d in studies if d.date and d.date > previous_visit_date]

        await context.update_progress(
            "Extracting Assessment and Plan from SOAP notes...",
            progress=0.5,
            importance=UpdateImportance.INFO,
        )

        # Step 4: Extract A&P sections from SOAP notes
        assessments_and_plans = []
        for note in soap_notes:
            ap_section = self._extract_assessment_and_plan(note.content)
            if ap_section:
                assessments_and_plans.append(
                    {
                        "date": note.date.isoformat() if note.date else "Unknown",
                        "name": note.name,
                        "assessment_and_plan": ap_section,
                    }
                )

        await context.update_progress(
            "Generating follow-up summary with OpenAI...",
            progress=0.6,
            importance=UpdateImportance.INFO,
        )

        # Step 5: Generate TODO list using OpenAI
        todo_list = await self._generate_todo_list(
            assessments_and_plans=assessments_and_plans,
            labs=labs,
            radiology=radiology,
            studies=studies,
            user_message=message,
        )

        await context.update_progress(
            "Generating results summary...",
            progress=0.8,
            importance=UpdateImportance.INFO,
        )

        # Step 6: Generate results summaries
        results_summary = await self._generate_results_summary(
            labs=labs,
            radiology=radiology,
            studies=studies,
        )

        # Step 7: Compile final output
        output = self._compile_output(
            todo_list=todo_list,
            results_summary=results_summary,
            soap_notes=soap_notes,
            labs=labs,
            radiology=radiology,
            studies=studies,
            last_visit_date=last_visit_date,
        )

        # Step 8: Write output document
        await context.update_progress(
            "Writing follow-up summary document...",
            progress=0.9,
            importance=UpdateImportance.INFO,
        )

        try:
            await context.document_client.write(
                name="Physician Follow-Up Summary",
                content=output,
                filename="followup_summary.md",
            )
        except Exception as e:
            self.logger.warning(f"Failed to write output document: {e}")

        await context.update_progress(
            "Follow-up summary complete!",
            progress=1.0,
            importance=UpdateImportance.NOTICE,
        )

        return output

    async def _classify_and_download_documents(
        self, docs: list[Document], context: AgentContext
    ) -> list[ClinicalDocument]:
        """
        Classify documents by type and download their content.

        Uses filename patterns and content analysis to classify documents as:
        - soap_note: SOAP notes, progress notes, H&P
        - lab: Laboratory results
        - radiology: Imaging reports (X-ray, CT, MRI, ultrasound)
        - study: Other clinical studies (EKG, spirometry, etc.)
        - other: Unclassified documents
        """
        clinical_docs: list[ClinicalDocument] = []

        for doc in docs:
            try:
                content = await context.document_client.download_text(doc.id)
            except Exception as e:
                self.logger.warning(f"Failed to download {doc.name}: {e}")
                continue

            doc_type = self._classify_document(doc.name, doc.filename, content)
            doc_date = self._extract_date(doc.name, doc.filename, content)

            clinical_docs.append(
                ClinicalDocument(
                    id=doc.id,
                    name=doc.name,
                    filename=doc.filename,
                    doc_type=doc_type,
                    date=doc_date,
                    content=content,
                )
            )

        return clinical_docs

    def _classify_document(self, name: str, filename: str, content: str) -> str:
        """Classify a document based on name, filename, and content patterns."""
        name_lower = name.lower()
        filename_lower = filename.lower()
        content_lower = content[:2000].lower()  # Check first 2000 chars

        # SOAP notes / Progress notes
        soap_patterns = [
            "soap",
            "progress note",
            "office visit",
            "h&p",
            "history and physical",
            "encounter note",
            "clinic note",
            "subjective",
            "chief complaint",
        ]
        if any(p in name_lower or p in filename_lower or p in content_lower for p in soap_patterns):
            # Verify it has SOAP structure
            if any(
                section in content_lower
                for section in ["assessment", "plan", "subjective", "objective"]
            ):
                return "soap_note"

        # Lab results
        lab_patterns = [
            "lab",
            "laboratory",
            "cbc",
            "cmp",
            "bmp",
            "lipid",
            "a1c",
            "hemoglobin",
            "glucose",
            "creatinine",
            "bun",
            "electrolyte",
            "thyroid",
            "tsh",
            "urinalysis",
            "culture",
        ]
        if any(p in name_lower or p in filename_lower or p in content_lower for p in lab_patterns):
            return "lab"

        # Radiology
        radiology_patterns = [
            "x-ray",
            "xray",
            "ct scan",
            "ct ",
            "mri",
            "ultrasound",
            "sonogram",
            "mammogram",
            "dexa",
            "bone density",
            "radiology",
            "imaging",
            "radiograph",
        ]
        if any(
            p in name_lower or p in filename_lower or p in content_lower for p in radiology_patterns
        ):
            return "radiology"

        # Other studies
        study_patterns = [
            "ekg",
            "ecg",
            "electrocardiogram",
            "echo",
            "echocardiogram",
            "spirometry",
            "pulmonary function",
            "pft",
            "stress test",
            "holter",
            "sleep study",
            "colonoscopy",
            "endoscopy",
            "biopsy",
        ]
        if any(
            p in name_lower or p in filename_lower or p in content_lower for p in study_patterns
        ):
            return "study"

        return "other"

    def _extract_date(self, name: str, filename: str, content: str) -> datetime | None:
        """Extract date from document metadata or content."""
        # Common date patterns
        date_patterns = [
            r"(\d{1,2})/(\d{1,2})/(\d{4})",  # MM/DD/YYYY
            r"(\d{4})-(\d{2})-(\d{2})",  # YYYY-MM-DD
            r"(\d{1,2})-(\d{1,2})-(\d{4})",  # MM-DD-YYYY
            r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* (\d{1,2}),? (\d{4})",
        ]

        # Check filename first
        for text in [filename, name, content[:500]]:
            for pattern in date_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        groups = match.groups()
                        if len(groups) == 3:
                            if groups[0].isdigit() and int(groups[0]) > 1900:
                                # YYYY-MM-DD
                                return datetime(int(groups[0]), int(groups[1]), int(groups[2]))
                            elif groups[2].isdigit() and int(groups[2]) > 1900:
                                # MM/DD/YYYY or MM-DD-YYYY
                                return datetime(int(groups[2]), int(groups[0]), int(groups[1]))
                            elif not groups[0].isdigit():
                                # Month name format
                                month_map = {
                                    "jan": 1,
                                    "feb": 2,
                                    "mar": 3,
                                    "apr": 4,
                                    "may": 5,
                                    "jun": 6,
                                    "jul": 7,
                                    "aug": 8,
                                    "sep": 9,
                                    "oct": 10,
                                    "nov": 11,
                                    "dec": 12,
                                }
                                month = month_map.get(groups[0][:3].lower())
                                if month:
                                    return datetime(int(groups[2]), month, int(groups[1]))
                    except (ValueError, TypeError):
                        continue

        return None

    def _extract_assessment_and_plan(self, content: str) -> str | None:
        """Extract the Assessment and Plan section from a SOAP note."""
        content_lower = content.lower()

        # Find Assessment section
        assessment_start = -1
        for marker in ["assessment:", "assessment and plan:", "a/p:", "a&p:", "impression:"]:
            idx = content_lower.find(marker)
            if idx != -1:
                assessment_start = idx
                break

        if assessment_start == -1:
            # Try to find just "assessment" as a section header
            for marker in ["\nassessment\n", "\nassessment ", "**assessment**"]:
                idx = content_lower.find(marker)
                if idx != -1:
                    assessment_start = idx
                    break

        if assessment_start == -1:
            return None

        # Find the end of the A&P section (next major section or end of document)
        section_markers = [
            "\nfollow",
            "\nreturn",
            "\nsignature",
            "\n---",
            "\n___",
            "electronically signed",
        ]

        assessment_end = len(content)
        for marker in section_markers:
            idx = content_lower.find(marker, assessment_start + 10)
            if idx != -1 and idx < assessment_end:
                assessment_end = idx

        return content[assessment_start:assessment_end].strip()

    async def _generate_todo_list(
        self,
        assessments_and_plans: list[dict[str, Any]],
        labs: list[ClinicalDocument],
        radiology: list[ClinicalDocument],
        studies: list[ClinicalDocument],
        user_message: str,
    ) -> str:
        """Use OpenAI to generate a TODO list from the clinical data."""
        if not self.openai_client:
            return "Error: OpenAI client not initialized"

        # Build the prompt
        ap_text = ""
        for ap in assessments_and_plans:
            ap_text += f"\n### SOAP Note from {ap['date']}:\n{ap['assessment_and_plan']}\n"

        results_text = ""
        if labs or radiology or studies:
            results_text = "\n## New Results Since Last Visit:\n"
            for lab in labs:
                results_text += f"\n### Lab: {lab.name} ({lab.date or 'Unknown date'})\n"
                results_text += lab.content[:1500] + ("..." if len(lab.content) > 1500 else "")
            for rad in radiology:
                results_text += f"\n### Radiology: {rad.name} ({rad.date or 'Unknown date'})\n"
                results_text += rad.content[:1500] + ("..." if len(rad.content) > 1500 else "")
            for study in studies:
                results_text += f"\n### Study: {study.name} ({study.date or 'Unknown date'})\n"
                results_text += study.content[:1500] + ("..." if len(study.content) > 1500 else "")

        prompt = f"""You are a clinical assistant helping a physician prepare for a patient follow-up visit.

Based on the following Assessment and Plan sections from previous SOAP notes, and any new test results, generate a comprehensive TODO list for the physician's next visit with this patient.

## Assessment and Plan from Previous Visits:
{ap_text}

{results_text}

{f"Additional context from physician: {user_message}" if user_message else ""}

Please generate a structured TODO list that includes:
1. **Follow-up Items**: Issues from previous visits that need follow-up
2. **Medication Review**: Any medications that need adjustment or monitoring
3. **Results to Discuss**: New lab/imaging results that need to be reviewed with the patient
4. **Pending Actions**: Any tests, referrals, or actions that were planned but may not be complete
5. **Preventive Care**: Any preventive care items due based on the clinical context

Format the output as a clear, actionable markdown checklist that the physician can use during the visit."""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a clinical documentation assistant. Generate clear, "
                        "actionable TODO lists for physician follow-up visits.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2000,
            )
            return response.choices[0].message.content or "Failed to generate TODO list"
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            return f"Error generating TODO list: {e}"

    async def _generate_results_summary(
        self,
        labs: list[ClinicalDocument],
        radiology: list[ClinicalDocument],
        studies: list[ClinicalDocument],
    ) -> str:
        """Generate a summary of new results using OpenAI."""
        if not self.openai_client:
            return ""

        if not labs and not radiology and not studies:
            return "No new labs, radiology, or studies since the last visit."

        results_text = ""
        for lab in labs:
            results_text += f"\n### Lab: {lab.name}\n{lab.content}\n"
        for rad in radiology:
            results_text += f"\n### Radiology: {rad.name}\n{rad.content}\n"
        for study in studies:
            results_text += f"\n### Study: {study.name}\n{study.content}\n"

        prompt = f"""Summarize the following clinical results for a physician's quick review.
Highlight any abnormal values, significant findings, or items requiring attention.

{results_text}

Provide a concise summary organized by:
1. **Abnormal Findings**: Any values outside normal range or concerning findings
2. **Notable Results**: Important findings even if within normal limits
3. **Trending Values**: Any values that may be trending in a concerning direction"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a clinical assistant summarizing test results for physicians.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=1500,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            return f"Error summarizing results: {e}"

    def _compile_output(
        self,
        todo_list: str,
        results_summary: str,
        soap_notes: list[ClinicalDocument],
        labs: list[ClinicalDocument],
        radiology: list[ClinicalDocument],
        studies: list[ClinicalDocument],
        last_visit_date: datetime | None,
    ) -> str:
        """Compile the final output document."""
        date_str = last_visit_date.strftime("%B %d, %Y") if last_visit_date else "Unknown"

        output = f"""# Physician Follow-Up Summary

**Generated:** {datetime.now().strftime("%B %d, %Y %I:%M %p")}
**Last Visit:** {date_str}

---

## Documents Analyzed

| Category | Count |
|----------|-------|
| SOAP Notes | {len(soap_notes)} |
| Lab Results | {len(labs)} |
| Radiology Reports | {len(radiology)} |
| Other Studies | {len(studies)} |

---

## TODO List for Next Visit

{todo_list}

---

## Summary of Results Since Last Visit

{results_summary}

---

## Source Documents

### SOAP Notes Reviewed:
"""
        for note in soap_notes:
            date_display = note.date.strftime("%Y-%m-%d") if note.date else "Unknown date"
            output += f"- {note.name} ({date_display})\n"

        if labs:
            output += "\n### Lab Results:\n"
            for lab in labs:
                date_display = lab.date.strftime("%Y-%m-%d") if lab.date else "Unknown date"
                output += f"- {lab.name} ({date_display})\n"

        if radiology:
            output += "\n### Radiology Reports:\n"
            for rad in radiology:
                date_display = rad.date.strftime("%Y-%m-%d") if rad.date else "Unknown date"
                output += f"- {rad.name} ({date_display})\n"

        if studies:
            output += "\n### Other Studies:\n"
            for study in studies:
                date_display = study.date.strftime("%Y-%m-%d") if study.date else "Unknown date"
                output += f"- {study.name} ({date_display})\n"

        output += """
---

*This summary was generated automatically. Please verify all information against source documents.*
"""
        return output


if __name__ == "__main__":
    agent = PhysicianFollowUpAgent()
    agent.serve(port=8003)
