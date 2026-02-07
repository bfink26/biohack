"""
Medical Symptom Classifier Agent

A simple agent that classifies medical symptoms into urgency categories.
This example demonstrates:
- Basic agent structure with Agent/AgentContext
- Progress updates during processing
- No document operations (simple input/output)

Usage:
    python examples/medical_classifier.py

Then send a message like:
    "Patient reports severe chest pain, shortness of breath, and dizziness"
"""

import json

from health_universe_a2a import Agent, AgentContext


class SymptomClassifierAgent(Agent):
    """
    Classifies medical symptoms into urgency categories.

    Urgency Levels:
    - EMERGENCY: Symptoms requiring immediate medical attention (call 911)
    - URGENT: Should see a doctor today
    - ROUTINE: Can schedule a regular appointment
    - SELF_CARE: Can likely be managed at home
    """

    # Emergency symptoms that require immediate attention
    EMERGENCY_KEYWORDS = [
        "chest pain",
        "difficulty breathing",
        "shortness of breath",
        "severe bleeding",
        "unconscious",
        "stroke",
        "heart attack",
        "seizure",
        "severe allergic",
        "anaphylaxis",
        "paralysis",
        "sudden numbness",
        "slurred speech",
    ]

    # Urgent symptoms that need same-day care
    URGENT_KEYWORDS = [
        "high fever",
        "persistent vomiting",
        "severe pain",
        "blood in urine",
        "blood in stool",
        "severe headache",
        "dehydration",
        "broken bone",
        "deep cut",
        "burns",
    ]

    # Routine symptoms for regular appointments
    ROUTINE_KEYWORDS = [
        "persistent cough",
        "rash",
        "mild fever",
        "earache",
        "joint pain",
        "fatigue",
        "weight changes",
        "sleep problems",
        "mild pain",
    ]

    def get_agent_name(self) -> str:
        return "Medical Symptom Classifier"

    def get_agent_description(self) -> str:
        return (
            "Classifies medical symptoms into urgency categories "
            "(Emergency, Urgent, Routine, Self-Care) and provides recommendations."
        )

    async def process_message(self, message: str, context: AgentContext) -> str:
        """
        Process symptom description and return urgency classification.

        Args:
            message: Description of symptoms
            context: Agent context for progress updates

        Returns:
            JSON string with urgency level and recommendations
        """
        await context.update_progress("Analyzing symptoms...", 0.2)

        # Normalize message for matching
        message_lower = message.lower()

        # Check for emergency symptoms
        emergency_matches = self._find_matches(message_lower, self.EMERGENCY_KEYWORDS)
        if emergency_matches:
            await context.update_progress("Emergency symptoms detected", 0.8)
            return self._format_response(
                urgency="EMERGENCY",
                matched_symptoms=emergency_matches,
                recommendation="Call 911 or go to the emergency room immediately. "
                "These symptoms require immediate medical attention.",
                message=message,
            )

        # Check for urgent symptoms
        urgent_matches = self._find_matches(message_lower, self.URGENT_KEYWORDS)
        if urgent_matches:
            await context.update_progress("Urgent symptoms detected", 0.8)
            return self._format_response(
                urgency="URGENT",
                matched_symptoms=urgent_matches,
                recommendation="Please see a doctor today or visit an urgent care clinic. "
                "Do not wait for a regular appointment.",
                message=message,
            )

        # Check for routine symptoms
        routine_matches = self._find_matches(message_lower, self.ROUTINE_KEYWORDS)
        if routine_matches:
            await context.update_progress("Routine symptoms detected", 0.8)
            return self._format_response(
                urgency="ROUTINE",
                matched_symptoms=routine_matches,
                recommendation="Schedule an appointment with your primary care physician. "
                "These symptoms should be evaluated but are not emergent.",
                message=message,
            )

        # Default to self-care
        await context.update_progress("Classification complete", 1.0)
        return self._format_response(
            urgency="SELF_CARE",
            matched_symptoms=[],
            recommendation="Based on the described symptoms, self-care may be appropriate. "
            "Monitor symptoms and seek medical attention if they worsen or persist.",
            message=message,
        )

    def _find_matches(self, text: str, keywords: list[str]) -> list[str]:
        """Find which keywords are present in the text."""
        matches = []
        for keyword in keywords:
            if keyword in text:
                matches.append(keyword)
        return matches

    def _format_response(
        self,
        urgency: str,
        matched_symptoms: list[str],
        recommendation: str,
        message: str,
    ) -> str:
        """Format the classification response as JSON."""
        response = {
            "urgency": urgency,
            "matched_symptoms": matched_symptoms,
            "recommendation": recommendation,
            "disclaimer": "This is an automated classification and should not replace "
            "professional medical advice. When in doubt, always consult a healthcare provider.",
            "original_message": message[:200] + ("..." if len(message) > 200 else ""),
        }
        return json.dumps(response, indent=2)


if __name__ == "__main__":
    agent = SymptomClassifierAgent()
    agent.serve(port=8001)
