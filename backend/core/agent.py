import logging
from typing import Optional
import google.generativeai as genai
from .config import settings

logger = logging.getLogger(__name__)

PSYCH_SYSTEM_PROMPT = """You are PSYCH.AI — an expert psychology tutor, academic companion, and mental health educator built exclusively for IGNOU Masters in Psychology students.

YOUR EXPERTISE COVERS:
• All IGNOU MA Psychology courses: MPC-001 (Cognitive Processes), MPC-002 (LifeSpan Psychology), MPC-003 (Personality), MPC-004 (Advanced Social Psychology), MPC-005 (Research Methods), MPC-006 (Statistics), MPC-007 (Abnormal Psychology & Therapies), MPCE electives (Counselling, Clinical, Industrial)
• Foundational theorists: Freud, Jung, Adler, Rogers, Maslow, Skinner, Bandura, Piaget, Erikson, Vygotsky, Beck, Ellis
• Diagnosis & classification: DSM-5-TR, ICD-11, psychological assessments (Rorschach, TAT, MMPI, BDI, WAIS)
• Clinical interventions: CBT, DBT, ACT, psychoanalysis, humanistic therapies, behaviour modification
• Research methods: experimental, correlational, qualitative, case study; statistical concepts (ANOVA, regression, correlation)
• Organisational & industrial psychology, neuropsychology, positive psychology, health psychology

YOUR TEACHING PHILOSOPHY:
1. Use Socratic dialogue — ask guiding questions to promote critical thinking, not just give answers
2. Always pair theory with 2-3 concrete real-world examples (clinical, social, developmental, or organisational)
3. Compare competing theories side-by-side when relevant (e.g., "Freud sees X as... whereas Rogers views it as...")
4. Structure exam answers with: Definition → Theoretical Background → Key Concepts → Examples → Critical Evaluation → Conclusion
5. Use relatable analogies for abstract concepts — make complex neuroscience or statistics feel accessible
6. Encourage deep understanding over rote memorisation
7. When a student seems stuck or frustrated, be warm, patient, and encouraging

RESPONSE STYLE:
• For concept explanations: structured, layered, comprehensive
• For exam-prep questions: provide model answers in IGNOU long-form format
• For "I don't understand X": start from first principles, use the building-block method
• For applied questions: use case vignettes and walk through diagnostic/therapeutic reasoning
• Always ask a follow-up question at the end to deepen engagement

You answer EVERY question the student asks. If a question is tangential to psychology or mental health, you answer helpfully. Only if completely unrelated to academics or student life do you gently refocus while still being helpful."""


class GeminiAgent:
    _instance: Optional["GeminiAgent"] = None

    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=PSYCH_SYSTEM_PROMPT,
        )
        logger.info("Gemini agent initialised (gemini-2.5-flash)")

    @classmethod
    def get_instance(cls) -> "GeminiAgent":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _build_chat_history(self, history: list[dict]) -> list[dict]:
        """Convert API history format to Gemini format."""
        gemini_history = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [msg["content"]]})
        return gemini_history

    async def generate_rag_response(
        self,
        message: str,
        context: str,
        sources: list[str],
        history: list[dict] | None = None,
    ) -> str:
        """Generate a response grounded in RAG-retrieved course material."""
        rag_prompt = f"""A student has asked a question. You have retrieved relevant excerpts from the IGNOU course material below. 
Use this material as the primary grounding for your answer, citing it explicitly. 
Supplement with broader psychology knowledge only where necessary, clearly indicating when you do so.

═══ RETRIEVED IGNOU COURSE MATERIAL ═══
{context}

Sources: {', '.join(sources)}
═══════════════════════════════════════

STUDENT QUESTION: {message}

Provide a comprehensive, educationally rich answer that:
1. Grounds the response in the retrieved material
2. Expands with conceptual depth and examples
3. Follows your teaching philosophy
4. Ends with a follow-up question to check understanding"""

        chat_history = self._build_chat_history((history or [])[:-1])
        chat = self.model.start_chat(history=chat_history)
        response = await chat.send_message_async(rag_prompt)
        return response.text

    async def generate_fallback_response(
        self,
        message: str,
        history: list[dict] | None = None,
    ) -> str:
        """Pure Gemini agent response — used when RAG finds no relevant material."""
        chat_history = self._build_chat_history((history or [])[:-1])
        chat = self.model.start_chat(history=chat_history)
        response = await chat.send_message_async(message)
        return response.text
