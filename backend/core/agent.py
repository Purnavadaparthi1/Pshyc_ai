import logging
from typing import Optional
import asyncio
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

    def _determine_response_style(self, message: str) -> str:
        """Determine if response should be brief or detailed based on user message."""
        brief_keywords = ["brief", "summary", "short", "quick", "concise", "tl;dr"]
        detailed_keywords = ["detailed", "explain", "long", "in depth", "comprehensive", "elaborate"]
        
        message_lower = message.lower()
        if any(kw in message_lower for kw in brief_keywords):
            return "brief"
        elif any(kw in message_lower for kw in detailed_keywords):
            return "detailed"
        else:
            return "comprehensive"

    async def _retry_api_call(self, chat, prompt: str, max_retries: int = 3) -> str:
        """Retry API call with exponential backoff on 429 errors."""
        for attempt in range(max_retries):
            try:
                response = await chat.send_message_async(prompt)
                return response.text
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Rate limit hit, retrying in {wait_time}s (attempt {attempt + 1})")
                    await asyncio.sleep(wait_time)
                else:
                    raise e

    async def generate_rag_response(
        self,
        message: str,
        context: str,
        sources: list[str],
        history: list[dict] | None = None,
    ) -> str:
        """Generate a response grounded in RAG-retrieved course material."""
        style = self._determine_response_style(message)
        
        if style == "brief":
            style_instruction = "Provide a concise summary (2-3 sentences) focusing on key points from the material."
        elif style == "detailed":
            style_instruction = "Provide an in-depth, comprehensive answer with examples, comparisons, and critical analysis."
        else:
            style_instruction = "Provide a comprehensive, educationally rich answer that grounds the response in the retrieved material, expands with conceptual depth and examples, follows your teaching philosophy, and ends with a follow-up question to check understanding."
        
        rag_prompt = f"""A student has asked a question. You have retrieved relevant excerpts from the IGNOU course material below. 
Use this material as the primary grounding for your answer, citing it explicitly. 
Supplement with broader psychology knowledge only where necessary, clearly indicating when you do so.

═══ RETRIEVED IGNOU COURSE MATERIAL ═══
{context}

Sources: {', '.join(sources)}
═══════════════════════════════════════

STUDENT QUESTION: {message}

{style_instruction}"""

        chat_history = self._build_chat_history((history or [])[:-1])
        chat = self.model.start_chat(history=chat_history)
        return await self._retry_api_call(chat, rag_prompt)

    async def generate_fallback_response(
        self,
        message: str,
        history: list[dict] | None = None,
    ) -> str:
        """Pure Gemini agent response — used when RAG finds no relevant material."""
        style = self._determine_response_style(message)
        
        if style == "brief":
            fallback_prompt = f"Provide a brief answer to: {message}"
        elif style == "detailed":
            fallback_prompt = f"""Provide a detailed, comprehensive answer to: {message}

Follow these guidelines:
• Use Socratic dialogue — ask guiding questions to promote critical thinking
• Pair theory with 2-3 concrete real-world examples
• Compare competing theories when relevant
• Structure answers with: Definition → Theoretical Background → Key Concepts → Examples → Critical Evaluation → Conclusion
• Use relatable analogies for abstract concepts
• Encourage deep understanding over rote memorisation
• End with a follow-up question"""
        else:
            fallback_prompt = f"""Answer this question as PSYCH.AI, an expert psychology tutor: {message}

Follow your teaching philosophy:
• Use Socratic dialogue — ask guiding questions to promote critical thinking
• Pair theory with 2-3 concrete real-world examples
• Compare competing theories when relevant
• Structure answers with: Definition → Theoretical Background → Key Concepts → Examples → Critical Evaluation → Conclusion
• Use relatable analogies for abstract concepts
• Encourage deep understanding over rote memorisation
• End with a follow-up question"""
        
        chat_history = self._build_chat_history((history or [])[:-1])
        chat = self.model.start_chat(history=chat_history)
        return await self._retry_api_call(chat, fallback_prompt)
