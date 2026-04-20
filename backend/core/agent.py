import re
import logging
from typing import Optional
import asyncio
import google.generativeai as genai
from .config import settings

logger = logging.getLogger(__name__)

PSYCH_SYSTEM_PROMPT = """
You are an IGNOU Psychology Tutor.

Always answer in this format:
1. Title
2. Definition
3. Explanation with subheadings
4. Key points (bulleted)
5. Conclusion

IMPORTANT INSTRUCTIONS:
- Start with a clear heading
- Use proper subheadings
- Use bullet points where needed
- Do NOT dump raw text
- Rewrite the answer in clean structured format
- Do NOT repeat irrelevant content
- If the context is insufficient, say so and do not hallucinate
- For greetings (hi, hello, hey, etc.), respond warmly and briefly — introduce yourself as PSYCH.AI, IGNOU's Psychology tutor, and invite the user to ask a question
"""


class GeminiAgent:
    @staticmethod
    def is_insufficient_context(reply: str, question: str) -> bool:
        """
        Dynamically detect if the LLM reply is insufficient by checking semantic similarity
        between the reply and the question. If the reply mostly repeats the question or does not
        add new information, trigger fallback. This avoids hardcoding phrases.
        """
        import difflib
        import re
        # Remove markdown and extra formatting for comparison
        def clean(text):
            text = re.sub(r"[#*\-]+", "", text)
            text = re.sub(r"\s+", " ", text)
            return text.strip().lower()

        reply_clean = clean(reply)
        question_clean = clean(question)

        # If reply is very short or just repeats the question, fallback
        if len(reply_clean) < 40:
            logger.info("Fallback: Gemini reply too short to be a valid answer.")
            return True

        # If reply contains the question verbatim or is highly similar, fallback
        seq = difflib.SequenceMatcher(None, reply_clean, question_clean)
        similarity = seq.ratio()
        if similarity > 0.7:
            logger.info(f"Fallback: Gemini reply too similar to question (similarity={similarity:.2f}).")
            return True

        # If reply contains phrases indicating lack of info, fallback
        fallback_patterns = [
            r"insufficient(ly)? (provided )?context",
            r"not enough context",
            r"no relevant content",
            r"context does not provide",
            r"sorry[\w\s,]*no relevant content",
            r"cannot (be generated|answer|respond) (from|using|with) (the )?(provided|given)? ?context",
            r"not (mentioned|explained|covered|found|present|available) (within|in|by|from|the)? (the )?(given|provided|current)? ?context",
            r"the provided text discusses",
            r"however,? specific (techniques|topics|concepts|details)",
            r"context (does not|is not|was not|cannot|fails to) (provide|cover|contain|include|address|support)",
            r"context (is|was)? ?(missing|insufficient|irrelevant|incomplete|lacking)",
            r"no information (available|found|present) (in|within|from|by|the)? (the )?context",
            r"not (present|covered|found|included) in (the )?context",
            r"there is no (information|guidance|answer|content) (in|within|from|by|the)? (the )?context",
            r"cannot answer (from|using|with) (the )?context",
            r"the context does not (contain|cover|provide|include|address)"
        ]
        for pattern in fallback_patterns:
            if re.search(pattern, reply_clean):
                logger.info(f"Fallback triggered by pattern: '{pattern}' in reply: {reply_clean}")
                return True

        return False

    # Token management for rate limiting
    MAX_TOKENS_PER_MIN = 60000  # Adjust based on your Gemini API quota
    TOKEN_SAFETY_MARGIN = 0.9   # Use only up to 90% of quota
    _tokens_used = 0
    _last_reset = None
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

    def _estimate_tokens(self, text: str) -> int:
        # Rough estimate: 1 token ≈ 4 chars (for English)
        return max(1, len(text) // 4)

    def _reset_token_counter_if_needed(self):
        import time
        now = time.time()
        if self._last_reset is None or now - self._last_reset > 60:
            self._tokens_used = 0
            self._last_reset = now

    def _can_make_request(self, estimated_tokens: int) -> bool:
        self._reset_token_counter_if_needed()
        return (self._tokens_used + estimated_tokens) < self.MAX_TOKENS_PER_MIN * self.TOKEN_SAFETY_MARGIN

    def clean_context(self, text: str) -> str:
        # Fix broken words (e.g., "behavioUr" → "behavio Ur")
        text = re.sub(r'(?<=\w)(?=[A-Z])', ' ', text)
        # Remove weird page artifacts
        text = re.sub(r'\d+Theories of Personality-I', '', text)
        # Remove duplicate sentences
        sentences = text.split('. ')
        seen = set()
        cleaned = []
        for s in sentences:
            if s not in seen:
                cleaned.append(s)
                seen.add(s)
        text = '. '.join(cleaned)
        # Normalize spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def post_process(self, text: str) -> str:
        # Remove repeated lines
        lines = text.split('\n')
        seen = set()
        result = []
        for line in lines:
            if line.strip() not in seen:
                result.append(line)
                seen.add(line.strip())
        return "\n".join(result)

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
        """Retry API call with exponential backoff on 429 errors, and exit early if close to rate limit."""
        estimated_tokens = self._estimate_tokens(prompt)
        if not self._can_make_request(estimated_tokens):
            logger.warning("Approaching Gemini API rate limit. Returning partial answer.")
            # Return a polite, structured partial answer
            return (
                "## Partial Answer\n"
                "### Rate Limit Notice\n"
                "- The Gemini API rate limit is nearly reached.\n"
                "- Please wait a minute and try again for a more complete answer.\n"
                "- Here is as much as could be generated without exceeding the limit.\n"
            )
        for attempt in range(max_retries):
            try:
                response = await chat.send_message_async(prompt)
                usage = getattr(response, 'usage_metadata', None)
                if usage:
                    self._tokens_used += usage.total_token_count
                    logger.info(f"Gemini API usage: prompt_tokens={usage.prompt_token_count}, output_tokens={usage.candidates_token_count}, total_tokens={usage.total_token_count}, tokens_used_this_minute={self._tokens_used}")
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
        """Generate a response grounded in RAG-retrieved course material. Only one Gemini API call per user input."""
        logger = logging.getLogger(__name__)
        logger.info(f"[GeminiAgent] generate_rag_response called with message: {message}\nContext: {context}\nSources: {sources}")
        style = self._determine_response_style(message)
        if style == "brief":
            style_instruction = "Provide a concise summary (2-3 sentences) focusing on key points from the material."
        elif style == "detailed":
            style_instruction = "Provide an in-depth, comprehensive answer with examples, comparisons, and critical analysis."
        else:
            style_instruction = "Provide a comprehensive, educationally rich answer that grounds the response in the retrieved material, expands with conceptual depth and examples, follows your teaching philosophy, and ends with a follow-up question to check understanding."

        cleaned_context = self.clean_context(context)
        rag_prompt = f"""
You are an IGNOU Psychology Tutor.

Your task is to generate a CLEAN and STRUCTURED answer.

STRICT RULES (MANDATORY):
- DO NOT copy sentences from context
- DO NOT repeat content
- FIX broken words
- REMOVE irrelevant lines
- USE ONLY headings and bullet points (NO plain paragraphs)
- FORMAT using markdown

OUTPUT FORMAT (MANDATORY):

## Main Title

### Heading 1
- Bullet point 1
- Bullet point 2

### Heading 2
- Bullet point 1
- Bullet point 2

### Heading 3
- Bullet point 1
- Bullet point 2

(Do NOT write plain paragraphs. Use only headings and bullet points for all content.)

Question:
{message}

Context:
{cleaned_context}

Now generate a well-structured answer.
"""

        # Use the full history if provided, otherwise just the current message
        chat_history = self._build_chat_history(history or [])
        chat = self.model.start_chat(history=chat_history)
        logger.info(f"[GeminiAgent] Sending RAG prompt to Gemini: {rag_prompt}")
        response = await self._retry_api_call(chat, rag_prompt)
        logger.info(f"[GeminiAgent] Gemini RAG response: {response}")
        return self.post_process(response)

    async def generate_fallback_response(
        self,
        message: str,
        history: list[dict] | None = None,
    ) -> str:
        """Pure Gemini agent response — used when RAG finds no relevant material. Only one Gemini API call per user input."""
        logger = logging.getLogger(__name__)
        logger.info(f"[GeminiAgent] generate_fallback_response called with message: {message}")
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

        # Use the full history if provided, otherwise just the current message
        chat_history = self._build_chat_history(history or [])
        chat = self.model.start_chat(history=chat_history)
        logger.info(f"[GeminiAgent] Sending fallback prompt to Gemini: {fallback_prompt}")
        response = await self._retry_api_call(chat, fallback_prompt)
        logger.info(f"[GeminiAgent] Gemini fallback response: {response}")
        return response