"""LangChain-powered text analysis service.

Each public method builds a small LCEL (LangChain Expression Language) chain:
    prompt | llm | output_parser

The chains are constructed once per instance so the LLM object is reused
across requests.
"""

import json
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.models import AnalysisResponse, CaptionResponse, TagsResponse


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_ANALYZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an expert content analyst. "
                "Always respond with valid JSON and nothing else. "
                "The JSON object must contain exactly these keys: "
                "\"summary\" (string), \"insights\" (array of strings, 3-5 items), "
                "\"sentiment\" (string: positive | negative | neutral)."
            ),
        ),
        (
            "human",
            "Analyze the following text and return the JSON object described.\n\nText:\n{text}",
        ),
    ]
)

_CAPTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a creative social-media copywriter who specializes in sports "
                "and entertainment content. "
                "Always respond with valid JSON and nothing else. "
                "The JSON object must contain exactly these keys: "
                "\"caption\" (string – one punchy sentence), "
                "\"summary\" (string – 1-2 sentences), "
                "\"sentiment\" (string: positive | negative | neutral)."
            ),
        ),
        (
            "human",
            "Generate a caption and analysis for the following text.\n\nText:\n{text}",
        ),
    ]
)

_TAGS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an SEO and social-media tagging expert. "
                "Always respond with valid JSON and nothing else. "
                "The JSON object must contain exactly one key: "
                "\"tags\" (array of strings, 5-10 lowercase hashtag-style tags "
                "without the # symbol)."
            ),
        ),
        (
            "human",
            "Generate relevant tags for the following text.\n\nText:\n{text}",
        ),
    ]
)


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------

class TextAnalyzerService:
    """Wraps LangChain chains for text analysis tasks."""

    def __init__(self, llm: ChatOpenAI | None = None) -> None:
        if llm is None:
            api_key = os.getenv("OPENAI_API_KEY")
            model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            llm = ChatOpenAI(api_key=api_key, model=model, temperature=0)

        parser = StrOutputParser()
        self._analyze_chain = _ANALYZE_PROMPT | llm | parser
        self._caption_chain = _CAPTION_PROMPT | llm | parser
        self._tags_chain = _TAGS_PROMPT | llm | parser

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> AnalysisResponse:
        """Return summary, insights, and sentiment for *text*."""
        raw = self._analyze_chain.invoke({"text": text})
        data = self._parse_json(raw)
        return AnalysisResponse(
            summary=data["summary"],
            insights=data["insights"],
            sentiment=data["sentiment"],
        )

    def analyze_caption(self, text: str) -> CaptionResponse:
        """Return a generated caption, summary, and sentiment for *text*."""
        raw = self._caption_chain.invoke({"text": text})
        data = self._parse_json(raw)
        return CaptionResponse(
            caption=data["caption"],
            summary=data["summary"],
            sentiment=data["sentiment"],
        )

    def generate_tags(self, text: str) -> TagsResponse:
        """Return a list of relevant tags for *text*."""
        raw = self._tags_chain.invoke({"text": text})
        data = self._parse_json(raw)
        return TagsResponse(tags=data["tags"])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Strip markdown code fences (if any) and parse JSON."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (```json or ```)
            cleaned = cleaned.split("\n", 1)[-1]
            # Remove closing fence
            if cleaned.endswith("```"):
                cleaned = cleaned[: cleaned.rfind("```")]
        return json.loads(cleaned)
