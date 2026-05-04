"""Tests for the AI Text Analyzer API.

The LangChain service is replaced with a lightweight mock so tests run
without an OpenAI API key.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Patch the TextAnalyzerService before importing the app so that the lifespan
# never tries to create a real ChatOpenAI instance.
SAMPLE_TEXT = (
    "LeBron James delivered another MVP-caliber performance tonight, "
    "scoring 40 points and leading the Lakers to a crucial playoff victory."
)


def _make_mock_analyzer():
    """Return a MagicMock that mimics TextAnalyzerService."""
    from app.models import AnalysisResponse, CaptionResponse, TagsResponse

    mock = MagicMock()
    mock.analyze.return_value = AnalysisResponse(
        summary="LeBron scored 40 points leading the Lakers to a playoff win.",
        insights=[
            "MVP-level performance",
            "40-point game",
            "Crucial playoff victory",
        ],
        sentiment="positive",
    )
    mock.analyze_caption.return_value = CaptionResponse(
        caption="King James takes over! 👑",
        summary="LeBron James scored 40 points in a critical Lakers playoff win.",
        sentiment="positive",
    )
    mock.generate_tags.return_value = TagsResponse(
        tags=["lebron", "lakers", "nba", "playoffs", "mvp", "basketball"]
    )
    return mock


@pytest.fixture()
def client():
    """TestClient with a mocked analyzer service."""
    mock_analyzer = _make_mock_analyzer()

    with patch("app.main.TextAnalyzerService", return_value=mock_analyzer):
        # Import app *after* the patch so the lifespan picks up the mock.
        from app.main import app

        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# CORS headers
# ---------------------------------------------------------------------------

def test_cors_headers_present(client):
    response = client.get("/", headers={"Origin": "http://localhost:3000"})
    assert response.headers.get("access-control-allow-origin") == "*"


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "AI Text Analyzer API" in body["message"]


# ---------------------------------------------------------------------------
# POST /analyze
# ---------------------------------------------------------------------------

def test_analyze_success(client):
    response = client.post("/analyze", json={"text": SAMPLE_TEXT})
    assert response.status_code == 200
    body = response.json()
    assert "summary" in body
    assert isinstance(body["summary"], str)
    assert "insights" in body
    assert isinstance(body["insights"], list)
    assert len(body["insights"]) >= 1
    assert "sentiment" in body
    assert body["sentiment"] in {"positive", "negative", "neutral"}


def test_analyze_empty_text_rejected(client):
    response = client.post("/analyze", json={"text": ""})
    assert response.status_code == 422  # Pydantic validation error


def test_analyze_missing_text_rejected(client):
    response = client.post("/analyze", json={})
    assert response.status_code == 422


def test_analyze_propagates_service_error(client):
    from app.main import _analyzer

    _analyzer.analyze.side_effect = RuntimeError("LLM unavailable")
    response = client.post("/analyze", json={"text": SAMPLE_TEXT})
    assert response.status_code == 502
    _analyzer.analyze.side_effect = None  # reset


# ---------------------------------------------------------------------------
# POST /analyze-caption
# ---------------------------------------------------------------------------

def test_analyze_caption_success(client):
    response = client.post("/analyze-caption", json={"text": SAMPLE_TEXT})
    assert response.status_code == 200
    body = response.json()
    assert "caption" in body
    assert isinstance(body["caption"], str)
    assert "summary" in body
    assert "sentiment" in body
    assert body["sentiment"] in {"positive", "negative", "neutral"}


def test_analyze_caption_empty_text_rejected(client):
    response = client.post("/analyze-caption", json={"text": ""})
    assert response.status_code == 422


def test_analyze_caption_propagates_service_error(client):
    from app.main import _analyzer

    _analyzer.analyze_caption.side_effect = RuntimeError("LLM unavailable")
    response = client.post("/analyze-caption", json={"text": SAMPLE_TEXT})
    assert response.status_code == 502
    _analyzer.analyze_caption.side_effect = None


# ---------------------------------------------------------------------------
# POST /generate-tags
# ---------------------------------------------------------------------------

def test_generate_tags_success(client):
    response = client.post("/generate-tags", json={"text": SAMPLE_TEXT})
    assert response.status_code == 200
    body = response.json()
    assert "tags" in body
    assert isinstance(body["tags"], list)
    assert len(body["tags"]) >= 1


def test_generate_tags_empty_text_rejected(client):
    response = client.post("/generate-tags", json={"text": ""})
    assert response.status_code == 422


def test_generate_tags_propagates_service_error(client):
    from app.main import _analyzer

    _analyzer.generate_tags.side_effect = RuntimeError("LLM unavailable")
    response = client.post("/generate-tags", json={"text": SAMPLE_TEXT})
    assert response.status_code == 502
    _analyzer.generate_tags.side_effect = None


# ---------------------------------------------------------------------------
# Analyzer service unit tests (JSON parsing helper)
# ---------------------------------------------------------------------------

def test_parse_json_plain():
    from app.services.analyzer import TextAnalyzerService

    raw = json.dumps({"key": "value"})
    assert TextAnalyzerService._parse_json(raw) == {"key": "value"}


def test_parse_json_with_markdown_fence():
    from app.services.analyzer import TextAnalyzerService

    raw = '```json\n{"key": "value"}\n```'
    assert TextAnalyzerService._parse_json(raw) == {"key": "value"}


def test_parse_json_with_plain_fence():
    from app.services.analyzer import TextAnalyzerService

    raw = '```\n{"key": "value"}\n```'
    assert TextAnalyzerService._parse_json(raw) == {"key": "value"}
