"""AI Text Analyzer API – main FastAPI application.

Endpoints
---------
GET  /            – health check
POST /analyze     – summary + key insights + sentiment
POST /analyze-caption – generated caption + summary + sentiment
POST /generate-tags   – relevant tags / hashtags
"""

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models import AnalysisResponse, CaptionResponse, TagsResponse, TextInput
from app.services.analyzer import TextAnalyzerService

load_dotenv()  # load .env file if present


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

_analyzer: TextAnalyzerService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Initialise (and later tear down) shared resources."""
    global _analyzer  # noqa: PLW0603
    _analyzer = TextAnalyzerService()
    yield
    _analyzer = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Text Analyzer API",
    description=(
        "Analyse text using LangChain + OpenAI. "
        "Returns summaries, key insights, sentiment, captions, and tags."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS – permissive defaults for demo/development.
# NOTE: allow_credentials=True requires an explicit origin list in production;
# browsers reject credentialed requests to a wildcard origin.
# Replace ["*"] with your actual frontend origin(s) before going live.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_analyzer() -> TextAnalyzerService:
    if _analyzer is None:  # pragma: no cover
        raise HTTPException(status_code=503, detail="Analyzer service not ready.")
    return _analyzer


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", summary="Health check")
def root():
    """Returns a simple status message to confirm the API is running."""
    return {"status": "ok", "message": "AI Text Analyzer API is running."}


@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Analyse text",
    description=(
        "Accepts a piece of text and returns a **summary**, a list of "
        "**key insights**, and the overall **sentiment** (positive / negative / neutral)."
    ),
)
def analyze(payload: TextInput) -> AnalysisResponse:
    analyzer = get_analyzer()
    try:
        return analyzer.analyze(payload.text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post(
    "/analyze-caption",
    response_model=CaptionResponse,
    summary="Generate caption and analyse text",
    description=(
        "Accepts a piece of text and returns an engaging **caption**, a brief "
        "**summary**, and the overall **sentiment**. "
        "Ideal for auto-generating captions for Top Recruit Tapes."
    ),
)
def analyze_caption(payload: TextInput) -> CaptionResponse:
    analyzer = get_analyzer()
    try:
        return analyzer.analyze_caption(payload.text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post(
    "/generate-tags",
    response_model=TagsResponse,
    summary="Generate tags",
    description=(
        "Accepts a piece of text and returns a list of relevant **tags** "
        "(hashtag-style, without the # symbol). "
        "Useful for social-media and SEO workflows."
    ),
)
def generate_tags(payload: TextInput) -> TagsResponse:
    analyzer = get_analyzer()
    try:
        return analyzer.generate_tags(payload.text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
