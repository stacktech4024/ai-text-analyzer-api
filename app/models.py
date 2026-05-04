"""Pydantic request and response models for the AI Text Analyzer API."""

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared request model
# ---------------------------------------------------------------------------

class TextInput(BaseModel):
    """Input payload that carries a piece of text to be analyzed."""

    text: str = Field(
        ...,
        min_length=1,
        description="The text to analyze (e.g. a player report or video caption).",
        examples=["LeBron James delivered another MVP-caliber performance tonight."],
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class AnalysisResponse(BaseModel):
    """Full analysis result returned by POST /analyze."""

    summary: str = Field(description="A concise 2-3 sentence summary of the text.")
    insights: list[str] = Field(description="3-5 key insights extracted from the text.")
    sentiment: str = Field(description="Overall sentiment: positive, negative, or neutral.")


class CaptionResponse(BaseModel):
    """Response returned by POST /analyze-caption."""

    caption: str = Field(description="An engaging caption generated from the text.")
    summary: str = Field(description="A brief summary of the original text.")
    sentiment: str = Field(description="Overall sentiment: positive, negative, or neutral.")


class TagsResponse(BaseModel):
    """Response returned by POST /generate-tags."""

    tags: list[str] = Field(description="Relevant tags or hashtags for the text.")
