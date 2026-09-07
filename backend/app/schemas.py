from typing import Literal

from pydantic import BaseModel, Field


Sentiment = Literal["positive", "neutral", "negative"]
Mode = Literal["text", "image", "multimodal"]


class EmotionScore(BaseModel):
    type: str
    score: int = Field(ge=0, le=100)


class TextEvidence(BaseModel):
    text: str
    score: int = Field(ge=0, le=100)
    polarity: Sentiment
    aspect: str


class ImageEvidence(BaseModel):
    region: str
    score: int = Field(ge=0, le=100)
    observation: str


class AspectResult(BaseModel):
    name: str
    sentiment: Sentiment
    intensity: int = Field(ge=0, le=100)
    evidence: list[str]


class ConsistencyResult(BaseModel):
    level: Literal["consistent", "partially_consistent", "conflicting", "not_applicable"]
    score: int = Field(ge=0, le=100)
    explanation: str


class AnalysisResponse(BaseModel):
    analysis_id: str
    mode: Mode
    overall: Sentiment
    intensity: int = Field(ge=0, le=100)
    confidence: int = Field(ge=0, le=100)
    emotions: list[EmotionScore]
    aspects: list[AspectResult]
    text_evidence: list[TextEvidence]
    image_evidence: list[ImageEvidence]
    consistency: ConsistencyResult
    dominant_source: Literal["text", "image", "balanced", "not_applicable"]
    insights: list[str]
    recommendation: str
    explanation: str
    engine: str
    latency_ms: int


class HealthResponse(BaseModel):
    status: Literal["ok"]
    engine: str
    llm_enabled: bool
    llm_model: str | None = None
