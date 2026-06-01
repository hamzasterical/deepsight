from pydantic import BaseModel, Field


class ForgeryResponse(BaseModel):
    verdict: str = Field(description="FORGED or AUTHENTIC")
    confidence: float = Field(ge=0.0, le=100.0, description="Confidence score 0-100")
    forgery_type: str = Field(description="Splicing, Copy-Move, Retouching, or Unknown")
    forged_area_percentage: float = Field(ge=0.0, le=100.0, description="Percentage of forged pixels")
    heatmap_base64: str = Field(description="Base64-encoded heatmap overlay image")
    processing_time_ms: float = Field(ge=0.0, description="Total processing time in milliseconds")


class ErrorResponse(BaseModel):
    detail: str = Field(description="Error description")
    code: int = Field(description="HTTP status code")
