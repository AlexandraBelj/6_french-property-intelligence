"""
PURPOSE
-------
Define the validated public request and response contracts for the
French Property Intelligence prediction API.

These schemas describe what the API user sends and receives.
Internal model features such as latitude and longitude are deliberately
not exposed as user inputs because they are obtained from address
geocoding.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """User-facing information required to value one property."""

    property_type: Literal[
        "house",
        "maison",
        "apartment",
        "appartement",
    ]

    address: str = Field(
        ...,
        min_length=5,
        max_length=300,
        description="French postal address to geocode.",
    )

    surface_habitable: float = Field(
        ...,
        gt=0,
        le=1000,
        description="Habitable surface in square metres.",
    )

    n_pieces: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Number of rooms. May be omitted.",
    )

    vefa: bool = Field(
        default=False,
        description="Whether the property is a VEFA transaction.",
    )

    @field_validator("address")
    @classmethod
    def clean_address(cls, value: str) -> str:
        """Remove unnecessary surrounding whitespace."""

        cleaned = value.strip()

        if not cleaned:
            raise ValueError("address cannot be empty.")

        return cleaned


class PredictionResponse(BaseModel):
    """Information returned after successful valuation."""

    estimated_price_eur: float
    property_type: str
    postcode: str
    resolved_address: str


class HealthResponse(BaseModel):
    """Minimal deployment health information."""

    status: str
    model_loaded: bool