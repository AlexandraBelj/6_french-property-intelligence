"""
PURPOSE
-------
Expose the French Property Intelligence services through FastAPI.

The API separates user-facing property information from internal
geographic and machine-learning features.

Valuation flow:

User address
    -> official French geocoding service
    -> latitude / longitude / postcode
    -> frozen PropertyValuationModel
    -> estimated transaction value

Context flow:

User address
    -> official French geocoding service
    -> verified geographic coordinates
    -> map / nearby environment / neighborhood services

The production model is loaded once when the API application starts.
It is NOT reloaded for every prediction.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.environment import (
    EnvironmentError,
    get_nearby_environment,
)
from api.geocoding import GeocodingError, geocode_address
from api.model_loader import ModelLoadingError, load_production_model
from api.schemas import (
    EnvironmentResponse,
    HealthResponse,
    LocationRequest,
    LocationResponse,
    NearbyPlaceResponse,
    PredictionRequest,
    PredictionResponse,
)


# ---------------------------------------------------------------------
# Logging
#
# PURPOSE:
# Record unexpected server-side errors in deployment logs without
# exposing internal implementation details to API clients.
# ---------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------

production_model = None


# ---------------------------------------------------------------------
# Startup / shutdown lifecycle
#
# PURPOSE:
# Load and validate the large production artifact once when the API
# starts rather than repeating that expensive operation per request.
# ---------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global production_model

    try:
        production_model = load_production_model()

    except ModelLoadingError as exc:
        # A missing, corrupted or untrusted model is a deployment failure.
        # The API must not start serving predictions in that state.
        raise RuntimeError(
            "Production model failed to load."
        ) from exc

    yield

    production_model = None


# ---------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------

app = FastAPI(
    title="French Property Intelligence API",
    description=(
        "Indicative residential property valuation and "
        "property-context API for metropolitan France."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    """
    Confirm that the API is running and the production model is loaded.
    """

    return HealthResponse(
        status="ok" if production_model is not None else "unavailable",
        model_loaded=production_model is not None,
    )


# ---------------------------------------------------------------------
# Location endpoint
#
# PURPOSE:
# Resolve a user-entered address into verified geographic coordinates.
#
# This endpoint remains deliberately separate from /predict.
# ---------------------------------------------------------------------

@app.post(
    "/location",
    response_model=LocationResponse,
)
def resolve_location(
    request: LocationRequest,
) -> LocationResponse:
    """Resolve one French address into geographic coordinates."""

    try:
        location = geocode_address(request.address)

    except GeocodingError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    return LocationResponse(
        latitude=location.latitude,
        longitude=location.longitude,
        postcode=location.postcode,
        resolved_address=location.resolved_address,
    )


# ---------------------------------------------------------------------
# Environment endpoint
#
# PURPOSE:
# Combine our verified geocoding service with Geoapify Places.
#
# The client supplies only the property address. Latitude and longitude
# are resolved internally, keeping geographic handling consistent with
# /predict and /location.
# ---------------------------------------------------------------------

@app.post(
    "/environment",
    response_model=EnvironmentResponse,
)
def property_environment(
    request: LocationRequest,
) -> EnvironmentResponse:
    """Return cleaned nearby environment information for one property."""

    # -------------------------------------------------------------
    # 1. Resolve the address using our existing geocoding service.
    # -------------------------------------------------------------

    try:
        location = geocode_address(request.address)

    except GeocodingError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    # -------------------------------------------------------------
    # 2. Retrieve nearby POIs from the contextual data provider.
    # -------------------------------------------------------------

    try:
        environment = get_nearby_environment(
            latitude=location.latitude,
            longitude=location.longitude,
        )

    except EnvironmentError as exc:

        logger.warning(
            "Environment service failed: %s",
            exc,
        )

        raise HTTPException(
            status_code=503,
            detail="Nearby environment information is unavailable.",
        ) from exc

    # -------------------------------------------------------------
    # 3. Convert internal POI objects into the public API contract.
    # -------------------------------------------------------------

    return EnvironmentResponse(
        resolved_address=location.resolved_address,
        postcode=location.postcode,
        latitude=location.latitude,
        longitude=location.longitude,
        transport=[
            NearbyPlaceResponse(
                name=place.name,
                address=place.address,
                distance_m=place.distance_m,
                latitude=place.latitude,
                longitude=place.longitude,
            )
            for place in environment["transport"]
        ],
        green_spaces=[
            NearbyPlaceResponse(
                name=place.name,
                address=place.address,
                distance_m=place.distance_m,
                latitude=place.latitude,
                longitude=place.longitude,
            )
            for place in environment["green_spaces"]
        ],
        supermarkets=[
            NearbyPlaceResponse(
                name=place.name,
                address=place.address,
                distance_m=place.distance_m,
                latitude=place.latitude,
                longitude=place.longitude,
            )
            for place in environment["supermarkets"]
        ],
    )


# ---------------------------------------------------------------------
# Prediction endpoint
#
# IMPORTANT:
# The validated production prediction behavior remains unchanged.
# ---------------------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_property(
    request: PredictionRequest,
) -> PredictionResponse:
    """
    Estimate the transaction value of one residential property.

    Address geocoding occurs before model inference. The model itself
    never receives the raw address.
    """

    if production_model is None:
        raise HTTPException(
            status_code=503,
            detail="Production model is not available.",
        )

    # -------------------------------------------------------------
    # 1. Resolve user-facing address into model geographic inputs.
    # -------------------------------------------------------------

    try:
        location = geocode_address(request.address)

    except GeocodingError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    # -------------------------------------------------------------
    # 2. Run the frozen production inference pipeline.
    # -------------------------------------------------------------

    try:
        estimated_price = production_model.predict(
            property_type=request.property_type,
            surface_habitable=request.surface_habitable,
            n_pieces=request.n_pieces,
            vefa=request.vefa,
            latitude=location.latitude,
            longitude=location.longitude,
            code_postal=location.postcode,
        )

    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Unexpected error during property valuation inference."
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc

    # -------------------------------------------------------------
    # 3. Return only the public prediction contract.
    # -------------------------------------------------------------

    return PredictionResponse(
        estimated_price_eur=round(
            float(estimated_price),
            2,
        ),
        property_type=request.property_type,
        postcode=location.postcode,
        resolved_address=location.resolved_address,
    )