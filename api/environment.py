"""
PURPOSE
-------
Retrieve nearby property-environment information from Geoapify Places.

This module is deliberately separate from:
- address geocoding;
- machine-learning inference;
- Streamlit presentation.

Architecture:

verified latitude / longitude
        -> Geoapify Places API
        -> nearby POIs
        -> cleanup / deduplication
        -> FastAPI /environment response

Only a small number of useful nearby places are returned so that the
Streamlit map remains readable.

The Geoapify API key must be provided through the environment variable:

    GEOAPIFY_API_KEY

It must never be hard-coded in the repository.
"""

import os
from dataclasses import dataclass

import requests


GEOAPIFY_PLACES_URL = "https://api.geoapify.com/v2/places"

REQUEST_TIMEOUT_SECONDS = 15

# Search a reasonably local neighborhood around the property.
SEARCH_RADIUS_METERS = 1500

# Request slightly more records than we expose because transport data
# can contain unnamed or duplicate platform records.
PROVIDER_RESULT_LIMIT = 20

# Keep the final product concise and readable.
RESULTS_PER_CATEGORY = 3


class EnvironmentError(RuntimeError):
    """Raised when nearby environment information cannot be retrieved."""


@dataclass(frozen=True)
class NearbyPlace:
    """Clean internal representation of one nearby point of interest."""

    name: str
    address: str
    distance_m: int
    latitude: float
    longitude: float


# ---------------------------------------------------------------------
# Geoapify category configuration
#
# PURPOSE:
# Keep provider-specific category names inside this module rather than
# leaking them into FastAPI or Streamlit.
# ---------------------------------------------------------------------

CATEGORY_CONFIG = {
    "transport": "public_transport",
    "green_spaces": "leisure.park",
    "supermarkets": "commercial.supermarket",
}


def _get_api_key() -> str:
    """Read the Geoapify API key from the server environment."""

    api_key = os.getenv("GEOAPIFY_API_KEY", "").strip()

    if not api_key:
        raise EnvironmentError(
            "Environment service is not configured."
        )

    return api_key


def _fetch_category(
    *,
    category: str,
    latitude: float,
    longitude: float,
) -> list[dict]:
    """
    Retrieve raw Geoapify features for one POI category.

    The search is centered on coordinates already verified by our
    official French geocoding service.
    """

    params = {
        "categories": category,
        "filter": (
            f"circle:{longitude},{latitude},{SEARCH_RADIUS_METERS}"
        ),
        "bias": f"proximity:{longitude},{latitude}",
        "limit": PROVIDER_RESULT_LIMIT,
        "apiKey": _get_api_key(),
    }

    try:
        response = requests.get(
            GEOAPIFY_PLACES_URL,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

    except requests.Timeout as exc:
        raise EnvironmentError(
            "Environment provider timed out."
        ) from exc

    except requests.RequestException as exc:
        raise EnvironmentError(
            "Environment provider is unavailable."
        ) from exc

    try:
        payload = response.json()

    except ValueError as exc:
        raise EnvironmentError(
            "Environment provider returned an invalid response."
        ) from exc

    features = payload.get("features")

    if not isinstance(features, list):
        raise EnvironmentError(
            "Environment provider returned an invalid response."
        )

    return features


def _clean_places(
    features: list[dict],
) -> list[NearbyPlace]:
    """
    Convert provider records into a small clean POI list.

    Cleanup rules:
    - ignore unnamed places;
    - require usable coordinates;
    - require a provider distance;
    - deduplicate repeated place names;
    - keep the nearest occurrence when duplicates exist;
    - return only the nearest useful results.
    """

    cleaned: list[NearbyPlace] = []

    for feature in features:

        if not isinstance(feature, dict):
            continue

        properties = feature.get("properties", {})
        geometry = feature.get("geometry", {})

        if not isinstance(properties, dict):
            continue

        if not isinstance(geometry, dict):
            continue

        name = properties.get("name")
        address = properties.get("formatted")
        distance = properties.get("distance")

        coordinates = geometry.get("coordinates")

        if not isinstance(name, str) or not name.strip():
            continue

        if not isinstance(coordinates, list) or len(coordinates) < 2:
            continue

        try:
            poi_longitude = float(coordinates[0])
            poi_latitude = float(coordinates[1])
            distance_m = int(round(float(distance)))

        except (TypeError, ValueError):
            continue

        cleaned.append(
            NearbyPlace(
                name=name.strip(),
                address=(
                    address.strip()
                    if isinstance(address, str)
                    else ""
                ),
                distance_m=distance_m,
                latitude=poi_latitude,
                longitude=poi_longitude,
            )
        )

    # Nearest places first.
    cleaned.sort(
        key=lambda place: place.distance_m
    )

    # -------------------------------------------------------------
    # Deduplicate repeated provider records.
    #
    # Transport datasets often contain several platform records for
    # the same named station. Because records are already sorted by
    # distance, the nearest occurrence is retained.
    # -------------------------------------------------------------

    unique_places: list[NearbyPlace] = []
    seen_names: set[str] = set()

    for place in cleaned:

        normalized_name = place.name.casefold()

        if normalized_name in seen_names:
            continue

        seen_names.add(normalized_name)
        unique_places.append(place)

        if len(unique_places) >= RESULTS_PER_CATEGORY:
            break

    return unique_places


def get_nearby_environment(
    *,
    latitude: float,
    longitude: float,
) -> dict[str, list[NearbyPlace]]:
    """
    Retrieve the three environment categories used by the application.

    PURPOSE:
    Provide one stable service interface to FastAPI while keeping all
    Geoapify-specific behavior isolated in this module.
    """

    result: dict[str, list[NearbyPlace]] = {}

    for public_name, provider_category in CATEGORY_CONFIG.items():

        features = _fetch_category(
            category=provider_category,
            latitude=latitude,
            longitude=longitude,
        )

        result[public_name] = _clean_places(
            features
        )

    return result