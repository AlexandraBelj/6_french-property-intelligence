"""
PURPOSE
-------
Convert a user-entered French address into the geographic information
required by the frozen property valuation model.

Geocoding is intentionally kept outside src/inference.py.

The service uses the official French IGN Géoplateforme geocoding API,
which provides BAN-based address search results.

The model requires:
- latitude
- longitude
- postcode

The API also keeps the resolved address, score and result type so that
the application can communicate what location was actually matched.
"""

from dataclasses import dataclass

import requests


GEOCODING_URL = "https://data.geopf.fr/geocodage/search"

REQUEST_TIMEOUT_SECONDS = 10

# Prevent clearly weak geocoding results from silently reaching the model.
MINIMUM_GEOCODING_SCORE = 0.50


class GeocodingError(Exception):
    """Raised when an address cannot be geocoded reliably."""


@dataclass(frozen=True)
class GeocodingResult:
    """Normalized geographic information returned by the geocoder."""

    latitude: float
    longitude: float
    postcode: str
    resolved_address: str
    score: float
    result_type: str


def geocode_address(address: str) -> GeocodingResult:
    """
    Resolve one French postal address.

    Parameters
    ----------
    address:
        User-entered French address.

    Returns
    -------
    GeocodingResult
        Coordinates, postcode and information about the selected match.

    Raises
    ------
    GeocodingError
        If the remote service fails, returns no result, or returns an
        incomplete / insufficiently reliable result.
    """

    cleaned_address = address.strip()

    if not cleaned_address:
        raise GeocodingError("Address cannot be empty.")

    try:
        response = requests.get(
            GEOCODING_URL,
            params={
                "q": cleaned_address,
                "limit": 1,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

    except requests.RequestException as exc:
        raise GeocodingError(
            "The address geocoding service is currently unavailable."
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise GeocodingError(
            "The geocoding service returned an invalid response."
        ) from exc

    features = payload.get("features", [])

    if not features:
        raise GeocodingError(
            "No geographic match was found for this address."
        )

    best_match = features[0]

    try:
        geometry = best_match["geometry"]
        properties = best_match["properties"]

        longitude, latitude = geometry["coordinates"]

        postcode = str(properties["postcode"])
        resolved_address = str(properties["label"])
        score = float(properties["score"])
        result_type = str(properties.get("type", "unknown"))

    except (KeyError, TypeError, ValueError) as exc:
        raise GeocodingError(
            "The geocoding result is incomplete."
        ) from exc

    if len(postcode) != 5 or not postcode.isdigit():
        raise GeocodingError(
            "The geocoding result does not contain a valid French postcode."
        )

    if not (-90 <= latitude <= 90):
        raise GeocodingError(
            "The geocoding result contains an invalid latitude."
        )

    if not (-180 <= longitude <= 180):
        raise GeocodingError(
            "The geocoding result contains an invalid longitude."
        )

    if score < MINIMUM_GEOCODING_SCORE:
        raise GeocodingError(
            "The address match is not reliable enough for valuation."
        )

    return GeocodingResult(
        latitude=float(latitude),
        longitude=float(longitude),
        postcode=postcode,
        resolved_address=resolved_address,
        score=score,
        result_type=result_type,
    )