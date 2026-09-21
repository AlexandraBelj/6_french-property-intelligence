"""
Production inference utilities for the French Property Intelligence project.

This module contains the deterministic preprocessing required by the final
LightGBM models.

Important:
- Address geocoding is intentionally NOT handled here.
- The caller must provide latitude, longitude, and postcode.
- No transformation is learned during inference.
- All categorical vocabularies and spatial cluster centers come from training.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree


class PropertyValuationModel:
    """
    Production inference wrapper for the final specialized property models.

    The wrapper:
    - routes houses and apartments to their respective LightGBM models;
    - reproduces training-fitted postcode categorical encoding;
    - assigns houses to their frozen virtual neighborhood;
    - preserves the exact feature order used during training.

    Geocoding is outside this class. The caller must already have:
    latitude, longitude, and postcode.
    """

    HOUSE_FEATURES = [
        "surface_habitable",
        "n_pieces",
        "vefa",
        "latitude",
        "longitude",
        "code_postal",
        "virtual_neighborhood",
    ]

    APARTMENT_FEATURES = [
        "surface_habitable",
        "n_pieces",
        "vefa",
        "latitude",
        "longitude",
        "code_postal",
    ]

    SUPPORTED_PROPERTY_TYPES = {
        "house": "house",
        "maison": "house",
        "apartment": "apartment",
        "appartement": "apartment",
    }

    def __init__(
        self,
        house_model: Any,
        apartment_model: Any,
        house_postcode_categories: list[str],
        apartment_postcode_categories: list[str],
        house_neighborhood_categories: list[int],
        cluster_centers: np.ndarray,
    ) -> None:
        """
        Store the fitted models and all training-derived inference state.

        No fitting occurs here.
        """

        self.house_model = house_model
        self.apartment_model = apartment_model

        self.house_postcode_categories = list(
            house_postcode_categories
        )

        self.apartment_postcode_categories = list(
            apartment_postcode_categories
        )

        self.house_neighborhood_categories = list(
            house_neighborhood_categories
        )

        self.cluster_centers = np.asarray(
            cluster_centers,
            dtype=np.float64,
        )

        # Reconstruct the same geographic projection used during training.
        #
        # always_xy=True means:
        # x input = longitude
        # y input = latitude
        self.lambert93_transformer = Transformer.from_crs(
            "EPSG:4326",
            "EPSG:2154",
            always_xy=True,
        )

        # Efficient nearest-center lookup.
        #
        # The tree is built only from frozen TRAINING-derived cluster centers.
        # It does not learn anything from inference data.
        self.cluster_tree = cKDTree(
            self.cluster_centers
        )

        self._validate_artifact_state()

    def _validate_artifact_state(self) -> None:
        """
        Validate the internal production artifact state.

        This catches corrupted or incomplete artifacts early.
        """

        if self.cluster_centers.ndim != 2:
            raise ValueError(
                "cluster_centers must be a 2-dimensional array."
            )

        if self.cluster_centers.shape[1] != 2:
            raise ValueError(
                "cluster_centers must contain X/Y coordinates."
            )

        if not np.isfinite(self.cluster_centers).all():
            raise ValueError(
                "cluster_centers contains non-finite values."
            )

        if len(self.house_neighborhood_categories) != len(
            self.cluster_centers
        ):
            raise ValueError(
                "Number of neighborhood categories does not match "
                "number of cluster centers."
            )

        if not self.house_postcode_categories:
            raise ValueError(
                "House postcode vocabulary is empty."
            )

        if not self.apartment_postcode_categories:
            raise ValueError(
                "Apartment postcode vocabulary is empty."
            )

    @staticmethod
    def _normalize_property_type(
        property_type: str,
    ) -> str:
        """
        Normalize French/English property-type labels.
        """

        if not isinstance(property_type, str):
            raise TypeError(
                "property_type must be a string."
            )

        normalized = property_type.strip().lower()

        mapping = {
            "house": "house",
            "maison": "house",
            "apartment": "apartment",
            "appartement": "apartment",
        }

        if normalized not in mapping:
            raise ValueError(
                "Unsupported property_type. "
                "Expected one of: "
                "house, maison, apartment, appartement."
            )

        return mapping[normalized]

    @staticmethod
    def _validate_numeric_inputs(
        surface_habitable: float,
        n_pieces: float | None,
        latitude: float,
        longitude: float,
    ) -> None:
        """
        Perform basic inference-time validation.

        These checks protect the model from malformed API inputs.
        They do NOT reproduce target-based training cleaning rules,
        because transaction price is unknown at inference time.
        """

        if not np.isfinite(surface_habitable):
            raise ValueError(
                "surface_habitable must be finite."
            )

        if surface_habitable <= 0:
            raise ValueError(
                "surface_habitable must be greater than zero."
            )

        if n_pieces is not None:
            if not np.isfinite(n_pieces):
                raise ValueError(
                    "n_pieces must be finite or None."
                )

        if not np.isfinite(latitude):
            raise ValueError(
                "latitude must be finite."
            )

        if not np.isfinite(longitude):
            raise ValueError(
                "longitude must be finite."
            )

        if not (-90 <= latitude <= 90):
            raise ValueError(
                "latitude must be between -90 and 90."
            )

        if not (-180 <= longitude <= 180):
            raise ValueError(
                "longitude must be between -180 and 180."
            )

    @staticmethod
    def _normalize_postcode(
        code_postal: str | int,
    ) -> str:
        """
        Normalize a French postcode to the 5-character representation
        used during model training.

        Examples:
        75001   -> "75001"
        "75001" -> "75001"
        1000    -> "01000"
        """

        if code_postal is None:
            raise ValueError(
                "code_postal cannot be None."
            )

        postcode = str(code_postal).strip()

        # Handle values that may arrive as strings such as "75001.0".
        if postcode.endswith(".0"):
            numeric_part = postcode[:-2]

            if numeric_part.isdigit():
                postcode = numeric_part

        if postcode.isdigit():
            postcode = postcode.zfill(5)

        if len(postcode) != 5:
            raise ValueError(
                "code_postal must contain 5 characters "
                "after normalization."
            )

        return postcode

    @staticmethod
    def _categorical_value(
        value: str | int,
        categories: list,
    ) -> pd.Categorical:
        """
        Construct a one-row pandas categorical using the frozen
        training vocabulary.

        Unknown values become NaN, matching the validation/test
        inference behavior used with LightGBM.
        """

        if value not in categories:
            value = None

        return pd.Categorical(
            [value],
            categories=categories,
        )

    def _get_virtual_neighborhood(
        self,
        latitude: float,
        longitude: float,
    ) -> int:
        """
        Assign a house to the nearest frozen training-derived
        virtual-neighborhood center.
        """

        x_l93, y_l93 = (
            self.lambert93_transformer.transform(
                longitude,
                latitude,
            )
        )

        point = np.array(
            [[x_l93, y_l93]],
            dtype=np.float64,
        )

        _, cluster_id = self.cluster_tree.query(
            point,
            k=1,
        )

        return int(cluster_id[0])

    def _build_house_features(
        self,
        surface_habitable: float,
        n_pieces: float | None,
        vefa: bool,
        latitude: float,
        longitude: float,
        code_postal: str,
    ) -> pd.DataFrame:
        """
        Build the exact seven-feature House input matrix.
        """

        neighborhood = (
            self._get_virtual_neighborhood(
                latitude=latitude,
                longitude=longitude,
            )
        )

        row = pd.DataFrame(
            {
                "surface_habitable": [
                    surface_habitable
                ],
                "n_pieces": [
                    np.nan
                    if n_pieces is None
                    else n_pieces
                ],
                "vefa": [bool(vefa)],
                "latitude": [latitude],
                "longitude": [longitude],
            }
        )

        row["code_postal"] = (
            self._categorical_value(
                code_postal,
                self.house_postcode_categories,
            )
        )

        row["virtual_neighborhood"] = (
            self._categorical_value(
                neighborhood,
                self.house_neighborhood_categories,
            )
        )

        return row[self.HOUSE_FEATURES]

    def _build_apartment_features(
        self,
        surface_habitable: float,
        n_pieces: float | None,
        vefa: bool,
        latitude: float,
        longitude: float,
        code_postal: str,
    ) -> pd.DataFrame:
        """
        Build the exact six-feature Apartment input matrix.
        """

        row = pd.DataFrame(
            {
                "surface_habitable": [
                    surface_habitable
                ],
                "n_pieces": [
                    np.nan
                    if n_pieces is None
                    else n_pieces
                ],
                "vefa": [bool(vefa)],
                "latitude": [latitude],
                "longitude": [longitude],
            }
        )

        row["code_postal"] = (
            self._categorical_value(
                code_postal,
                self.apartment_postcode_categories,
            )
        )

        return row[self.APARTMENT_FEATURES]

    def predict(
        self,
        property_type: str,
        surface_habitable: float,
        n_pieces: float | None,
        vefa: bool,
        latitude: float,
        longitude: float,
        code_postal: str | int,
    ) -> float:
        """
        Estimate the transaction value of one residential property.

        Parameters
        ----------
        property_type:
            "house" / "maison" or
            "apartment" / "appartement".

        surface_habitable:
            Habitable surface in square metres.

        n_pieces:
            Number of rooms. May be None.

        vefa:
            Whether the property is a VEFA transaction.

        latitude / longitude:
            WGS84 coordinates produced by the future geocoding layer.

        code_postal:
            French postcode.

        Returns
        -------
        float
            Estimated transaction price in euros.
        """

        normalized_type = (
            self._normalize_property_type(
                property_type
            )
        )

        self._validate_numeric_inputs(
            surface_habitable=surface_habitable,
            n_pieces=n_pieces,
            latitude=latitude,
            longitude=longitude,
        )

        normalized_postcode = (
            self._normalize_postcode(
                code_postal
            )
        )

        if normalized_type == "house":

            X = self._build_house_features(
                surface_habitable=surface_habitable,
                n_pieces=n_pieces,
                vefa=vefa,
                latitude=latitude,
                longitude=longitude,
                code_postal=normalized_postcode,
            )

            prediction = self.house_model.predict(X)[0]

        else:

            X = self._build_apartment_features(
                surface_habitable=surface_habitable,
                n_pieces=n_pieces,
                vefa=vefa,
                latitude=latitude,
                longitude=longitude,
                code_postal=normalized_postcode,
            )

            prediction = self.apartment_model.predict(X)[0]

        return float(prediction)