"""
PURPOSE
-------
Main Streamlit interface for French Property Intelligence.

The application provides a property-intelligence journey:

1. The user describes a residential property.
2. Streamlit sends the information to the FastAPI service.
3. FastAPI geocodes the address and runs the frozen production model.
4. Streamlit presents the valuation in a compact dashboard.
5. The environment tab displays the verified property location.
6. Additional tabs will progressively provide:
   - nearby transport, green spaces and shops;
   - neighborhood / Street View exploration;
   - GenAI property description.

IMPORTANT
---------
Streamlit does NOT load model.pkl directly.

Production architecture:

Streamlit -> FastAPI -> IGN geocoding -> S3 model -> prediction

The ML model remains frozen behind the FastAPI service.
"""

import os

import folium
import requests
import streamlit as st

from streamlit_folium import st_folium


# ---------------------------------------------------------------------
# Application configuration
#
# PURPOSE:
# Configure the Streamlit page and keep the API endpoint configurable
# between local development and Hugging Face deployment.
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="French Property Intelligence",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

PREDICT_URL = f"{API_URL}/predict"
LOCATION_URL = f"{API_URL}/location"

REQUEST_TIMEOUT_SECONDS = 30


# ---------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------

def format_euros(value: float) -> str:
    """Format a monetary value using French-style thousands spacing."""

    return f"{value:,.0f} €".replace(",", " ")


def format_price_per_m2(price: float, surface: float) -> str:
    """Calculate and format the indicative estimated price per m²."""

    if surface <= 0:
        return "—"

    value = price / surface

    return f"{value:,.0f} €/m²".replace(",", " ")


# ---------------------------------------------------------------------
# Session state
#
# PURPOSE:
# Keep the latest successful valuation visible when Streamlit reruns.
# Contextual services can therefore use the same verified property.
# ---------------------------------------------------------------------

if "valuation_result" not in st.session_state:
    st.session_state.valuation_result = None


# ---------------------------------------------------------------------
# Styling
#
# PURPOSE:
# Create a clear product hierarchy while remaining inside Streamlit.
# ---------------------------------------------------------------------

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1280px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 1.7rem 2rem;
        border: 1px solid rgba(128, 128, 128, 0.20);
        border-radius: 18px;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        margin: 0;
        margin-bottom: 0.35rem;
    }

    .hero p {
        margin: 0;
        opacity: 0.72;
        font-size: 1.05rem;
    }

    .section-card {
        border: 1px solid rgba(128, 128, 128, 0.20);
        border-radius: 18px;
        padding: 1.4rem;
    }

    .estimate-card {
        padding: 2.2rem 1.5rem;
        border: 1px solid rgba(128, 128, 128, 0.20);
        border-radius: 18px;
        text-align: center;
        margin-bottom: 1.2rem;
    }

    .estimate-label {
        font-size: 0.95rem;
        opacity: 0.68;
        margin-bottom: 0.35rem;
    }

    .estimate-value {
        font-size: 2.8rem;
        font-weight: 750;
        line-height: 1.15;
        margin-bottom: 0.5rem;
    }

    .estimate-note {
        opacity: 0.62;
        font-size: 0.82rem;
    }

    .location-card {
        border: 1px solid rgba(128, 128, 128, 0.20);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
    }

    .location-title {
        font-weight: 650;
        margin-bottom: 0.3rem;
    }

    .location-text {
        font-size: 1rem;
    }

    .placeholder-card {
        border: 1px dashed rgba(128, 128, 128, 0.35);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1rem;
    }

    .placeholder-title {
        font-size: 1.1rem;
        font-weight: 650;
        margin-bottom: 0.4rem;
    }

    .placeholder-text {
        opacity: 0.68;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🏡 French Property Intelligence</h1>
        <p>
            Estimation et analyse d'un bien immobilier résidentiel
            en France métropolitaine.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Main dashboard
#
# PURPOSE:
# Keep property inputs and the primary valuation visible together.
# ---------------------------------------------------------------------

input_column, result_column = st.columns(
    [0.36, 0.64],
    gap="large",
)


# =====================================================================
# LEFT COLUMN — PROPERTY INPUTS
# =====================================================================

with input_column:

    st.subheader("Votre bien")

    st.caption(
        "Renseignez les principales caractéristiques du logement."
    )

    with st.form("valuation_form"):

        address = st.text_input(
            "Adresse du bien",
            placeholder="Ex. 15 Rue de Rivoli 75001 Paris",
        )

        property_type_label = st.selectbox(
            "Type de bien",
            options=[
                "Appartement",
                "Maison",
            ],
        )

        input_1, input_2 = st.columns(2)

        with input_1:

            surface_habitable = st.number_input(
                "Surface (m²)",
                min_value=1.0,
                max_value=1000.0,
                value=65.0,
                step=1.0,
            )

        with input_2:

            n_pieces = st.number_input(
                "Nombre de pièces",
                min_value=1,
                max_value=50,
                value=3,
                step=1,
            )

        vefa = st.checkbox(
            "Bien vendu en VEFA",
            value=False,
            help=(
                "Vente en l'état futur d'achèvement : "
                "achat d'un logement neuf avant sa construction "
                "ou son achèvement."
            ),
        )

        submitted = st.form_submit_button(
            "Estimer le bien",
            use_container_width=True,
            type="primary",
        )

    st.caption(
        "L'adresse est géolocalisée automatiquement afin "
        "d'intégrer la localisation dans l'estimation."
    )


# =====================================================================
# PREDICTION REQUEST
#
# PURPOSE:
# Call the existing FastAPI valuation service.
# No ML inference occurs directly inside Streamlit.
# =====================================================================

if submitted:

    cleaned_address = address.strip()

    if len(cleaned_address) < 5:

        st.session_state.valuation_result = None

        with result_column:
            st.error(
                "Veuillez renseigner une adresse suffisamment précise."
            )

    else:

        property_type = (
            "apartment"
            if property_type_label == "Appartement"
            else "house"
        )

        payload = {
            "property_type": property_type,
            "address": cleaned_address,
            "surface_habitable": float(surface_habitable),
            "n_pieces": int(n_pieces),
            "vefa": bool(vefa),
        }

        try:

            with result_column:

                with st.spinner(
                    "Géolocalisation et calcul de l'estimation..."
                ):

                    response = requests.post(
                        PREDICT_URL,
                        json=payload,
                        timeout=REQUEST_TIMEOUT_SECONDS,
                    )

            # ---------------------------------------------------------
            # Deliberate validation errors returned by FastAPI.
            # ---------------------------------------------------------

            if response.status_code == 422:

                st.session_state.valuation_result = None

                try:

                    error_payload = response.json()

                    detail = error_payload.get(
                        "detail",
                        "Les informations saisies ne peuvent pas être traitées.",
                    )

                except ValueError:

                    detail = (
                        "Les informations saisies ne peuvent pas être traitées."
                    )

                with result_column:
                    st.error(detail)

            # ---------------------------------------------------------
            # Unexpected server/API error.
            # ---------------------------------------------------------

            elif response.status_code != 200:

                st.session_state.valuation_result = None

                with result_column:
                    st.error(
                        "Le service d'estimation est temporairement "
                        "indisponible. Veuillez réessayer."
                    )

            # ---------------------------------------------------------
            # Successful valuation.
            # ---------------------------------------------------------

            else:

                result = response.json()

                estimated_price = float(
                    result["estimated_price_eur"]
                )

                resolved_address = result["resolved_address"]
                postcode = result["postcode"]

                # -----------------------------------------------------
                # Resolve coordinates for contextual features.
                #
                # PURPOSE:
                # /predict remains dedicated to valuation.
                # /location provides verified coordinates for the map
                # and future environment / neighborhood services.
                #
                # A failure here must NOT invalidate a successful
                # property valuation.
                # -----------------------------------------------------

                latitude = None
                longitude = None

                try:

                    location_response = requests.post(
                        LOCATION_URL,
                        json={
                            "address": cleaned_address,
                        },
                        timeout=REQUEST_TIMEOUT_SECONDS,
                    )

                    if location_response.status_code == 200:

                        location_result = location_response.json()

                        latitude = float(
                            location_result["latitude"]
                        )

                        longitude = float(
                            location_result["longitude"]
                        )

                except (
                    requests.RequestException,
                    KeyError,
                    TypeError,
                    ValueError,
                ):
                    # Contextual services are optional enhancements.
                    # The successful valuation remains valid if the
                    # location endpoint is temporarily unavailable.
                    pass

                st.session_state.valuation_result = {
                    "estimated_price": estimated_price,
                    "resolved_address": resolved_address,
                    "postcode": postcode,
                    "surface_habitable": float(surface_habitable),
                    "n_pieces": int(n_pieces),
                    "property_type": property_type,
                    "property_type_label": property_type_label,
                    "vefa": bool(vefa),
                    "user_address": cleaned_address,
                    "latitude": latitude,
                    "longitude": longitude,
                }

        except requests.Timeout:

            st.session_state.valuation_result = None

            with result_column:
                st.error(
                    "Le service d'estimation met trop de temps à répondre. "
                    "Veuillez réessayer."
                )

        except requests.ConnectionError:

            st.session_state.valuation_result = None

            with result_column:
                st.error(
                    "Impossible de joindre le service d'estimation."
                )

        except (KeyError, TypeError, ValueError):

            st.session_state.valuation_result = None

            with result_column:
                st.error(
                    "La réponse du service d'estimation est invalide."
                )

        except requests.RequestException:

            st.session_state.valuation_result = None

            with result_column:
                st.error(
                    "Une erreur réseau est survenue pendant l'estimation."
                )


# =====================================================================
# RIGHT COLUMN — PRIMARY VALUATION
# =====================================================================

with result_column:

    valuation = st.session_state.valuation_result

    if valuation is None:

        st.subheader("Estimation")

        st.markdown(
            """
            <div class="placeholder-card">
                <div class="placeholder-title">
                    Votre estimation apparaîtra ici
                </div>
                <div class="placeholder-text">
                    Renseignez le bien à gauche puis lancez l'estimation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        estimated_price = valuation["estimated_price"]
        surface = valuation["surface_habitable"]

        price_per_m2 = format_price_per_m2(
            estimated_price,
            surface,
        )

        st.subheader("Estimation du bien")

        # IMPORTANT:
        # Keep the HTML compact and without Markdown indentation.
        # This prevents Streamlit/Markdown from interpreting the HTML
        # content as a literal code block.

        estimate_html = (
            '<div class="estimate-card">'
            '<div class="estimate-label">'
            'Estimation indicative du prix de vente'
            '</div>'
            '<div class="estimate-value">'
            f'{format_euros(estimated_price)}'
            '</div>'
            '<div class="estimate-note">'
            'Estimation issue du modèle statistique — '
            'elle ne constitue pas une expertise immobilière.'
            '</div>'
            '</div>'
        )

        st.markdown(
            estimate_html,
            unsafe_allow_html=True,
        )

        metric_1, metric_2, metric_3 = st.columns(3)

        metric_1.metric(
            "Prix estimé au m²",
            price_per_m2,
        )

        metric_2.metric(
            "Surface",
            f"{surface:.0f} m²",
        )

        metric_3.metric(
            "Pièces",
            valuation["n_pieces"],
        )

        location_html = (
            '<div class="location-card">'
            '<div class="location-title">'
            '📍 Localisation reconnue'
            '</div>'
            '<div class="location-text">'
            f'{valuation["resolved_address"]} — '
            f'{valuation["postcode"]}'
            '</div>'
            '</div>'
        )

        st.markdown(
            location_html,
            unsafe_allow_html=True,
        )

        st.caption(
            "Vérifiez que la localisation reconnue correspond bien "
            "au bien que vous souhaitez estimer."
        )


# ---------------------------------------------------------------------
# Context dashboard
#
# PURPOSE:
# Organize property-intelligence services into dedicated views instead
# of creating one very long page.
# ---------------------------------------------------------------------

if st.session_state.valuation_result is not None:

    st.divider()

    (
        valuation_tab,
        environment_tab,
        neighborhood_tab,
        description_tab,
    ) = st.tabs(
        [
            "📊 Estimation",
            "🗺️ Environnement",
            "👁️ Quartier",
            "✨ Description",
        ]
    )

    # -----------------------------------------------------------------
    # VALUATION TAB
    # -----------------------------------------------------------------

    with valuation_tab:

        valuation = st.session_state.valuation_result

        st.subheader("Synthèse du bien")

        summary_1, summary_2, summary_3, summary_4 = st.columns(4)

        summary_1.metric(
            "Valeur estimée",
            format_euros(
                valuation["estimated_price"]
            ),
        )

        summary_2.metric(
            "Prix estimé / m²",
            format_price_per_m2(
                valuation["estimated_price"],
                valuation["surface_habitable"],
            ),
        )

        summary_3.metric(
            "Type",
            valuation["property_type_label"],
        )

        summary_4.metric(
            "VEFA",
            "Oui" if valuation["vefa"] else "Non",
        )

        st.info(
            "Cette estimation constitue une aide à la décision fondée "
            "sur les caractéristiques du bien et sa localisation. "
            "Elle ne remplace pas une expertise immobilière."
        )

    # -----------------------------------------------------------------
    # ENVIRONMENT TAB
    #
    # PURPOSE:
    # Display the verified geographic position of the property.
    #
    # This map is deliberately independent from ML inference.
    # Nearby transport, green spaces and shops will later be added as
    # contextual layers around this same verified property position.
    # -----------------------------------------------------------------

    with environment_tab:

        st.subheader("Environnement du bien")

        st.caption(
            "Explorez la localisation reconnue du bien "
            "et son environnement."
        )

        valuation = st.session_state.valuation_result

        latitude = valuation.get("latitude")
        longitude = valuation.get("longitude")

        if latitude is None or longitude is None:

            st.warning(
                "La carte est temporairement indisponible. "
                "L'estimation du bien reste néanmoins valide."
            )

        else:

            # ---------------------------------------------------------
            # Interactive property map
            #
            # PURPOSE:
            # Provide immediate geographic context around the property.
            # The same map can later receive POI markers without
            # changing the frozen valuation model.
            # ---------------------------------------------------------

            property_map = folium.Map(
                location=[
                    latitude,
                    longitude,
                ],
                zoom_start=16,
                control_scale=True,
            )

            folium.Marker(
                location=[
                    latitude,
                    longitude,
                ],
                tooltip="Bien analysé",
                popup=valuation["resolved_address"],
                icon=folium.Icon(
                    icon="home",
                    prefix="fa",
                ),
            ).add_to(property_map)

            st_folium(
                property_map,
                use_container_width=True,
                height=480,
                returned_objects=[],
                key="property_environment_map",
            )

            st.markdown(
                f"**📍 {valuation['resolved_address']}**"
            )

            st.caption(
                "La position affichée correspond à la localisation "
                "reconnue par le service de géocodage. Elle peut être "
                "résolue au niveau de la rue plutôt qu'au bâtiment exact."
            )

    # -----------------------------------------------------------------
    # NEIGHBORHOOD TAB
    # -----------------------------------------------------------------

    with neighborhood_tab:

        st.subheader("Explorer le quartier")

        st.markdown(
            """
            <div class="placeholder-card">
                <div class="placeholder-title">
                    👁️ Vue du quartier
                </div>
                <div class="placeholder-text">
                    Cette vue accueillera l'exploration visuelle
                    du quartier autour de l'adresse reconnue.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------
    # GENAI DESCRIPTION TAB
    # -----------------------------------------------------------------

    with description_tab:

        st.subheader("Description immobilière")

        st.markdown(
            """
            <div class="placeholder-card">
                <div class="placeholder-title">
                    ✨ Description générée
                </div>
                <div class="placeholder-text">
                    Une description pourra être générée à partir
                    des informations vérifiées du bien, de son
                    estimation et de son environnement.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )