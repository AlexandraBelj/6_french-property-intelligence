"""
PURPOSE
-------
Main Streamlit interface for French Property Intelligence.

The application provides a property-intelligence journey:

1. The user describes a residential property.
2. Streamlit sends the information to the FastAPI service.
3. FastAPI geocodes the address and runs the frozen production model.
4. Streamlit presents the valuation in a compact dashboard.
5. The environment tab displays:
   - the verified property location;
   - nearby public transport;
   - nearby green spaces;
   - nearby supermarkets.
6. Additional tabs will progressively provide:
   - neighborhood / Street View exploration;
   - GenAI property description.

IMPORTANT
---------
Streamlit does NOT load model.pkl directly.

Production architecture:

Streamlit -> FastAPI -> IGN geocoding -> S3 model -> prediction
                    -> Geoapify Places -> nearby environment

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
ENVIRONMENT_URL = f"{API_URL}/environment"
DESCRIPTION_URL = f"{API_URL}/description"

# Standard timeout for valuation and contextual API services.
REQUEST_TIMEOUT_SECONDS = 30

# GenAI can require more time than the deterministic API services,
# especially when the external inference provider has a cold start.
GENERATION_TIMEOUT_SECONDS = 90


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


def format_distance(distance_m: int) -> str:
    """
    Format a nearby-place distance for compact display.

    Distances under one kilometre are displayed in metres.
    Longer distances are displayed in kilometres.
    """

    if distance_m < 1000:
        return f"{distance_m} m"

    return f"{distance_m / 1000:.1f} km".replace(".", ",")


def render_nearby_places(
    title: str,
    places: list[dict],
) -> None:
    """
    Render one compact POI category.

    PURPOSE:
    Keep the environment summary readable instead of displaying a large
    raw table of provider data.
    """

    st.markdown(f"#### {title}")

    if not places:
        st.caption("Aucun point d'intérêt proche identifié.")
        return

    for place in places:

        name = place.get("name", "Lieu")
        distance = place.get("distance_m")

        if isinstance(distance, (int, float)):
            distance_text = format_distance(int(distance))
        else:
            distance_text = "Distance indisponible"

        st.markdown(
            f"**{name}**  \n"
            f"📍 {distance_text}"
        )


# ---------------------------------------------------------------------
# Session state
#
# PURPOSE:
# Keep the latest successful valuation and its contextual information
# visible when Streamlit reruns.
# ---------------------------------------------------------------------

if "valuation_result" not in st.session_state:
    st.session_state.valuation_result = None

# PURPOSE:
# Store the generated description independently from the valuation.
# This prevents a GenAI request from being repeated every time
# Streamlit reruns because of another UI interaction.
if "property_description" not in st.session_state:
    st.session_state.property_description = None


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

    .environment-legend {
        border: 1px solid rgba(128, 128, 128, 0.20);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.8rem;
        font-size: 0.92rem;
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
                # Resolve coordinates for the base property map.
                #
                # A failure here must NOT invalidate the valuation.
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
                    pass

                # -----------------------------------------------------
                # Retrieve contextual environment information.
                #
                # PURPOSE:
                # Enrich the product with nearby transport, parks and
                # supermarkets without affecting ML inference.
                #
                # This service is optional. A failure must never erase
                # or invalidate a successful property valuation.
                # -----------------------------------------------------

                environment = None

                try:

                    environment_response = requests.post(
                        ENVIRONMENT_URL,
                        json={
                            "address": cleaned_address,
                        },
                        timeout=REQUEST_TIMEOUT_SECONDS,
                    )

                    if environment_response.status_code == 200:

                        environment = environment_response.json()

                except (
                    requests.RequestException,
                    ValueError,
                ):
                    pass

                # -----------------------------------------------------
                # Save one coherent property-analysis result.
                # A new valuation invalidates the previous description.
                # -----------------------------------------------------

                st.session_state.property_description = None

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
                    "environment": environment,
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
    # Combine the verified property position with nearby contextual POIs.
    #
    # The environment information is descriptive only. It is NOT fed
    # back into the frozen machine-learning valuation model.
    # -----------------------------------------------------------------

    with environment_tab:

        st.subheader("Environnement du bien")

        st.caption(
            "Explorez la localisation reconnue du bien et les principaux "
            "services à proximité."
        )

        valuation = st.session_state.valuation_result

        latitude = valuation.get("latitude")
        longitude = valuation.get("longitude")
        environment = valuation.get("environment")

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
            # Display the property and the nearest contextual POIs on
            # one readable map.
            # ---------------------------------------------------------

            property_map = folium.Map(
                location=[
                    latitude,
                    longitude,
                ],
                zoom_start=16,
                control_scale=True,
            )

            # ---------------------------------------------------------
            # Main property marker
            # ---------------------------------------------------------

            folium.Marker(
                location=[
                    latitude,
                    longitude,
                ],
                tooltip="Bien analysé",
                popup=valuation["resolved_address"],
                icon=folium.Icon(
                    color="red",
                    icon="home",
                    prefix="fa",
                ),
            ).add_to(property_map)

            # ---------------------------------------------------------
            # Contextual POI markers
            # ---------------------------------------------------------

            if environment:

                poi_categories = [
                    (
                        "transport",
                        "Transport",
                        "blue",
                        "train",
                    ),
                    (
                        "green_spaces",
                        "Espace vert",
                        "green",
                        "tree",
                    ),
                    (
                        "supermarkets",
                        "Supermarché",
                        "orange",
                        "shopping-cart",
                    ),
                ]

                for (
                    category_key,
                    category_label,
                    marker_color,
                    marker_icon,
                ) in poi_categories:

                    for place in environment.get(
                        category_key,
                        [],
                    ):

                        try:

                            poi_latitude = float(
                                place["latitude"]
                            )

                            poi_longitude = float(
                                place["longitude"]
                            )

                        except (
                            KeyError,
                            TypeError,
                            ValueError,
                        ):
                            continue

                        distance = place.get(
                            "distance_m"
                        )

                        if isinstance(
                            distance,
                            (int, float),
                        ):
                            distance_text = format_distance(
                                int(distance)
                            )
                        else:
                            distance_text = (
                                "Distance indisponible"
                            )

                        popup_text = (
                            f"{category_label} — "
                            f"{place.get('name', 'Lieu')} "
                            f"({distance_text})"
                        )

                        folium.Marker(
                            location=[
                                poi_latitude,
                                poi_longitude,
                            ],
                            tooltip=place.get(
                                "name",
                                category_label,
                            ),
                            popup=popup_text,
                            icon=folium.Icon(
                                color=marker_color,
                                icon=marker_icon,
                                prefix="fa",
                            ),
                        ).add_to(property_map)

            # ---------------------------------------------------------
            # Legend
            # ---------------------------------------------------------

            st.markdown(
                (
                    '<div class="environment-legend">'
                    '🏠 Bien analysé &nbsp;&nbsp; '
                    '🚇 Transport &nbsp;&nbsp; '
                    '🌳 Espaces verts &nbsp;&nbsp; '
                    '🛒 Supermarchés'
                    '</div>'
                ),
                unsafe_allow_html=True,
            )

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

            # ---------------------------------------------------------
            # Nearby environment summary
            # ---------------------------------------------------------

            if environment:

                st.markdown("### À proximité")

                transport_column, green_column, shop_column = st.columns(
                    3,
                    gap="large",
                )

                with transport_column:

                    render_nearby_places(
                        "🚇 Transports",
                        environment.get(
                            "transport",
                            [],
                        ),
                    )

                with green_column:

                    render_nearby_places(
                        "🌳 Espaces verts",
                        environment.get(
                            "green_spaces",
                            [],
                        ),
                    )

                with shop_column:

                    render_nearby_places(
                        "🛒 Supermarchés",
                        environment.get(
                            "supermarkets",
                            [],
                        ),
                    )

                st.caption(
                    "Distances indicatives fournies à partir de la "
                    "localisation reconnue du bien."
                )

            else:

                st.info(
                    "Les informations de proximité sont temporairement "
                    "indisponibles. La carte et l'estimation restent utilisables."
                )

    # -----------------------------------------------------------------
    # NEIGHBORHOOD TAB
    #
    # PURPOSE:
    # Allow the user to visually explore the neighborhood around the
    # verified property location using Google Street View.
    #
    # Street View is deliberately kept outside the valuation model.
    # The application sends only the verified geographic coordinates
    # through a standard Google Maps URL; no Google API key is required.
    # -----------------------------------------------------------------

    with neighborhood_tab:

        st.subheader("Explorer le quartier")

        st.caption(
            "Visualisez les rues et l'environnement autour de la "
            "localisation reconnue du bien."
        )

        valuation = st.session_state.valuation_result

        latitude = valuation.get("latitude")
        longitude = valuation.get("longitude")

        if latitude is None or longitude is None:

            st.warning(
                "La localisation du bien est temporairement indisponible. "
                "L'exploration du quartier ne peut pas être ouverte."
            )

        else:

            # ---------------------------------------------------------
            # Google Maps Street View URL
            #
            # PURPOSE:
            # Open the Street View panorama nearest to the verified
            # property coordinates without requiring another API,
            # backend service or application secret.
            # ---------------------------------------------------------

            street_view_url = (
                "https://www.google.com/maps/@"
                "?api=1"
                "&map_action=pano"
                f"&viewpoint={latitude},{longitude}"
            )

            st.markdown(
                """
                <div class="placeholder-card">
                    <div class="placeholder-title">
                        👁️ Explorer le quartier en Street View
                    </div>
                    <div class="placeholder-text">
                        Parcourez visuellement les rues autour du bien
                        dans Google Street View.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.link_button(
                "🌐 Ouvrir Street View",
                street_view_url,
                use_container_width=True,
                type="primary",
            )

            st.markdown(
                f"**📍 Point de départ : "
                f"{valuation['resolved_address']}**"
            )

            st.caption(
                "Street View ouvre le panorama disponible le plus proche "
                "des coordonnées reconnues. La vue peut donc se situer "
                "à proximité du bien plutôt que devant le bâtiment exact."
            )
            
    # -----------------------------------------------------------------
    # GENAI DESCRIPTION TAB
    #
    # PURPOSE:
    # Generate a concise real-estate description only from facts already
    # verified by the valuation and contextual services. GenAI does not
    # calculate the property value and failure here never invalidates it.
    # -----------------------------------------------------------------

    with description_tab:

        st.subheader("Description immobilière")

        st.caption(
            "Générez une présentation du bien à partir de ses "
            "caractéristiques, de l'estimation et des informations "
            "d'environnement disponibles."
        )

        valuation = st.session_state.valuation_result
        environment = valuation.get("environment") or {}

        def description_places(category: str) -> list[dict]:
            """Return a compact POI list for the GenAI API."""

            places = environment.get(category, [])
            return [
                {
                    "name": place.get("name", "Lieu"),
                    "distance_m": place.get("distance_m"),
                }
                for place in places[:3]
            ]

        description_payload = {
            "property_type": valuation["property_type"],
            "surface_habitable": valuation["surface_habitable"],
            "n_pieces": valuation["n_pieces"],
            "vefa": valuation["vefa"],
            "resolved_address": valuation["resolved_address"],
            "estimated_price_eur": valuation["estimated_price"],
            "environment": {
                "transport": description_places("transport"),
                "green_spaces": description_places("green_spaces"),
                "supermarkets": description_places("supermarkets"),
            },
        }

        button_label = (
            "✨ Générer la description"
            if st.session_state.property_description is None
            else "✨ Régénérer la description"
        )

        if st.button(
            button_label,
            type="primary",
            use_container_width=True,
        ):

            try:
                with st.spinner(
                    "Génération de la description immobilière..."
                ):
                    description_response = requests.post(
                        DESCRIPTION_URL,
                        json=description_payload,
                        timeout=GENERATION_TIMEOUT_SECONDS,
                    )

                if description_response.status_code == 200:
                    description_result = description_response.json()
                    generated_description = description_result["description"]

                    if (
                        not isinstance(generated_description, str)
                        or not generated_description.strip()
                    ):
                        st.error(
                            "La description reçue est invalide. "
                            "Veuillez réessayer."
                        )
                    else:
                        st.session_state.property_description = (
                            generated_description.strip()
                        )

                elif description_response.status_code == 503:
                    st.warning(
                        "La génération de description est temporairement "
                        "indisponible. Votre estimation reste disponible."
                    )
                else:
                    st.error(
                        "La description n'a pas pu être générée. "
                        "Veuillez réessayer."
                    )

            except requests.Timeout:
                st.warning(
                    "La génération prend plus de temps que prévu. "
                    "Veuillez réessayer dans quelques instants."
                )
            except requests.ConnectionError:
                st.warning(
                    "Le service de génération est temporairement "
                    "inaccessible. Votre estimation reste disponible."
                )
            except (
                requests.RequestException,
                KeyError,
                TypeError,
                ValueError,
            ):
                st.error(
                    "Une erreur est survenue pendant la génération "
                    "de la description."
                )

        if st.session_state.property_description:
            st.markdown("---")
            st.markdown(st.session_state.property_description)
            st.caption(
                "✨ Texte généré automatiquement à partir des informations "
                "disponibles. L'estimation reste indicative et ne constitue "
                "pas une expertise immobilière."
            )
