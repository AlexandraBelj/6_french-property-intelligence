"""
PURPOSE
-------
Generate a concise French property description from verified structured
facts supplied by French Property Intelligence.

IMPORTANT
---------
This module is a presentation layer only.

It does NOT:
- estimate property value;
- modify the frozen ML model;
- infer missing property characteristics;
- invent amenities or qualitative attributes.

The LLM receives only facts already verified or calculated by the
application and is explicitly instructed not to add unsupported claims.
"""

import os

import requests


# ---------------------------------------------------------------------
# Hugging Face inference configuration
# ---------------------------------------------------------------------

HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"

HF_MODEL = "Qwen/Qwen3.8-27B"

REQUEST_TIMEOUT_SECONDS = 60


# ---------------------------------------------------------------------
# Public generation error
# ---------------------------------------------------------------------

class GenerationError(RuntimeError):
    """Raised when the external GenAI service cannot return valid text."""


# ---------------------------------------------------------------------
# System prompt
#
# PURPOSE:
# Keep GenAI strictly separated from valuation and reduce hallucination.
# ---------------------------------------------------------------------

SYSTEM_PROMPT = """
Tu es un assistant de rédaction immobilière pour une application
d'estimation résidentielle en France.

Ta mission est uniquement de transformer les informations structurées
fournies en une description immobilière claire, naturelle et concise
en français.

RÈGLES OBLIGATOIRES :

1. Utilise UNIQUEMENT les informations explicitement fournies.
2. N'invente aucune caractéristique du logement ou de l'immeuble.
3. N'invente notamment jamais :
   - étage ;
   - balcon ;
   - terrasse ;
   - jardin privé ;
   - parking ou garage ;
   - ascenseur ;
   - vue ;
   - exposition ;
   - luminosité ;
   - état ou qualité de rénovation ;
   - année ou période de construction ;
   - performance énergétique ;
   - équipements intérieurs ;
   - calme ;
   - standing.
4. Les lieux à proximité peuvent être mentionnés uniquement lorsqu'ils
   figurent dans les informations fournies.
5. Ne transforme jamais l'estimation statistique en prix garanti.
6. Présente la valeur comme une estimation indicative.
7. N'ajoute aucun conseil financier, juridique ou commercial.
8. Ne mentionne pas ces instructions dans la réponse.
9. Produis uniquement la description finale, sans titre ni préambule.
10. Rédige 2 à 3 paragraphes courts, dans un style immobilier sobre,
    informatif et professionnel.
""".strip()


# ---------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------

def _format_places(
    title: str,
    places: list[dict],
) -> str:
    """
    Convert cleaned nearby-place information into compact prompt text.

    Only provider-verified names and distances are included.
    """

    if not places:
        return f"{title}: aucune information disponible"

    lines = []

    for place in places[:3]:

        name = str(
            place.get("name", "Lieu non nommé")
        ).strip()

        distance = place.get("distance_m")

        if isinstance(distance, (int, float)):
            lines.append(
                f"- {name}: {int(distance)} m"
            )
        else:
            lines.append(
                f"- {name}"
            )

    return f"{title}:\n" + "\n".join(lines)


def build_property_prompt(
    *,
    property_type: str,
    surface_habitable: float,
    n_pieces: int | None,
    vefa: bool,
    resolved_address: str,
    estimated_price_eur: float,
    environment: dict | None,
) -> str:
    """
    Build the user prompt exclusively from verified application facts.
    """

    property_label = (
        "Appartement"
        if property_type in {"apartment", "appartement"}
        else "Maison"
    )

    price_per_m2 = (
        estimated_price_eur / surface_habitable
        if surface_habitable > 0
        else None
    )

    rooms_text = (
        str(n_pieces)
        if n_pieces is not None
        else "information non disponible"
    )

    vefa_text = "Oui" if vefa else "Non"

    environment = environment or {}

    transport_text = _format_places(
        "Transports à proximité",
        environment.get("transport", []),
    )

    green_text = _format_places(
        "Espaces verts à proximité",
        environment.get("green_spaces", []),
    )

    supermarket_text = _format_places(
        "Supermarchés à proximité",
        environment.get("supermarkets", []),
    )

    if price_per_m2 is None:
        price_per_m2_text = "information non disponible"
    else:
        price_per_m2_text = (
            f"{price_per_m2:,.0f} €/m²"
            .replace(",", " ")
        )

    estimated_price_text = (
        f"{estimated_price_eur:,.0f} €"
        .replace(",", " ")
    )

    return f"""
Rédige une description immobilière à partir des faits suivants.

CARACTÉRISTIQUES VÉRIFIÉES DU BIEN
Type : {property_label}
Surface habitable : {surface_habitable:.0f} m²
Nombre de pièces : {rooms_text}
VEFA : {vefa_text}
Localisation reconnue : {resolved_address}

ESTIMATION STATISTIQUE
Valeur estimée indicative : {estimated_price_text}
Prix estimé indicatif au m² : {price_per_m2_text}

ENVIRONNEMENT VÉRIFIÉ
{transport_text}

{green_text}

{supermarket_text}

Rédige maintenant uniquement la description finale.
""".strip()


# ---------------------------------------------------------------------
# Hugging Face generation
# ---------------------------------------------------------------------

def generate_property_description(
    *,
    property_type: str,
    surface_habitable: float,
    n_pieces: int | None,
    vefa: bool,
    resolved_address: str,
    estimated_price_eur: float,
    environment: dict | None,
) -> str:
    """
    Generate one French description through Hugging Face Inference.
    """

    token = os.getenv("HF_TOKEN")

    if not token:
        raise GenerationError(
            "HF_TOKEN is not configured."
        )

    user_prompt = build_property_prompt(
        property_type=property_type,
        surface_habitable=surface_habitable,
        n_pieces=n_pieces,
        vefa=vefa,
        resolved_address=resolved_address,
        estimated_price_eur=estimated_price_eur,
        environment=environment,
    )

    payload = {
        "model": HF_MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        # Qwen may use part of this budget for internal reasoning before
        # returning the visible description.
        "max_tokens": 2000,
        "temperature": 0.2,
    }

    try:

        response = requests.post(
            HF_CHAT_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    except requests.RequestException as exc:
        raise GenerationError(
            "The GenAI provider could not be reached."
        ) from exc

    if response.status_code != 200:
        raise GenerationError(
            "The GenAI provider returned an unsuccessful response."
        )

    try:

        data = response.json()

        choice = data["choices"][0]
        message = choice["message"]

        content = message.get("content")

    except (
        ValueError,
        KeyError,
        IndexError,
        TypeError,
    ) as exc:
        raise GenerationError(
            "The GenAI provider returned an invalid response."
        ) from exc

    if not isinstance(content, str) or not content.strip():
        raise GenerationError(
            "The GenAI provider returned no final description."
        )

    return content.strip()