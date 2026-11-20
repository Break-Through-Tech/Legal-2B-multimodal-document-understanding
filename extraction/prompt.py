"""The extraction prompt, and turning the model's reply back into JSON (issue #10).

No model code here, so it can be tested without a GPU (tests/test_extraction.py).
"""

import json
import re
from pathlib import Path

from PIL import Image

SCHEMAS_DIR = Path(__file__).resolve().parents[1] / "schemas" / "classes"


def load_schema(doc_type: str) -> dict:
    """The per-type schema from schemas/classes/ (built in PR #4)."""
    return json.loads((SCHEMAS_DIR / f"{doc_type}.json").read_text())


def build_prompt(doc_type: str, schema: dict) -> str:
    """What the model reads next to the page image."""
    name = doc_type.replace("_", " ").lower()
    return (
        f"This image is a {name} document.\n"
        "Fill in the JSON schema below with the data in the image.\n"
        "- Copy names, numbers and dates exactly as they appear.\n"
        "- Use null for a field that is not in the document.\n"
        "- Reply with the JSON object only, no other text.\n\n"
        f"JSON schema:\n{json.dumps(schema)}"
    )


def parse_reply(text: str) -> dict | None:
    """The JSON object in the model's reply, or None if there isn't a valid one.

    Small models often wrap the JSON in ```json fences or add a sentence around it, so this
    keeps only the text from the first { to the last }.
    """
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def shrink_to(image: Image.Image, max_pixels: int) -> Image.Image:
    """Scale the image down (keeping its shape) so it has at most `max_pixels` pixels.

    The model turns every 32x32 block of pixels into one token, so this caps the
    number of image tokens. Smaller images are only converted to RGB.
    """
    image = image.convert("RGB")
    pixels = image.width * image.height
    if pixels <= max_pixels:
        return image
    scale = (max_pixels / pixels) ** 0.5
    return image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.LANCZOS)
