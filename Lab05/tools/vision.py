"""Vision tool — describes/answers about an image via a multimodal
Ollama model (default: qwen2.5vl:3b)."""
import os

from config import VISION_MAX_IMAGE_BYTES, VISION_MODEL
import ollama_client

_ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def analyze_image(image_path, prompt=None):
    """Describe or answer a question about an image."""
    if not isinstance(image_path, str) or not image_path:
        return {"error": "image_path is required."}
    if not os.path.isfile(image_path):
        return {"error": f"File not found: {image_path}"}
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in _ALLOWED_IMAGE_EXTENSIONS:
        return {"error": f"Unsupported image extension: {ext or '(none)'}"}
    try:
        size = os.path.getsize(image_path)
    except OSError as e:
        return {"error": f"Cannot read image file: {e}"}
    if size > VISION_MAX_IMAGE_BYTES:
        return {"error": f"Image is too large: {size} bytes."}
    user_prompt = (
        prompt
        or "Describe this image in 2-4 sentences. Mention objects, "
           "people, setting, colours, and any text visible."
    )
    try:
        resp = ollama_client.chat(
            messages=[{"role": "user", "content": user_prompt}],
            model=VISION_MODEL,
            images=[image_path],
            temperature=0.3,
        )
    except Exception as e:
        return {
            "error": (
                f"Vision call failed: {e}. Make sure the model "
                f"'{VISION_MODEL}' is pulled: `ollama pull {VISION_MODEL}`."
            ),
        }
    msg = (resp or {}).get("message") or {}
    description = (msg.get("content") or "").strip()
    return {
        "image_path": image_path,
        "prompt": user_prompt,
        "model": VISION_MODEL,
        "description": description or "(empty response)",
    }


SCHEMA = {
    "type": "function",
    "function": {
        "name": "analyze_image",
        "description": (
            "Analyse an image attached by the user. Returns a textual "
            "description of its contents. Use this whenever the user "
            "refers to a picture/photo. The image_path is provided to "
            "the agent automatically when the user attaches a photo."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Local path to the image file.",
                },
                "prompt": {
                    "type": "string",
                    "description": (
                        "Optional question about the image, e.g. "
                        "'What city is this?' or 'Read the text.'."
                    ),
                },
            },
            "required": ["image_path"],
        },
    },
}
