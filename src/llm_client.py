"""Thin wrapper around the Google GenAI SDK with retry / back-off.

Every LLM call in the project flows through ``generate_text`` so that
rate-limit (429 / RESOURCE_EXHAUSTED) and transient server errors are
retried transparently with exponential back-off.
"""

import json
import logging
import re
import time

from google import genai
from google.genai import types

from src.config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Singleton client
# ------------------------------------------------------------------
_client = genai.Client(api_key=GEMINI_API_KEY)


# ------------------------------------------------------------------
# Core generation
# ------------------------------------------------------------------
def generate_text(
    prompt: str,
    system_instruction: str,
    model_id: str,
    temperature: float = 0.0,
    max_retries: int = 5,
) -> str:
    """Generate text from Gemini with exponential-backoff retry on 429s.

    Parameters
    ----------
    prompt : str
        The user / task prompt.
    system_instruction : str
        System-level instruction prepended by the API.
    model_id : str
        Gemini model identifier (e.g. ``gemini-2.5-flash``).
    temperature : float
        Sampling temperature (0.0 = greedy).
    max_retries : int
        Maximum number of retry attempts on rate-limit / transient errors.

    Returns
    -------
    str
        Raw text response from the model.
    """
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
    )

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = _client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=config,
            )
            return response.text or ""

        except Exception as exc:                               # noqa: BLE001
            last_error = exc
            err_str = str(exc).upper()

            retryable = (
                "429" in str(exc)
                or "RESOURCE_EXHAUSTED" in err_str
                or "500" in str(exc)
                or "503" in str(exc)
                or "UNAVAILABLE" in err_str
            )
            if retryable and attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning(
                    "API error (attempt %d/%d): %s — sleeping %ds …",
                    attempt + 1,
                    max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
                continue

            raise

    raise RuntimeError(
        f"LLM call failed after {max_retries} attempts: {last_error}"
    )


# ------------------------------------------------------------------
# JSON extraction helpers (used by engineered_full mode)
# ------------------------------------------------------------------
def extract_json(text: str) -> dict | list:
    """Best-effort JSON extraction from raw LLM output.

    Tries in order:
      1. Direct ``json.loads``.
      2. Strip Markdown code fences and retry.
      3. Regex for the first ``{…}`` or ``[…]`` block.
    Raises ``ValueError`` if all strategies fail.
    """
    text = text.strip()

    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. Regex first JSON block
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", cleaned)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from LLM output:\n{text[:500]}")
