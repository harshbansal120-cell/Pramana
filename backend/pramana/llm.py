"""LLM access behind three roles.

generator - writes tutor answers, questions and the concept graph
verifier  - independently solves and judges what the generator wrote (cross-model validation)
bulk      - high-volume work: figure descriptions, query rewriting

Each role is a "provider:model" string from the environment, so the verifier can be a
different model family from the generator. All calls return JSON matching a schema.
"""
from __future__ import annotations

import base64
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import config


class LLMError(RuntimeError):
    pass


class LLMUnavailable(LLMError):
    """No credentials for the requested role."""


def obj(**props: dict) -> dict:
    """Strict object schema: every property required, nothing extra."""
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def arr(items: dict) -> dict:
    return {"type": "array", "items": items}


STR = {"type": "string"}
INT = {"type": "integer"}
NUM = {"type": "number"}
BOOL = {"type": "boolean"}
def nullable(schema: dict) -> dict:
    return {"anyOf": [schema, {"type": "null"}]}


NSTR = nullable(STR)
NNUM = nullable(NUM)


def enum(*values: str) -> dict:
    return {"type": "string", "enum": list(values)}


class _Anthropic:
    def __init__(self, model: str):
        import anthropic

        self._sdk = anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def json(self, system: str, user: str, schema: dict, images: list[Path], max_tokens: int) -> dict:
        content: list[dict] = []
        for path in images:
            data = base64.standard_b64encode(Path(path).read_bytes()).decode("utf-8")
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}})
        content.append({"type": "text", "text": user})
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": content}],
                output_config={"format": {"type": "json_schema", "schema": schema}},
            )
        except self._sdk.AuthenticationError as e:
            raise LLMUnavailable(f"Anthropic credentials rejected: {e}") from e
        except self._sdk.APIStatusError as e:
            raise LLMError(f"Anthropic API error {e.status_code}: {e.message}") from e
        except self._sdk.APIConnectionError as e:
            raise LLMError(f"Could not reach the Anthropic API: {e}") from e
        if resp.stop_reason == "refusal":
            raise LLMError("The model declined this request.")
        if resp.stop_reason == "max_tokens":
            raise LLMError("The model ran out of output tokens before finishing.")
        text = next(b.text for b in resp.content if b.type == "text")
        return json.loads(text)


class _Google:
    def __init__(self, model: str):
        from google import genai
        from google.genai import types

        self.client = genai.Client()
        self.types = types
        self.model = model

    def json(self, system: str, user: str, schema: dict, images: list[Path], max_tokens: int) -> dict:
        types = self.types
        parts: list[Any] = [types.Part.from_bytes(data=Path(p).read_bytes(), mime_type="image/png") for p in images]
        parts.append(user + "\n\nReturn only JSON matching this JSON Schema:\n" + json.dumps(schema))
        last: Exception | None = None
        for _ in range(2):  # schema is prompt-enforced here, so allow one retry on malformed output
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=parts,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        response_mime_type="application/json",
                        max_output_tokens=max_tokens,
                    ),
                )
                data = json.loads(resp.text)
                missing = [k for k in schema.get("required", []) if k not in data]
                if missing:
                    raise ValueError(f"missing keys {missing}")
                return data
            except (ValueError, TypeError) as e:
                last = e
            except Exception as e:  # SDK raises several unrelated error types
                raise LLMError(f"Gemini API error: {e}") from e
        raise LLMError(f"Gemini returned malformed JSON: {last}")


_PROVIDERS = {"anthropic": _Anthropic, "google": _Google}


@lru_cache(maxsize=None)
def _provider(spec: str):
    name, _, model = spec.partition(":")
    if name not in _PROVIDERS or not model:
        raise LLMError(f"Bad model spec '{spec}'. Use provider:model with provider in {sorted(_PROVIDERS)}.")
    try:
        return _PROVIDERS[name](model)
    except Exception as e:
        raise LLMUnavailable(f"Cannot initialise {spec}: {e}") from e


def spec_for(role: str) -> str:
    return {"generator": config.GENERATOR, "verifier": config.VERIFIER, "bulk": config.BULK}[role]


def available(role: str = "generator") -> bool:
    """True when the role's provider has credentials. Does not make a network call."""
    import os

    name = spec_for(role).partition(":")[0]
    if name == "anthropic":
        return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
    if name == "google":
        return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    return False


def text(role: str, system: str, user: str, max_tokens: int = 8000) -> str:
    """One free-text completion. Used by the evaluation judge, which parses its own output format."""
    spec = spec_for(role)
    if not available(role):
        raise LLMUnavailable(f"No API key for the {role} model ({spec}). Add it to .env.")
    provider = _provider(spec)
    if isinstance(provider, _Anthropic):
        try:
            resp = provider.client.messages.create(
                model=provider.model, max_tokens=max_tokens, messages=[{"role": "user", "content": user}],
                **({"system": system} if system else {}))
        except provider._sdk.APIStatusError as e:
            raise LLMError(f"Anthropic API error {e.status_code}: {e.message}") from e
        except provider._sdk.APIConnectionError as e:
            raise LLMError(f"Could not reach the Anthropic API: {e}") from e
        if resp.stop_reason == "refusal":
            raise LLMError("The model declined this request.")
        return "".join(b.text for b in resp.content if b.type == "text")
    try:
        resp = provider.client.models.generate_content(
            model=provider.model, contents=user,
            config=provider.types.GenerateContentConfig(system_instruction=system or None, max_output_tokens=max_tokens))
        return resp.text or ""
    except Exception as e:
        raise LLMError(f"Gemini API error: {e}") from e


def ask(
    role: str,
    system: str,
    user: str,
    schema: dict,
    *,
    images: list[Path] | None = None,
    max_tokens: int = 16000,
    cache: bool = False,
) -> dict:
    """Run one structured call. With cache=True the response is stored on disk by content hash."""
    spec = spec_for(role)
    images = [Path(p) for p in (images or [])]
    key_path: Path | None = None
    if cache:
        h = hashlib.sha256()
        for part in (spec, system, user, json.dumps(schema, sort_keys=True)):
            h.update(part.encode("utf-8"))
        for p in images:
            h.update(p.read_bytes())
        key_path = config.CACHE / f"{h.hexdigest()}.json"
        if key_path.exists():
            return json.loads(key_path.read_text(encoding="utf-8"))
    if not available(role):
        raise LLMUnavailable(f"No API key for the {role} model ({spec}). Add it to .env.")
    out = _provider(spec).json(system, user, schema, images, max_tokens)
    if key_path:
        key_path.write_text(json.dumps(out), encoding="utf-8")
    return out
