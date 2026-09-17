"""Output guardrails: structured-output validation and leak checks.

The RETRY action is what turns the diagram's arrow from "Output guardrails"
back into "Context construction" into something real -- a failed check hands
feedback to the next iteration instead of just refusing.
"""

from __future__ import annotations

import json
import re
from typing import Any

from aie.guardrails.input import PII_PATTERNS
from aie.types import GuardrailAction, GuardrailResult

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


class StructuredOutput:
    """Parses and validates JSON output against a shallow schema.

    Deliberately not a full JSON Schema implementation -- it checks type,
    required keys and property types, which is what catches the failures that
    actually happen. Swap in ``jsonschema`` when you need the rest.
    """

    name = "structured_output"
    blocking = True
    fail_closed = True

    def __init__(self, schema: dict[str, Any] | None, *, max_repairs: int = 1) -> None:
        self.schema = schema
        self.max_repairs = max_repairs

    def check(self, content: str, *, context: dict | None = None) -> GuardrailResult:
        if not self.schema:
            return GuardrailResult(self.name, GuardrailAction.ALLOW, content)

        iteration = (context or {}).get("iteration", 1)
        payload = content.strip()
        fenced = _JSON_FENCE.search(payload)
        if fenced:
            payload = fenced.group(1)

        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            return self._fail(content, f"output is not valid JSON ({exc.msg})", iteration)

        errors = _validate(parsed, self.schema)
        if errors:
            return self._fail(content, "; ".join(errors), iteration)

        return GuardrailResult(
            name=self.name,
            action=GuardrailAction.ALLOW,
            content=json.dumps(parsed),
            message="schema satisfied",
        )

    def _fail(self, content: str, reason: str, iteration: int) -> GuardrailResult:
        # Retry once with feedback, then stop: a model that has failed the
        # schema twice is not going to find it on the third pass, and each
        # round is another full generation the user is waiting for.
        action = (
            GuardrailAction.RETRY if iteration <= self.max_repairs else GuardrailAction.BLOCK
        )
        return GuardrailResult(
            name=self.name,
            action=action,
            content=content,
            findings=[reason],
            message=f"Your previous response failed validation: {reason}. "
            f"Respond with JSON matching the requested schema and nothing else.",
        )


_TYPES: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
}


def _type_error(value: Any, expected: str, path: str) -> str | None:
    """Type check with the bool/int trap handled.

    ``isinstance(True, int)`` is True in Python, so a naive check lets
    ``{"confidence": true}`` satisfy ``{"type": "number"}``.
    """
    if expected not in _TYPES:
        return None
    if expected in ("number", "integer") and isinstance(value, bool):
        return f"{path}: expected {expected}, got bool"
    if isinstance(value, _TYPES[expected]):
        return None
    return f"{path}: expected {expected}, got {type(value).__name__}"


def _validate(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    expected = schema.get("type")
    if isinstance(expected, str):
        error = _type_error(value, expected, path)
        if error:
            return [error]  # wrong type: sub-checks would be noise

    errors: list[str] = []
    if expected == "object":
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required key {key!r}")
        for key, subschema in (schema.get("properties") or {}).items():
            if key in value:
                errors.extend(_validate(value[key], subschema, f"{path}.{key}"))
    elif expected == "array" and "items" in schema:
        for index, item in enumerate(value):
            errors.extend(_validate(item, schema["items"], f"{path}[{index}]"))
    return errors


class PIILeakCheck:
    """Blocks PII in the response that did not come from the user's own input.

    Retrieval can surface another customer's record; the model will happily
    quote it. Anything matching a PII pattern that was not already present in
    the (pre-redaction) input is treated as a leak.
    """

    name = "pii_leak"
    blocking = True
    fail_closed = True

    def check(self, content: str, *, context: dict | None = None) -> GuardrailResult:
        source = (context or {}).get("input_text", "")
        leaked: list[str] = []
        for label, pattern in PII_PATTERNS:
            for match in pattern.finditer(content):
                if match.group(0) not in source:
                    leaked.append(f"{label}:{match.group(0)[:4]}...")

        if not leaked:
            return GuardrailResult(self.name, GuardrailAction.ALLOW, content)
        return GuardrailResult(
            name=self.name,
            action=GuardrailAction.BLOCK,
            content="I can't share that -- the answer contained personal data I'm not able to return.",
            findings=leaked,
            message=f"blocked {len(leaked)} leaked PII span(s)",
        )


class NonEmptyOutput:
    """Catches the silent failure: a successful call that returned nothing."""

    name = "non_empty"
    blocking = True
    fail_closed = True

    def check(self, content: str, *, context: dict | None = None) -> GuardrailResult:
        if content.strip():
            return GuardrailResult(self.name, GuardrailAction.ALLOW, content)
        iteration = (context or {}).get("iteration", 1)
        return GuardrailResult(
            name=self.name,
            action=GuardrailAction.RETRY if iteration <= 1 else GuardrailAction.BLOCK,
            content=content,
            findings=["empty response"],
            message="The previous response was empty. Answer the question directly.",
        )
