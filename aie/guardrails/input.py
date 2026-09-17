"""Input guardrails: PII redaction and prompt-injection heuristics."""

from __future__ import annotations

import re

from aie.types import GuardrailAction, GuardrailResult

# Ordered: credit cards and SSNs before the generic long-number patterns.
PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Order matters. Structured secrets first: the phone pattern will happily
    # eat a ten-digit run out of the middle of an API key otherwise.
    ("api_key", re.compile(r"\b(?:sk|pk|ghp|gho|xox[baprs])[-_][A-Za-z0-9_-]{16,}\b")),
    ("email", re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
    # Word-boundary lookarounds, not just digit ones, so an identifier that
    # happens to contain ten digits is left alone.
    ("phone", re.compile(r"(?<![\w-])(?:\+\d{1,3}[ -]?)?(?:\(\d{3}\)|\d{3})[ -]?\d{3}[ -]?\d{4}(?![\w-])")),
    ("ipv4", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]


def _luhn(digits: str) -> bool:
    """Card-number checksum, so we don't redact every long order number."""
    total, parity = 0, len(digits) % 2
    for index, char in enumerate(digits):
        digit = int(char)
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


class PIIRedaction:
    """Redacts PII in place rather than blocking.

    Redaction, not refusal, is the right default: the user asked a legitimate
    question that happens to contain their own phone number, and blocking it
    teaches them to work around you.
    """

    name = "pii_redaction"
    blocking = True
    fail_closed = True

    def __init__(self, patterns: list[tuple[str, re.Pattern[str]]] | None = None) -> None:
        self._patterns = patterns or PII_PATTERNS

    def check(self, content: str, *, context: dict | None = None) -> GuardrailResult:
        findings: list[str] = []
        redacted = content

        for label, pattern in self._patterns:
            def replace(match: re.Match[str], label: str = label) -> str:
                text = match.group(0)
                if label == "credit_card":
                    digits = re.sub(r"\D", "", text)
                    if not (13 <= len(digits) <= 19 and _luhn(digits)):
                        return text
                findings.append(label)
                return f"[REDACTED_{label.upper()}]"

            redacted = pattern.sub(replace, redacted)

        if not findings:
            return GuardrailResult(self.name, GuardrailAction.ALLOW, content)
        return GuardrailResult(
            name=self.name,
            action=GuardrailAction.MODIFY,
            content=redacted,
            findings=sorted(set(findings)),
            message=f"redacted {len(findings)} PII span(s)",
        )


INJECTION_MARKERS = [
    re.compile(r"ignore (?:all |any )?(?:previous|prior|above) instructions", re.I),
    re.compile(r"disregard (?:your|the) (?:system )?prompt", re.I),
    re.compile(r"reveal (?:your|the) (?:system )?prompt", re.I),
    re.compile(r"you are now (?:a|an|in) [a-z ]{0,24}mode", re.I),
]


class PromptInjectionHeuristic:
    """Cheap regex pre-filter, shipped in shadow mode.

    It exists to collect data, not to gate traffic: regex injection detection
    has a false-positive rate that will block legitimate questions about prompt
    engineering. Promote it to ``blocking=True`` only once the shadow numbers
    justify it.
    """

    name = "prompt_injection"
    blocking = False
    fail_closed = False

    def check(self, content: str, *, context: dict | None = None) -> GuardrailResult:
        hits = [p.pattern for p in INJECTION_MARKERS if p.search(content)]
        if not hits:
            return GuardrailResult(self.name, GuardrailAction.ALLOW, content)
        return GuardrailResult(
            name=self.name,
            action=GuardrailAction.BLOCK,
            content=content,
            findings=hits,
            message="input resembles a prompt-injection attempt",
        )
