"""
engine.py  -  intent-agnostic template expansion machinery.
================================================================================
The reusable part: given a template string and a slot dict, produce filled
sentences. Knows nothing about which intents exist -- it just resolves {slot}
(lookup) and {a|b|c} (inline choice) tokens. Kept separate so it can be tested
on its own, e.g. "does {a|b} only ever pick a or b?".
"""

import random
import re

_TOKEN = re.compile(r"\{([^{}]+)\}")


def fill(template: str, slots: dict, rng: random.Random) -> str:
    """Replace {slot} (lookup) or {a|b|c} (inline choice) tokens; tidy spaces."""

    def repl(m):
        body = m.group(1)
        if "|" in body:
            return rng.choice(body.split("|"))
        return rng.choice(slots[body])

    return re.sub(r"\s+", " ", _TOKEN.sub(repl, template)).strip()


def expand(template: str, slots: dict, rng: random.Random, k: int) -> list:
    """Up to k UNIQUE fillings of a template (fewer if combos run out)."""
    seen, out, attempts = set(), [], 0
    while len(out) < k and attempts < k * 12:
        attempts += 1
        s = fill(template, slots, rng)
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def validate_templates(templates, base_slots) -> None:
    """Every {slot} (non-inline) token in `templates` must exist in `base_slots`."""
    for intent, splits in templates.items():
        for split, tmpls in splits.items():
            for t in tmpls:
                for body in _TOKEN.findall(t):
                    if "|" in body:
                        continue
                    assert body in base_slots, (
                        f"unknown slot '{{{body}}}' in {intent}/{split}: {t!r}"
                    )
