#!/usr/bin/env python3
"""The house rule for thousands, in one place.

Lancet separates thousands with a space from five digits up and leaves four
digits unseparated: 7591 deaths, but 84 924 participants. The rule lived only
inside build_word.py's LaTeX-to-Word pass, so anything that formatted a count
without going through that pass printed commas instead. The study flow diagram
did exactly that, and shipped in the appendix with 205,794 against the
manuscript's 205 794 on the facing page.
"""
from __future__ import annotations

import re

NBSP = "\u00a0"
# One whole comma-grouped number, so a seven-digit count regroups as a unit.
# Matching a single ",ddd" group instead, which is what this rule used to do,
# turned 1,234,567 into 1234,567: the match landed on the last group only.
_GROUPED = re.compile(r"\b\d{1,3}(?:,\d{3})+\b(?!\d)")


def _regroup(m: re.Match) -> str:
    digits = m.group(0).replace(",", "")
    if len(digits) <= 4:
        return digits
    parts = []
    while len(digits) > 3:
        parts.insert(0, digits[-3:])
        digits = digits[:-3]
    parts.insert(0, digits)
    return NBSP.join(parts)


def thousands(text: str) -> str:
    """Rewrite every comma-grouped number in a block of text."""
    return _GROUPED.sub(_regroup, text)


def count(n: int) -> str:
    """Format one integer the way this journal prints it."""
    return thousands(f"{n:,}")


if __name__ == "__main__":
    for n in (999, 1000, 7591, 9999, 10097, 84924, 205794, 1234567):
        print(f"{n:>9,}  ->  {count(n)}")
