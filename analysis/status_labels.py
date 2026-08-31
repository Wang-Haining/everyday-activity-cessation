"""Turn pipeline status codes into something a reader can act on.

The support gates emit values like NOT_EVALUABLE_EPV. They are the right thing
to record in an artifact and the wrong thing to print in a table: a reader has
no way to know that EPV is events per fitted parameter, or that CLUSTERS means
respondents. Every code that can reach a published table is mapped here, and
anything unmapped falls back to a reason rather than to the raw code.
"""
from __future__ import annotations

EXACT = {
    "NOT_EVALUABLE_EPV": "Too few events per fitted parameter",
    "NOT_EVALUABLE_TOTAL_EVENTS_LT_EPV": "Too few events per fitted parameter",
    "NOT_EVALUABLE_INTERACTION_EPV": "Too few events per fitted parameter",
    "NOT_EVALUABLE_CLUSTERS": "Fewer than 30 respondents",
    "NOT_EVALUABLE_META_K1": "Fewer than three cohorts",
    # The pipeline pools two cohorts descriptively and says so. Reading it as a
    # plain PASS printed it as an estimate and dropped the caveat.
    "PASS_DESCRIPTIVE_K2": "Fewer than three cohorts",
    "NOT_EVALUABLE_META_K": "Fewer than three cohorts",
    "NOT_EVALUABLE_EMPTY": "No eligible person-intervals",
    "NOT_EVALUABLE_CV_SUPPORT": "Too few events to cross-validate",
    "NOT_EVALUABLE_RETIREMENT_NOT_MEASURED": "Retirement-linked status not recorded",
    "NOT_EVALUABLE_WORK_EXIT_NO_RETIREMENT_EVENTS": "No events after a retirement-linked exit",
    "NOT_APPLICABLE_NO_COMPARABLE_INTERVAL": "No interval in the comparable window",
}

SUFFIX = (
    ("_GROUP_N", "Too few respondents in one group"),
    ("_EVENTS", "Too few events in one group"),
    ("_N", "Too few respondents in one group"),
)


def label(status: object) -> str:
    """A reader-facing reason for one status value."""
    s = str(status or "").strip()
    if not s:
        return "--"
    # The exact mapping is consulted first, and deliberately. Reading anything
    # that starts with PASS as a clean pass is what let PASS_DESCRIPTIVE_K2,
    # a status named to mean "two-cohort descriptive pooling, below the
    # three-cohort minimum", print as an unqualified "Estimated".
    if s in EXACT:
        return EXACT[s]
    if s.upper().startswith("PASS") or s.lower() == "available":
        return "Estimated"
    up = s.upper()
    if up.startswith("NOT_EVALUABLE") or up == "NOT_EVALUABLE":
        for suffix, text in SUFFIX:
            if up.endswith(suffix):
                return text
        return "Did not meet the support rules"
    if up.startswith("NOT_APPLICABLE"):
        return "Not applicable"
    return s.replace("_", " ").capitalize()
