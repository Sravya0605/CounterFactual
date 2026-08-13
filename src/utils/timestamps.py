"""Shared CAPE timestamp normalization.

CAPE report timestamps are strings like '2025-09-10 19:13:50,303'. This
is the single source of truth for parsing them into comparable floats --
do not reimplement this in graph_builder.py, feasibility.py, or
graph_features.py. If a new CAPE timestamp format shows up, fix it here
once.
"""
from typing import Any

_TIMESTAMP_FORMATS = (
    "%Y-%m-%d %H:%M:%S,%f",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
)


def normalize_timestamp(value: Any) -> float:
    """Parse a single timestamp value into a comparable float.

    Raises ValueError on an unparseable string -- callers that need a
    non-raising fallback (e.g. sort keys) should catch it explicitly and
    decide their own fallback behavior, rather than this function
    silently picking one.
    """
    from datetime import datetime

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        stripped = value.strip()
        for fmt in _TIMESTAMP_FORMATS:
            try:
                return datetime.strptime(stripped, fmt).timestamp()
            except ValueError:
                pass

    raise ValueError(f"Unsupported timestamp format: {value!r}")