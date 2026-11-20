"""
util.duration
=============
Best-effort duration parsing for time-horizon vs decision-window checks.

Accepts ISO 8601 durations ("PT15M", "P1DT2H", "P2W") and simple English
phrases ("90 seconds", "24 hours", "2 days", "15 min"). Returns seconds, or
None when the text cannot be interpreted as a concrete duration (for example
"no constraint" or "as soon as possible"). Calendar units are approximated
(year = 365 d, month = 30 d) — adequate for an ordering check, not a calendar.
"""

from __future__ import annotations

import re

_ISO = re.compile(
    r"^P(?:(?P<Y>\d+)Y)?(?:(?P<Mo>\d+)M)?(?:(?P<W>\d+)W)?(?:(?P<D>\d+)D)?"
    r"(?:T(?:(?P<H>\d+)H)?(?:(?P<Mi>\d+)M)?(?:(?P<S>\d+)S)?)?$"
)
_ISO_SECONDS = {
    "Y": 365 * 86400, "Mo": 30 * 86400, "W": 7 * 86400, "D": 86400,
    "H": 3600, "Mi": 60, "S": 1,
}
_UNIT = {
    "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
    "d": 86400, "day": 86400, "days": 86400,
    "w": 604800, "week": 604800, "weeks": 604800,
}
_PHRASE = re.compile(r"(\d+(?:\.\d+)?)\s*([a-z]+)")


def parse_duration_seconds(text: str | None) -> float | None:
    """Return the duration in seconds, or None if it is not a concrete one."""
    if not text:
        return None
    s = text.strip()
    m = _ISO.match(s.upper())
    if m and any(m.groupdict().values()):
        return float(sum(int(v) * _ISO_SECONDS[k] for k, v in m.groupdict().items() if v))
    total, found = 0.0, False
    for num, unit in _PHRASE.findall(s.lower()):
        if unit in _UNIT:
            total += float(num) * _UNIT[unit]
            found = True
    return total if found else None
