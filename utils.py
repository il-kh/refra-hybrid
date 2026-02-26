from typing import Optional


def parse_float(value: str) -> Optional[float]:
    """Return float or None for blank / non-numeric CSV cells."""
    v = value.strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None