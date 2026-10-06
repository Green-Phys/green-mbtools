from packaging.version import InvalidVersion, Version


# Version for Green-MBTools Package
__version__ = "1.0.0"


def require_input_version(input_version, minimum_version="1.0.0"):
    """Raise ValueError if the input version is missing, invalid, or too old.

    Versions equal to minimum_version are accepted.
    """
    minimum = Version(minimum_version)
    try:
        current = Version(input_version)
    except (InvalidVersion, TypeError) as exc:
        raise ValueError(
            f"Missing or invalid input version: {input_version!r}. "
            f"green-mbtools {minimum} or newer is required. "
            "Regenerate input.h5 and rerun the weak-coupling calculation."
        ) from exc

    if current < minimum:
        raise ValueError(
            f"Input version {current} is unsupported; "
            f"green-mbtools {minimum} or newer is required. "
            "Regenerate input.h5 and rerun the weak-coupling calculation."
        )
