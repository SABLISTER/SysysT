"""Path resolution utilities."""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_claims_path(config, prefer_filtered: bool = True) -> Path:
    """Return claims_filtered.json if it exists, else claims.json.

    Parameters
    ----------
    config : Config
        Pipeline configuration (must have ``claims_dir`` attribute).
    prefer_filtered : bool
        When *True* (default), prefer the filtered variant produced by
        Stage 5.  Set to *False* to always return the unfiltered file.

    Returns
    -------
    Path
        Absolute path to the best available claims JSON file.
    """
    filtered = config.claims_dir / "claims_filtered.json"
    if prefer_filtered and filtered.exists():
        logger.debug("Using filtered claims: %s", filtered)
        return filtered
    base = config.claims_dir / "claims.json"
    logger.debug("Using base claims: %s", base)
    return base
