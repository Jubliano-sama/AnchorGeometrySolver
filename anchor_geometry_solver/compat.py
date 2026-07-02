from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
SMARTCLICKER = WORK / "SmartClicker-GUI"
OUTPUTS = ROOT / "outputs"


def ensure_legacy_paths() -> None:
    """Expose the bundled experiment scripts and SmartClicker package."""

    for path in (WORK, SMARTCLICKER):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


ensure_legacy_paths()

from uwb_capture.anchor_geometry import AnchorPairDistance  # noqa: E402


__all__ = ["AnchorPairDistance", "OUTPUTS", "ROOT", "SMARTCLICKER", "WORK", "ensure_legacy_paths"]
