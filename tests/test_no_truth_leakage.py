"""
Guards for the central claim of the project: the AI and the Gatekeeper must not see the truth.
These scan the source of src/ai_engine and src/gatekeeper (they pass trivially while those packages are empty
and will start failing the moment someone reaches for ground truth).
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = [
    r"\btrue_\w+",                 # any truth column
    r"true_params",                # the real plant parameters
    r"\bTRUTH_PREFIX\b",
    r"reactor_core",               # the plant itself
    r"\bReactorCore\b",
    r"make_plant_pair",            # would let a consumer regenerate the true plant
    r"mismatch_offsets",
    r"""\[["']label["']\]|\.label\b""",   # ground-truth class (training / evaluation code only)
]
LABEL_PATTERN = FORBIDDEN[-1]
FEATURE_SIDE_ONLY = {"features.py"}


def _sources(pkg):
    return [p for p in (ROOT / "src" / pkg).rglob("*.py")]


@pytest.mark.parametrize("pkg", ["ai_engine", "gatekeeper"])
def test_no_truth_access_in_source(pkg):
    for path in _sources(pkg):
        text = path.read_text()
        label_allowed = pkg == "ai_engine" and path.name not in FEATURE_SIDE_ONLY   # training needs y; features/gatekeeper must not
        for pat in FORBIDDEN:
            if pat == LABEL_PATTERN and label_allowed:
                continue
            assert not re.search(pat, text), f"{path.relative_to(ROOT)} touches ground truth: /{pat}/"
