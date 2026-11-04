"""Statement classification lives in the runtime skill so lab/ and the sandbox share one implementation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "submission/skills/mini-palace/scripts"))

from triage import StatementProfile, clean_statement, profile_statement  # noqa: E402,F401
