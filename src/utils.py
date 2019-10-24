from pathlib import Path
from typing import Dict


def project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent


def prob_distribution_from_dict(dct: Dict):
    total = sum(dct.values())
    dct = {k: v / total for k, v in dct.items()}
    return dct
