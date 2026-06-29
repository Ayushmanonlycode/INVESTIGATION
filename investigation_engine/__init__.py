"""
Investigation Engine — AI Investigation Intelligence Engine.

Autonomous dataset investigation with structured evidence output.
Every module answers one question: "Should an analyst investigate this?"
"""

__version__ = "0.1.0"
__author__ = "Investigation Engine Team"

from investigation_engine.models.finding import Finding
from investigation_engine.models.severity import Severity
from investigation_engine.models.dataset import DatasetInfo
from investigation_engine.models.investigation import InvestigationResult
from investigation_engine.models.investigation_hypothesis import Investigation
from investigation_engine.core.engine import InvestigationEngine

__all__ = [
    "Finding",
    "Severity",
    "DatasetInfo",
    "InvestigationResult",
    "Investigation",
    "InvestigationEngine",
    "__version__",
]
