"""Core package — engine, plugin system, and data loading."""

from investigation_engine.core.engine import InvestigationEngine
from investigation_engine.core.plugin import register_module, get_registered_modules

__all__ = [
    "InvestigationEngine",
    "register_module",
    "get_registered_modules",
]
