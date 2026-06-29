"""Plugin registry and discovery system.

Provides a decorator-based registration mechanism for investigation modules.
New modules are added to the system by:

    1. Subclassing BaseInvestigationModule
    2. Decorating the class with @register_module
    3. Placing the file in the investigation_engine/modules/ directory

No existing code needs to be modified. The engine auto-discovers modules
at startup via importlib scanning of the modules package.

Architecture Notes:
    - The registry is a module-level singleton (dict)
    - Registration happens at import time via the decorator
    - Discovery uses importlib to find and import all module files
    - Duplicate module names raise a clear error at registration time
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import TypeVar

from loguru import logger

from investigation_engine.modules.base import BaseInvestigationModule

T = TypeVar("T", bound=type[BaseInvestigationModule])

# Module-level plugin registry
_REGISTRY: dict[str, type[BaseInvestigationModule]] = {}


class ModuleRegistrationError(Exception):
    """Raised when a module cannot be registered."""


def register_module(cls: T) -> T:
    """Class decorator to register an investigation module.

    Validates that the class has the required attributes and registers
    it in the global plugin registry.

    Args:
        cls: The investigation module class to register.

    Returns:
        The same class, unmodified.

    Raises:
        ModuleRegistrationError: If the module name is already registered
            or required attributes are missing.

    Example:
        @register_module
        class OutlierDetectionModule(BaseInvestigationModule):
            name = "outlier_detection"
            description = "Detects statistical outliers in numeric columns."
            ...
    """
    # Validate required class attributes
    if not hasattr(cls, "name") or not cls.name:
        raise ModuleRegistrationError(
            f"Module class '{cls.__qualname__}' must define a non-empty 'name' class attribute."
        )

    if not hasattr(cls, "description") or not cls.description:
        raise ModuleRegistrationError(
            f"Module class '{cls.__qualname__}' must define a non-empty 'description' class attribute."
        )

    # Check for duplicate registration
    if cls.name in _REGISTRY:
        existing = _REGISTRY[cls.name]
        raise ModuleRegistrationError(
            f"Module name '{cls.name}' is already registered by "
            f"'{existing.__qualname__}'. Cannot register '{cls.__qualname__}'."
        )

    # Validate it's a proper subclass
    if not issubclass(cls, BaseInvestigationModule):
        raise ModuleRegistrationError(
            f"'{cls.__qualname__}' must be a subclass of BaseInvestigationModule."
        )

    _REGISTRY[cls.name] = cls
    logger.debug("Registered investigation module: {} ({})", cls.name, cls.__qualname__)

    return cls


def get_registered_modules() -> dict[str, type[BaseInvestigationModule]]:
    """Return a copy of the current module registry.

    Returns:
        Dictionary mapping module names to their classes.
    """
    return dict(_REGISTRY)


def get_module(name: str) -> type[BaseInvestigationModule] | None:
    """Retrieve a specific registered module by name.

    Args:
        name: The module name to look up.

    Returns:
        The module class if registered, None otherwise.
    """
    return _REGISTRY.get(name)


def list_module_names() -> list[str]:
    """Return sorted list of all registered module names.

    Returns:
        Sorted list of module name strings.
    """
    return sorted(_REGISTRY.keys())


def clear_registry() -> None:
    """Clear all registered modules. Used primarily in testing.

    Warning:
        This removes ALL registered modules. Should only be used
        in test fixtures and never in production code.
    """
    _REGISTRY.clear()
    logger.debug("Module registry cleared.")


def discover_modules(package_name: str = "investigation_engine.modules") -> int:
    """Auto-discover and import all modules in the given package.

    Scans the specified package directory for Python modules and imports
    them, triggering their @register_module decorators.

    Args:
        package_name: Dotted package path to scan for modules.

    Returns:
        Number of newly discovered module files (not all may contain modules).

    Raises:
        ImportError: If the package cannot be imported.
    """
    try:
        package = importlib.import_module(package_name)
    except ImportError as e:
        logger.error("Failed to import module package '{}': {}", package_name, e)
        raise

    if not hasattr(package, "__path__"):
        logger.warning("Package '{}' has no __path__, skipping discovery.", package_name)
        return 0

    discovered = 0
    for _importer, module_name, is_pkg in pkgutil.walk_packages(
        package.__path__,
        prefix=f"{package_name}.",
    ):
        # Skip __init__ and base module
        if module_name.endswith("__init__") or module_name.endswith(".base"):
            continue

        try:
            importlib.import_module(module_name)
            discovered += 1
            logger.debug("Discovered module file: {}", module_name)
        except Exception as e:
            logger.warning("Failed to import module '{}': {}", module_name, e)

    logger.info(
        "Module discovery complete: {} files scanned, {} modules registered.",
        discovered,
        len(_REGISTRY),
    )
    return discovered
