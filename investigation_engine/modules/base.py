"""Base investigation module interface.

All investigation modules MUST inherit from BaseInvestigationModule and
implement the required abstract methods. This ensures a uniform contract
across the entire plugin ecosystem.

A module answers one question: "Should an analyst investigate this?"

Design Contracts:
    1. A module MUST be stateless — no mutable instance state between runs.
    2. A module MUST return a list of Finding objects (may be empty).
    3. A module MUST implement can_run() to declare its prerequisites.
    4. A module MUST NOT raise exceptions to the engine — handle internally
       and return empty findings if processing fails.
    5. A module MUST NOT produce side effects (no file I/O, no network calls).
    6. A module SHOULD respect config thresholds from Settings.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import pandas as pd
from loguru import logger

from investigation_engine.config.settings import Settings
from investigation_engine.models.dataset import DatasetInfo
from investigation_engine.models.finding import Finding


class BaseInvestigationModule(ABC):
    """Abstract base class for all investigation modules.

    Subclasses must define class-level attributes and implement
    the abstract methods. The engine calls modules in this order:

        1. can_run(df, dataset_info) → bool
        2. investigate(df, dataset_info, config) → list[Finding]

    Attributes:
        name: Unique identifier for this module (used in config, CLI, output).
        description: Human-readable description of what this module investigates.
        version: Semantic version of the module implementation.
        tags: Classification tags for grouping (e.g., "quality", "statistical").
    """

    name: ClassVar[str]
    description: ClassVar[str]
    version: ClassVar[str] = "0.1.0"
    tags: ClassVar[list[str]] = []

    @abstractmethod
    def investigate(
        self,
        df: pd.DataFrame,
        dataset_info: DatasetInfo,
        config: Settings,
    ) -> list[Finding]:
        """Execute the investigation on the provided dataset.

        This is the core method where the module performs its analysis
        and produces findings. Must be deterministic for the same input.

        Args:
            df: The dataset to investigate.
            dataset_info: Pre-computed metadata about the dataset.
            config: Engine configuration with thresholds and settings.

        Returns:
            List of findings. May be empty if nothing noteworthy is found.
            Must never return None.
        """
        ...

    @abstractmethod
    def can_run(
        self,
        df: pd.DataFrame,
        dataset_info: DatasetInfo,
    ) -> bool:
        """Determine whether this module can run on the given dataset.

        Modules should check for necessary preconditions:
            - Minimum row count
            - Required column types (numeric, categorical, datetime)
            - Minimum column count
            - Dataset size constraints

        This method must be fast and must not perform heavy computation.

        Args:
            df: The dataset to check.
            dataset_info: Pre-computed metadata about the dataset.

        Returns:
            True if the module can meaningfully investigate this dataset.
        """
        ...

    def get_skip_reason(
        self,
        df: pd.DataFrame,
        dataset_info: DatasetInfo,
    ) -> str | None:
        """Provide a human-readable reason if the module cannot run.

        Override this method to give specific skip reasons that will
        be recorded in the InvestigationResult.

        Args:
            df: The dataset to check.
            dataset_info: Pre-computed metadata about the dataset.

        Returns:
            Skip reason string, or None if the module can run.
        """
        if not self.can_run(df, dataset_info):
            return f"Module '{self.name}' prerequisites not met for dataset '{dataset_info.name}'."
        return None

    def validate_findings(self, findings: list[Finding]) -> list[Finding]:
        """Validate and sanitize findings before returning to the engine.

        Default implementation filters out findings with invalid data.
        Modules can override for custom validation.

        Args:
            findings: Raw findings from the investigate method.

        Returns:
            Validated findings.
        """
        validated: list[Finding] = []
        for finding in findings:
            try:
                # Re-validate through Pydantic
                validated.append(Finding.model_validate(finding.model_dump()))
            except Exception as e:
                logger.warning(
                    "Module '{}' produced invalid finding '{}': {}",
                    self.name,
                    finding.title,
                    e,
                )
        return validated

    def get_module_info(self) -> dict[str, Any]:
        """Return metadata about this module for registration and discovery.

        Returns:
            Dictionary with module metadata.
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "tags": self.tags,
            "class": self.__class__.__qualname__,
            "module_path": self.__class__.__module__,
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', version='{self.version}')>"

    def __str__(self) -> str:
        return f"{self.name} v{self.version}: {self.description}"
