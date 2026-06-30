"""Investigation Engine orchestrator.

The central orchestrator that coordinates the entire investigation workflow:
    1. Load the dataset
    2. Discover registered investigation modules
    3. Filter modules based on configuration and dataset compatibility
    4. Execute each module's investigation
    5. Collect and aggregate findings
    6. Return a structured InvestigationResult

Design Principles:
    - Fail-safe: a crashing module never kills the investigation
    - Deterministic: same input produces same findings (given same config)
    - Observable: all execution is logged with timing metadata
    - Extensible: modules are discovered at runtime via the plugin registry
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from loguru import logger

from investigation_engine.config.settings import Settings
from investigation_engine.core.loader import DatasetLoader
from investigation_engine.core.plugin import discover_modules, get_registered_modules
from investigation_engine.fusion.engine import EvidenceFusionEngine
from investigation_engine.models.dataset import DatasetInfo
from investigation_engine.models.finding import Finding
from investigation_engine.models.investigation import InvestigationResult, ModuleExecutionRecord
from investigation_engine.modules.base import BaseInvestigationModule
from investigation_engine.reasoning.metrics import ReasoningMetricsEngine
from investigation_engine.reasoning.provenance.renderer import ProvenanceTreeRenderer
from investigation_engine.utils.logging import configure_logging


class InvestigationEngine:
    """AI Investigation Intelligence Engine.

    Orchestrates autonomous dataset investigation by coordinating
    registered investigation modules and aggregating their findings.

    Usage:
        engine = InvestigationEngine()
        result = engine.investigate("data.csv")

        for finding in result.findings:
            print(finding.title, finding.severity)

    Attributes:
        config: Engine configuration.
    """

    def __init__(self, config: Settings | None = None) -> None:
        """Initialize the Investigation Engine.

        Args:
            config: Engine configuration. Uses defaults if None.
        """
        self._config = config or Settings()
        self._loader = DatasetLoader(self._config)
        self._fusion_engine = EvidenceFusionEngine(self._config)
        self._metrics_engine = ReasoningMetricsEngine(self._config.evidence_compression)
        self._provenance_renderer = ProvenanceTreeRenderer()
        self._modules_discovered = False

        # Configure logging
        configure_logging(self._config.logging)

        logger.info("Investigation Engine initialized (v0.1.0)")

    @property
    def config(self) -> Settings:
        """Current engine configuration."""
        return self._config

    def _ensure_modules_discovered(self) -> None:
        """Discover modules if not already done."""
        if not self._modules_discovered:
            discover_modules()
            self._modules_discovered = True

    def _get_active_modules(
        self,
        requested_modules: list[str] | None = None,
    ) -> list[BaseInvestigationModule]:
        """Get the list of module instances to execute.

        Applies filtering based on:
            1. Explicit request (if provided)
            2. Enabled/disabled lists from configuration
            3. Module availability in the registry

        Args:
            requested_modules: Optional explicit list of module names to run.

        Returns:
            List of instantiated module objects.
        """
        self._ensure_modules_discovered()
        registry = get_registered_modules()

        if not registry:
            logger.warning("No investigation modules registered.")
            return []

        # Determine which modules to use
        if requested_modules is not None:
            # Use explicitly requested modules
            module_names = requested_modules
        elif self._config.engine.enabled_modules is not None:
            # Use config-defined enabled list
            module_names = self._config.engine.enabled_modules
        else:
            # Use all registered modules
            module_names = list(registry.keys())

        # Apply disabled list
        disabled = set(self._config.engine.disabled_modules)
        module_names = [name for name in module_names if name not in disabled]

        # Instantiate modules
        active_modules: list[BaseInvestigationModule] = []
        for name in module_names:
            if name not in registry:
                logger.warning("Requested module '{}' is not registered. Skipping.", name)
                continue
            try:
                module_instance = registry[name]()
                active_modules.append(module_instance)
            except Exception as e:
                logger.error("Failed to instantiate module '{}': {}", name, e)

        logger.info(
            "Active modules: {} of {} registered",
            len(active_modules),
            len(registry),
        )
        return active_modules

    def _execute_module(
        self,
        module: BaseInvestigationModule,
        df: pd.DataFrame,
        dataset_info: DatasetInfo,
    ) -> tuple[list[Finding], ModuleExecutionRecord]:
        """Execute a single investigation module with full error handling.

        Args:
            module: The module instance to execute.
            df: The dataset to investigate.
            dataset_info: Pre-computed dataset metadata.

        Returns:
            Tuple of (findings, execution_record).
        """
        start_time = time.perf_counter()

        # Check if module can run
        try:
            can_run = module.can_run(df, dataset_info)
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error("Module '{}' can_run() raised an exception: {}", module.name, e)
            return [], ModuleExecutionRecord(
                module_name=module.name,
                status="failed",
                duration_seconds=elapsed,
                error_message=f"can_run() failed: {e}",
            )

        if not can_run:
            elapsed = time.perf_counter() - start_time
            skip_reason = module.get_skip_reason(df, dataset_info)
            logger.info("Module '{}' skipped: {}", module.name, skip_reason)
            return [], ModuleExecutionRecord(
                module_name=module.name,
                status="skipped",
                duration_seconds=elapsed,
                skipped_reason=skip_reason,
            )

        # Execute investigation
        try:
            logger.info("Executing module: {}", module.name)
            findings = module.investigate(df, dataset_info, self._config)

            # Validate findings
            findings = module.validate_findings(findings)

            # Enforce max findings per module
            max_findings = self._config.engine.max_findings_per_module
            if len(findings) > max_findings:
                logger.warning(
                    "Module '{}' produced {} findings, truncating to {}.",
                    module.name,
                    len(findings),
                    max_findings,
                )
                findings = findings[:max_findings]

            elapsed = time.perf_counter() - start_time
            logger.info(
                "Module '{}' completed: {} findings in {:.3f}s",
                module.name,
                len(findings),
                elapsed,
            )

            return findings, ModuleExecutionRecord(
                module_name=module.name,
                status="success",
                finding_count=len(findings),
                duration_seconds=elapsed,
            )

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(
                "Module '{}' failed after {:.3f}s: {}",
                module.name,
                elapsed,
                e,
                exc_info=True,
            )
            return [], ModuleExecutionRecord(
                module_name=module.name,
                status="failed",
                duration_seconds=elapsed,
                error_message=str(e),
            )

    def investigate(
        self,
        source: str,
        *,
        modules: list[str] | None = None,
        table_name: str | None = None,
        query: str | None = None,
        **loader_kwargs: Any,
    ) -> InvestigationResult:
        """Execute a complete investigation on a dataset.

        This is the primary entry point for the Investigation Engine.
        It loads the dataset, runs all applicable modules, and returns
        a structured result.

        Args:
            source: File path or connection string for the dataset.
            modules: Optional list of specific module names to run.
            table_name: Table name for PostgreSQL sources.
            query: Custom SQL query for PostgreSQL sources.
            **loader_kwargs: Additional arguments for the data loader.

        Returns:
            InvestigationResult containing all findings and execution metadata.
        """
        investigation_start = datetime.now(timezone.utc)
        wall_start = time.perf_counter()

        logger.info("=" * 72)
        logger.info("INVESTIGATION STARTED")
        logger.info("Source: {}", source)
        logger.info("=" * 72)

        # Load dataset
        df, dataset_info = self._loader.load(
            source,
            table_name=table_name,
            query=query,
            **loader_kwargs,
        )

        # Get active modules
        active_modules = self._get_active_modules(modules)

        # Execute modules
        all_findings: list[Finding] = []
        all_records: list[ModuleExecutionRecord] = []
        executed: list[str] = []
        failed: list[str] = []
        skipped: list[str] = []

        for module in active_modules:
            findings, record = self._execute_module(module, df, dataset_info)
            all_findings.extend(findings)
            all_records.append(record)

            if record.status == "success":
                executed.append(module.name)
            elif record.status == "failed":
                failed.append(module.name)
                if self._config.engine.fail_fast:
                    logger.error("Fail-fast enabled. Stopping investigation.")
                    break
            elif record.status == "skipped":
                skipped.append(module.name)

        # Run post-investigation reasoning pipeline
        reasoning = self._fusion_engine.build_reasoning_artifacts(all_findings)

        # Build result
        wall_elapsed = time.perf_counter() - wall_start
        investigation_end = datetime.now(timezone.utc)

        result = InvestigationResult(
            dataset_info=dataset_info,
            findings=all_findings,
            evidence_units=reasoning.evidence_units,
            knowledge_objects=reasoning.knowledge_objects,
            hypotheses=reasoning.hypotheses,
            investigations=reasoning.investigation_queue.investigations,
            module_records=all_records,
            modules_executed=executed,
            modules_failed=failed,
            modules_skipped=skipped,
            started_at=investigation_start,
            completed_at=investigation_end,
            duration_seconds=wall_elapsed,
            metadata={
                "engine_version": "0.1.0",
                "config_snapshot": self._config.model_dump(mode="json"),
                "reasoning_summary": {
                    **reasoning.investigation_queue.metadata,
                    "layer_timings": reasoning.layer_timings,
                },
            },
        )
        result.reasoning_metrics = self._metrics_engine.build(
            all_findings,
            result.evidence_units,
            result.knowledge_objects,
            result.hypotheses,
            result.investigations,
            graph=reasoning.evidence_graph,
            communities=reasoning.evidence_communities,
            layer_timings=reasoning.layer_timings,
            total_runtime_seconds=wall_elapsed,
            dataset_memory_bytes=dataset_info.memory_usage_bytes,
        )
        result.provenance_trees = self._provenance_renderer.build(result)

        # Sort findings by severity
        result.sort_findings_by_severity()

        logger.info("=" * 72)
        logger.info("INVESTIGATION COMPLETE")
        logger.info(
            "Duration: {:.3f}s | Modules: {} executed, {} failed, {} skipped",
            wall_elapsed,
            len(executed),
            len(failed),
            len(skipped),
        )
        logger.info(
            "Findings: {} total ({} actionable)",
            result.total_findings,
            result.actionable_findings,
        )
        logger.info("=" * 72)

        return result

    def investigate_dataframe(
        self,
        df: pd.DataFrame,
        *,
        name: str = "in_memory_dataset",
        modules: list[str] | None = None,
    ) -> InvestigationResult:
        """Investigate a pre-loaded DataFrame.

        Convenience method for programmatic usage where the data is
        already loaded into memory.

        Args:
            df: The DataFrame to investigate.
            name: Human-readable name for the dataset.
            modules: Optional list of specific module names to run.

        Returns:
            InvestigationResult containing all findings.
        """
        investigation_start = datetime.now(timezone.utc)
        wall_start = time.perf_counter()

        logger.info("Investigating in-memory DataFrame: '{}'", name)

        # Build dataset info directly
        dataset_info = self._loader._build_dataset_info(df, name, "dataframe")

        # Get and execute modules
        active_modules = self._get_active_modules(modules)

        all_findings: list[Finding] = []
        all_records: list[ModuleExecutionRecord] = []
        executed: list[str] = []
        failed: list[str] = []
        skipped: list[str] = []

        for module in active_modules:
            findings, record = self._execute_module(module, df, dataset_info)
            all_findings.extend(findings)
            all_records.append(record)

            if record.status == "success":
                executed.append(module.name)
            elif record.status == "failed":
                failed.append(module.name)
            elif record.status == "skipped":
                skipped.append(module.name)

        wall_elapsed = time.perf_counter() - wall_start

        # Run post-investigation reasoning pipeline
        reasoning = self._fusion_engine.build_reasoning_artifacts(all_findings)

        result = InvestigationResult(
            dataset_info=dataset_info,
            findings=all_findings,
            evidence_units=reasoning.evidence_units,
            knowledge_objects=reasoning.knowledge_objects,
            hypotheses=reasoning.hypotheses,
            investigations=reasoning.investigation_queue.investigations,
            module_records=all_records,
            modules_executed=executed,
            modules_failed=failed,
            modules_skipped=skipped,
            started_at=investigation_start,
            completed_at=datetime.now(timezone.utc),
            duration_seconds=wall_elapsed,
            metadata={
                "engine_version": "0.1.0",
                "reasoning_summary": {
                    **reasoning.investigation_queue.metadata,
                    "layer_timings": reasoning.layer_timings,
                },
            },
        )

        result.reasoning_metrics = self._metrics_engine.build(
            all_findings,
            result.evidence_units,
            result.knowledge_objects,
            result.hypotheses,
            result.investigations,
            graph=reasoning.evidence_graph,
            communities=reasoning.evidence_communities,
            layer_timings=reasoning.layer_timings,
            total_runtime_seconds=wall_elapsed,
            dataset_memory_bytes=dataset_info.memory_usage_bytes,
        )
        result.provenance_trees = self._provenance_renderer.build(result)
        result.sort_findings_by_severity()
        return result
