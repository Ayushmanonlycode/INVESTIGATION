# AI Investigation Intelligence Engine

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](#project-status)
[![Tests](https://img.shields.io/badge/tests-78%20passed-green.svg)](#testing)
[![License](https://img.shields.io/badge/license-MIT-purple.svg)](#license)

The **AI Investigation Intelligence Engine** is a deterministic research prototype for
investigating structured datasets. Instead of stopping at descriptive profiling, it converts
structural and statistical observations into a smaller, prioritized, explainable investigation
queue.

The engine implements a **Hierarchical Evidentiary Reasoning Framework (HERF)**:

```text
Dataset
  -> Investigator modules
  -> Atomic findings
  -> Finding similarity graph
  -> Evidence communities and evidence units
  -> Semantic knowledge objects
  -> Evidence-backed diagnostic hypotheses
  -> Prioritized investigations
  -> Explanations, metrics, graphs, benchmarks, and publication assets
```

The current release does not require an LLM. Terms such as `hypothesis`, `likely cause`, and
`confidence` describe deterministic diagnostic reasoning. They are not causal proof or
calibrated probabilities.

## Contents

- [Research motivation](#research-motivation)
- [Project status](#project-status)
- [Quick start](#quick-start)
- [Architecture](#architecture)
- [Dataset ingestion](#dataset-ingestion)
- [Investigator modules](#investigator-modules)
- [Reasoning pipeline](#reasoning-pipeline)
- [Scoring](#scoring)
- [Performance profiles](#performance-profiles)
- [CLI reference](#cli-reference)
- [Python API](#python-api)
- [Configuration](#configuration)
- [Extending the engine](#extending-the-engine)
- [Outputs and research workflows](#outputs-and-research-workflows)
- [Determinism and failures](#determinism-and-failures)
- [Testing](#testing)
- [Known limitations](#known-limitations)
- [Repository structure](#repository-structure)
- [Development history and merging](#development-history-and-merging)
- [Phase 4 roadmap](#phase-4-roadmap)

## Research motivation

Traditional EDA and profiling tools primarily answer:

> What statistics and quality indicators exist in this dataset?

They can leave an analyst with many independent alerts, correlation pairs, missing-value tables,
and charts that still require interpretation. This engine instead asks:

> Which observations appear related, what diagnostic claim do they support, why does it matter,
> and what should be checked first?

The framework is organized around four ideas:

1. **Finding normalization**: every investigator emits the same typed `Finding` contract.
2. **Evidence compression**: related findings are grouped through a deterministic similarity graph.
3. **Progressive abstraction**: evidence becomes knowledge, hypotheses, and work items without
   discarding provenance.
4. **Explainable prioritization**: confidence and priority are decomposed into explicit factors.

The intended research contribution is the end-to-end evidence-fusion workflow. The individual
statistics, community algorithms, and rules are established techniques.

## Project status

Package version: **`0.1.0`**

Python: **3.12 or later**

Maturity: **alpha research prototype**

| Capability | Status | Detail |
| --- | --- | --- |
| CSV, TSV, Excel, PostgreSQL loading | Implemented | In-memory pandas execution |
| DataFrame API | Implemented | Bypasses source loading |
| Runtime plugin discovery | Implemented | Decorator registration |
| Integrity Investigator | Implemented | Structural and missingness checks |
| Relationship Investigator | Implemented | Correlation, MI, VIF, feature groups |
| Graph-Based Evidence Compression | Implemented | Louvain plus two alternatives |
| Knowledge and hypothesis layers | Implemented | Deterministic semantic rules |
| Investigation prioritization | Implemented | Explicit factor breakdowns |
| Provenance and explanations | Implemented | Full upstream IDs |
| Benchmarks, ablations, evaluation | Implemented | Internal research workflows |
| Balanced/full profiles | Implemented in current working tree | Deterministic scan budgets |
| Web UI | Not implemented | CLI and Python API only |
| Temporal/streaming analysis | Planned | Phase 4 |
| Human-feedback learning | Planned | Phase 4 |
| Distributed execution | Planned | Phase 4 |
| Optional LLM narrative | Planned | Must remain downstream of evidence |

### Current scope

The engine detects structural integrity and statistical relationship issues, compresses redundant
findings, and proposes analyst validation work. It does **not** train a fraud model, label
transactions as fraudulent, repair the source data, infer causality, or continuously monitor a
stream.

## Quick start

### Install

```bash
git clone git@github.com:Ayushmanonlycode/INVESTIGATION.git
cd INVESTIGATION

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### Inspect the engine

```bash
investigation-engine info
investigation-engine schema
```

`info` discovers modules and prints their metadata. `schema` prints JSON schemas for the public
Pydantic models.

### Investigate a dataset

```bash
investigation-engine investigate data/dataset.csv
investigation-engine investigate data/dataset.csv --verbose
investigation-engine investigate data/dataset.csv --explain --metrics
investigation-engine investigate data/dataset.csv --benchmark
```

Export the complete result:

```bash
investigation-engine investigate data/dataset.csv --output result.json
```

## Architecture

```mermaid
flowchart TD
    subgraph Ingestion
        Source[CSV / TSV / Excel / PostgreSQL / DataFrame]
        Loader[DatasetLoader]
        Frame[pandas DataFrame]
        Profile[DatasetInfo + ColumnProfile]
        Source --> Loader
        Loader --> Frame
        Loader --> Profile
    end

    subgraph Investigation
        Registry[Plugin registry]
        Integrity[Integrity Investigator]
        Relationship[Relationship Investigator]
        Finding[Finding objects]
        Registry --> Integrity
        Registry --> Relationship
        Frame --> Integrity
        Frame --> Relationship
        Profile --> Integrity
        Profile --> Relationship
        Integrity --> Finding
        Relationship --> Finding
    end

    subgraph Reasoning
        Graph[Finding similarity graph]
        Community[Evidence communities]
        Evidence[EvidenceUnit objects]
        Knowledge[KnowledgeObject objects]
        Hypothesis[Hypothesis objects]
        Queue[Investigation queue]
        Finding --> Graph --> Community --> Evidence --> Knowledge --> Hypothesis --> Queue
    end

    subgraph Output
        Rich[Rich CLI]
        Reports[JSON / CSV / Markdown / LaTeX]
        Trees[Provenance trees]
        Graphs[Mermaid / GraphML / SVG]
        Research[Metrics / benchmarks / ablations]
        Queue --> Rich
        Queue --> Reports
        Queue --> Trees
        Queue --> Graphs
        Queue --> Research
    end
```

### Execution lifecycle

For a file or database source, `InvestigationEngine.investigate()`:

1. records UTC and wall-clock start times;
2. loads the source and builds reusable dataset metadata;
3. discovers modules under `investigation_engine.modules`;
4. applies enabled, disabled, and CLI module filters;
5. calls each module's `can_run()`;
6. executes compatible modules and records duration, count, status, or error;
7. caps each module at `max_findings_per_module`;
8. builds evidence, knowledge, hypotheses, and investigations;
9. calculates reasoning metrics and provenance trees;
10. sorts findings by severity and confidence;
11. returns one `InvestigationResult`.

`investigate_dataframe()` follows the same module and reasoning lifecycle but skips source
loading.

## Dataset ingestion

| Source | Detection | Reader |
| --- | --- | --- |
| CSV | `.csv` | `pandas.read_csv`, PyArrow engine |
| TSV | `.tsv` | CSV loader; specify the separator through the API when required |
| Excel | `.xlsx`, `.xls` | `pandas.read_excel`, OpenPyXL |
| PostgreSQL | PostgreSQL URI prefixes | `pandas.read_sql` |
| DataFrame | Python API | Existing in-memory frame |

PostgreSQL requires a table:

```bash
investigation-engine investigate +  "postgresql://user:password@host:5432/database" +  --table events
```

or a query:

```bash
investigation-engine investigate +  "postgresql://user:password@host:5432/database" +  --query "SELECT * FROM events LIMIT 10000"
```

Do not commit database credentials to configuration, shell scripts, logs, or reports.

### Metadata

The loader records dataset name, source type, dimensions, deep memory usage, duplicate-row count,
missing-value presence, dtype mapping, and a profile for each column. A `ColumnProfile` contains
null and unique counts, numeric/categorical/datetime/boolean flags, and up to five non-null sample
values. Investigators reuse these values instead of recomputing exact structural counts.

### Global sampling

`IE_ENGINE__MAX_ROWS_SAMPLE` defaults to `1_000_000`. Larger loaded datasets are sampled
deterministically with seed `42`.

- Global sampling changes the DataFrame seen by every module, so counts describe that sample.
- Balanced profile budgets only bound selected expensive scans; exact metadata still describes
  the loaded DataFrame.
- `--profile full` does not override the global loader limit.

## Investigator modules

Investigators are stateless plugins that receive a DataFrame, `DatasetInfo`, and `Settings`,
then return validated `Finding` objects.

### Integrity Investigator

Name: `integrity_investigator`

| Domain | Behavior |
| --- | --- |
| Missing values | Empty and partially missing columns with severity thresholds |
| Duplicate rows | Duplicate count, ratio, and representative rows |
| Duplicate features | Hash candidates followed by exact equality verification |
| Identifier integrity | Name/uniqueness heuristics, missing IDs, duplicate IDs |
| Constant features | At most one unique non-null value |
| Near-constant features | Dominant-value ratio above threshold |
| Datatype integrity | Numeric-looking strings and mixed object types |
| Cardinality | Suspicious high/low cardinality |
| Missingness relationships | Correlation graph over missing indicators |
| Structural health | Weighted dataset-level integrity summary |

The performance-hardened implementation reuses loader metadata for nulls, uniqueness, duplicates,
constants, identifiers, and cardinality. Missingness relationships are returned as connected column
groups rather than thousands of independent pairs.

### Relationship Investigator

Name: `relationship_investigator`

The module requires at least two non-constant numeric columns.

| Domain | Behavior |
| --- | --- |
| Pearson | Strong linear relationships |
| Spearman | Strong monotonic relationships |
| Significance | Pairwise tests and p-values for selected candidates |
| Mutual information | Target-aware or pairwise nonlinear dependency |
| Target dependency | Common names such as target, label, class, outcome, and y |
| VIF | Multicollinearity estimation |
| Redundancy groups | Components of very highly correlated features |
| Feature communities | Connected groups at configurable strengths |

In balanced mode, every threshold-crossing Pearson and Spearman edge is retained for group
discovery, while only the strongest bounded candidates receive separate significance tests and
findings.

### Finding cap

`IE_ENGINE__MAX_FINDINGS_PER_MODULE` defaults to `50`. A warning is logged when output is
truncated. This controls graph and report size but can exclude lower-ranked observations. Research
experiments should report the cap and test sensitivity to larger values.

## Reasoning pipeline

### Typed layers

| Layer | Model | Responsibility | Provenance |
| --- | --- | --- | --- |
| Observation | `Finding` | Atomic investigator result | Module and affected data |
| Evidence | `EvidenceUnit` | Compressed analytical fact | Finding IDs |
| Knowledge | `KnowledgeObject` | Semantic concept | Evidence IDs |
| Hypothesis | `Hypothesis` | Diagnostic claim and validation plan | Evidence and knowledge IDs |
| Decision | `Investigation` | Prioritized analyst work item | All upstream IDs |

### Findings

A finding contains a deterministic ID, module, title, description, structured evidence, severity,
confidence, recommendation, affected columns/rows, and scoring metadata. Severity values and
weights are:

| Severity | Weight | Actionable |
| --- | ---: | --- |
| Critical | `1.0` | Yes |
| High | `0.8` | Yes |
| Medium | `0.5` | Yes |
| Low | `0.2` | No |
| Info | `0.0` | No |

### Finding graph

Each finding is a node. Candidate pairs are indexed by shared columns, feature families, semantic
terms, and finding categories. Oversized index buckets are bounded by
`max_index_bucket_size`.

Similarity combines:

| Component | Default weight | Internal signals |
| --- | ---: | --- |
| Structural | `0.40` | Columns, rows, metadata, investigator, feature family |
| Statistical | `0.35` | Severity, confidence, comparable numeric evidence |
| Semantic | `0.25` | Prefix, token, and domain-term overlap |

An edge is created when the combined score reaches `edge_weight_threshold`, default `0.18`.
Edges retain component scores and merge explanations.

### Community detection and evidence

Supported algorithms:

- `louvain` (default);
- `greedy_modularity`;
- `connected_components`.

Louvain and greedy modularity expose resolution settings. Randomized algorithms use the configured
seed, default `42`.

Each detected community becomes an `EvidenceUnit` containing category, title, summary, supporting
findings, community identity/strength, component similarities, confidence, strength, affected
columns, merge explanations, source categories, maximum severity, and contributing investigators.

Compression ratio alone is not evidence of quality. Issue coverage, group purity, and analyst
utility must be evaluated independently.

### Knowledge objects

The constructor maps evidence to semantic concepts and deterministically merges duplicate concepts.
Knowledge objects retain supporting evidence, confidence, abstraction level, related concepts,
affected columns, semantic terms, categories, and provenance.

The current vocabulary includes generic quality concepts and radio/neighbor-cell concepts from the
original research cases. It is not yet a domain-general ontology.

### Hypotheses

Hypotheses contain a diagnostic statement, support, contradictions, unresolved evidence, knowledge
objects, confidence breakdown, plausible causes, validation steps, workstream metadata, and
provenance. They are diagnostic propositions, not automatically verified causal claims.

### Investigations

Hypotheses are linked using:

| Signal | Weight |
| --- | ---: |
| Shared columns | `0.20` |
| Shared concepts | `0.25` |
| Shared evidence | `0.15` |
| Shared feature families | `0.15` |
| Confidence similarity | `0.10` |
| Shared workstream | `0.15` |

Edges require `0.28`. Connected groups are refined using semantic cohesion and maximum
evidence-share guardrails before becoming investigations.

An investigation includes title, summary, diagnostic statement, priority, confidence, evidence
strength, likely causes, validation steps, affected columns, contradictions, gaps, explanations,
and complete upstream provenance.

## Scoring

### Confidence

```text
confidence = clamp(
    0.38 * normalized mean evidence strength
  + 0.27 * mean community strength
  + 0.22 * cross-investigator agreement
  - 0.08 * contradiction factor
  - 0.05 * unknown-evidence factor
)
```

Agreement reaches `1.0` at two investigators; the unknown factor reaches `1.0` at five distinct
gaps. The result is clamped to `[0, 1]`. This is a heuristic score, not a calibrated probability.

### Priority

```text
priority = clamp(
    0.30 * evidence_strength
  + 20.0 * confidence
  + 15.0 * semantic_coverage
  + 15.0 * maximum_severity_weight
  + 10.0 * cross-investigator_agreement
  + 10.0 * hypothesis_agreement
  + 10.0 * semantic_cohesion
  - 20.0 * contradiction_penalty
)
```

`evidence_strength` is `0-100`; other factors are `0-1`. Priority is clamped to `0-100`.
The factor explanation is serialized with the investigation.

Evidence strength describes empirical support; confidence describes consistency of reasoning
signals. Neither proves causality.

## Performance profiles

### Balanced

`balanced` is the CLI and configuration default.

| Operation | Default budget | Preserved behavior |
| --- | ---: | --- |
| Pearson | `50,000` rows | All strong edges retained for groups |
| Spearman | `25,000` rows | All strong edges retained for groups |
| Candidate significance tests | `200` pairs/method | Strongest individual findings |
| VIF | `50,000` rows | Same thresholds |
| Missingness graph | `50,000` rows | Exact null counts/ratios |
| Pairwise MI | `5,000` rows, `500` pairs | Deterministic column shortlist |
| Target MI | `5,000` rows/feature | Skips extra pairwise MI scan |

```bash
investigation-engine investigate data.csv --profile balanced
```

### Full

`full` removes balanced Pearson, Spearman, VIF, missingness, candidate-test, and MI pair budgets.
It scans all eligible correlation candidates and uses symmetric pairwise MI.

```bash
investigation-engine investigate data.csv --profile full
```

Qualifications:

- `mi_max_rows` and `mi_max_columns` remain explicit MI limits;
- global `max_rows_sample` still applies;
- the module finding cap still applies;
- full mode can be much slower and use substantially more memory.

Use balanced mode for iteration and large datasets. Use full mode for validation and final research
measurements. Compare modes with finding overlap, rank correlation, evidence-community stability,
ground-truth coverage, runtime, and peak memory.

## CLI reference

```text
investigation-engine
├── investigate
├── benchmark
├── ablate
├── evaluate
├── visualize
├── paper-assets
├── info
└── schema
```

### Investigate

```bash
investigation-engine investigate SOURCE [OPTIONS]
```

| Option | Short | Purpose |
| --- | --- | --- |
| `--modules TEXT` | `-m` | Comma-separated module names |
| `--table TEXT` | `-t` | PostgreSQL table |
| `--query TEXT` | `-q` | PostgreSQL SQL |
| `--output PATH` | `-o` | Export path |
| `--log-level TEXT` | `-l` | Logging level |
| `--profile balanced\|full` | | Coverage/performance profile |
| `--verbose` | `-v` | Findings and investigation dashboards |
| `--explain` | | Complete provenance trees |
| `--metrics` | | Reasoning/compression metrics |
| `--benchmark` | | Raw, rule, and graph comparison |

```bash
investigation-engine investigate data.csv +  --modules integrity_investigator,relationship_investigator +  --profile full +  --verbose --explain --metrics +  --output result.json
```

### Research and reporting commands

```bash
investigation-engine benchmark data.csv --output benchmark.md
investigation-engine ablate data.csv --output ablation.csv
investigation-engine evaluate --output evaluation.md
investigation-engine visualize data.csv --graph-type evidence --output graph.mmd
investigation-engine paper-assets data.csv --output-dir paper_assets
```

`visualize` accepts `evidence`, `knowledge`, `hypothesis`, or `investigation`. Evaluation
uses built-in scikit-learn datasets and optional local datasets when available.

Output behavior depends on the selected flags and suffix. Complete results default to JSON.
Metric, benchmark, evaluation, and ablation tables support JSON, CSV, Markdown, or LaTeX where the
command provides the corresponding exporter. Explain output supports JSON or Markdown lineage.

## Python API

```python
import pandas as pd

from investigation_engine.config.settings import Settings
from investigation_engine.core.engine import InvestigationEngine

df = pd.read_csv("data/dataset.csv")

settings = Settings()
settings.engine.analysis_profile = "balanced"

engine = InvestigationEngine(settings)
result = engine.investigate_dataframe(df, name="dataset")

print(result.dataset_info.row_count)
print(result.get_severity_summary())
print(result.total_findings)
print(len(result.evidence_units))
print(len(result.knowledge_objects))
print(len(result.hypotheses))
print(len(result.investigations))
```

Source API:

```python
engine = InvestigationEngine(Settings())
result = engine.investigate(
    "data/dataset.xlsx",
    modules=["integrity_investigator", "relationship_investigator"],
)

result_json = result.model_dump_json(indent=2)
```

Query helpers:

```python
from investigation_engine.models.severity import Severity

critical = result.findings_by_severity(Severity.CRITICAL)
integrity = result.findings_by_module("integrity_investigator")
amount_findings = result.findings_for_column("amount")
```

### Result fields

| Field | Description |
| --- | --- |
| `dataset_info` | Source and dataset metadata |
| `findings` | Severity-sorted atomic output |
| `evidence_units` | Graph-compressed findings |
| `knowledge_objects` | Semantic evidence concepts |
| `hypotheses` | Diagnostic claims |
| `investigations` | Priority-sorted work queue |
| `module_records` | Status, count, duration, error |
| `modules_executed/failed/skipped` | Execution summary |
| `reasoning_metrics` | Counts, graph, compression, timing, memory |
| `provenance_trees` | Rendered lineage |
| `benchmark_reports` | Optional comparisons |

## Configuration

Priority: programmatic overrides, environment variables, `.env`, then defaults. Variables use
`IE_`, are case-insensitive, and use `__` between nested groups.

### Engine

| Variable | Default |
| --- | ---: |
| `IE_ENGINE__ANALYSIS_PROFILE` | `balanced` |
| `IE_ENGINE__MAX_ROWS_SAMPLE` | `1000000` |
| `IE_ENGINE__MAX_FINDINGS_PER_MODULE` | `50` |
| `IE_ENGINE__PARALLEL_MODULES` | `false` (experimental) |
| `IE_ENGINE__FAIL_FAST` | `false` |
| `IE_ENGINE__ENABLED_MODULES` | unset |
| `IE_ENGINE__DISABLED_MODULES` | empty |
| `IE_ENGINE__INVESTIGATION_SEMANTIC_COHESION_THRESHOLD` | `0.46` |
| `IE_ENGINE__MAX_INVESTIGATION_EVIDENCE_SHARE` | `0.60` |

### Integrity

| Suffix under `IE_INTEGRITY__` | Default |
| --- | ---: |
| `MISSING_CRITICAL/HIGH/MEDIUM/LOW_THRESHOLD` | `0.50/0.30/0.10/0.05` |
| `MISSING_EMPTY_COLUMN_THRESHOLD` | `0.99` |
| `DUPLICATE_CRITICAL/HIGH/MEDIUM/LOW_THRESHOLD` | `0.20/0.10/0.05/0.01` |
| `CONSTANT_DOMINANCE_THRESHOLD` | `0.9999` |
| `NEAR_CONSTANT_DOMINANCE_THRESHOLD` | `0.95` |
| `HIGH_CARDINALITY_RATIO` | `0.90` |
| `LOW_CARDINALITY_MAX_UNIQUE` | `2` |
| `ID_UNIQUENESS_THRESHOLD` | `0.95` |
| `NUMERIC_STRING_SAMPLE_SIZE` | `1000` |
| `NUMERIC_STRING_THRESHOLD` | `0.80` |
| `MIXED_TYPE_THRESHOLD` | `0.01` |
| `MISSINGNESS_CORRELATION_THRESHOLD` | `0.70` |
| `MISSINGNESS_MIN_MISSING_RATIO` | `0.01` |
| `MISSINGNESS_MAX_ROWS` | `50000` |

Structural health weights are missing `0.25`, duplicates `0.15`, constants `0.10`, types
`0.20`, identifiers `0.15`, and cardinality `0.15`.

### Relationship

| Suffix under `IE_RELATIONSHIP__` | Default |
| --- | ---: |
| `PEARSON_THRESHOLD` / `SPEARMAN_THRESHOLD` | `0.80` / `0.80` |
| `REDUNDANCY_THRESHOLD` | `0.95` |
| `STRONG_GROUP_THRESHOLD` | `0.90` |
| `COMMUNITY_THRESHOLD` | `0.75` |
| `MUTUAL_INFORMATION_THRESHOLD` | `0.20` |
| `VIF_HIGH_THRESHOLD` / `VIF_VERY_HIGH_THRESHOLD` | `5.0` / `10.0` |
| `MIN_PAIR_OBSERVATIONS` | `30` |
| `PEARSON_MAX_ROWS` / `SPEARMAN_MAX_ROWS` | `50000` / `25000` |
| `VIF_MAX_ROWS` | `50000` |
| `MAX_CORRELATION_CANDIDATES_PER_METHOD` | `200` |
| `MI_MAX_ROWS` / `MI_MAX_COLUMNS` / `MI_MAX_PAIRS` | `5000` / `150` / `500` |
| `MI_N_NEIGHBORS` | `3` |

### Evidence compression

| Suffix under `IE_EVIDENCE_COMPRESSION__` | Default |
| --- | ---: |
| `ALGORITHM` | `louvain` |
| `EDGE_WEIGHT_THRESHOLD` | `0.18` |
| `STRUCTURAL/STATISTICAL/SEMANTIC_WEIGHT` | `0.40/0.35/0.25` |
| `MAX_INDEX_BUCKET_SIZE` | `250` |
| `LOUVAIN_RESOLUTION` / `GREEDY_RESOLUTION` | `1.0` / `1.0` |
| `RANDOM_SEED` | `42` |
| `BENCHMARK_REPEAT_RUNS` | `3` |

Shared-column, feature-family, semantic-term, and category candidate indexes plus all internal
similarity weights are independently configurable in `EvidenceCompressionSettings`.

Example:

```ini
IE_ENGINE__ANALYSIS_PROFILE=balanced
IE_ENGINE__MAX_FINDINGS_PER_MODULE=100
IE_RELATIONSHIP__PEARSON_MAX_ROWS=100000
IE_RELATIONSHIP__MI_MAX_PAIRS=1000
IE_EVIDENCE_COMPRESSION__RANDOM_SEED=42
IE_LOGGING__LOG_LEVEL=INFO
```

## Extending the engine

Create a module under `investigation_engine/modules/`:

```python
from typing import ClassVar

import pandas as pd

from investigation_engine.config.settings import Settings
from investigation_engine.core.plugin import register_module
from investigation_engine.models.dataset import DatasetInfo
from investigation_engine.models.finding import Finding
from investigation_engine.modules.base import BaseInvestigationModule


@register_module
class DistributionInvestigator(BaseInvestigationModule):
    name = "distribution_investigator"
    description = "Detects distributional patterns requiring review."
    version = "0.1.0"
    tags: ClassVar[list[str]] = ["distribution", "statistics"]

    def can_run(self, df: pd.DataFrame, info: DatasetInfo) -> bool:
        return info.row_count >= 30 and bool(info.numeric_columns)

    def investigate(
        self,
        df: pd.DataFrame,
        info: DatasetInfo,
        config: Settings,
    ) -> list[Finding]:
        findings: list[Finding] = []
        # Add deterministic analysis and typed findings.
        return self.validate_findings(findings)
```

Contracts:

1. Modules are stateless and deterministic.
2. `can_run()` is fast and declares prerequisites.
3. `investigate()` returns a list, including an empty list.
4. Modules do not perform network calls or file writes.
5. Thresholds and budgets live in typed settings.
6. Evidence payloads contain reproducible metrics.
7. Findings pass Pydantic revalidation.

Discovery imports non-base files in `investigation_engine.modules`. Duplicate names raise
`ModuleRegistrationError`.

## Outputs and research workflows

### Terminal and files

Rich output includes dataset/module summaries, severity counts, executive summary, reasoning flow,
investigation dashboards, causes, validation, confidence factors, and provenance. Export formats
include JSON, CSV, Markdown, LaTeX, Mermaid, GraphML, and SVG.

### Metrics

The metrics engine reports counts per layer, compression factors, duplicate reduction, graph
nodes/edges/density/modularity/communities, community sizes, reasoning depth, evidence per
hypothesis, knowledge per hypothesis, confidence, layer timing, total runtime, and dataset memory.
These describe system behavior, not diagnostic correctness.

### Provenance

```text
Investigation
  -> Hypothesis IDs
  -> KnowledgeObject IDs
  -> EvidenceUnit IDs
  -> Finding IDs
  -> Investigator modules
  -> Statistical evidence
```

### Comparisons and ablations

The method comparator contrasts raw findings, legacy rule compression, and Graph-Based Evidence
Compression. Ablations include baseline, no graph compression, no knowledge layer, no hypothesis
layer, no semantic splitting, and no confidence propagation.

### Evaluation limits

The bundled evaluator is a smoke/systems benchmark using Iris, Wine, Breast Cancer, and optional
local datasets. It does not provide ground-truth issue labels. A publication-quality evaluation
should add controlled corruptions, finding precision/recall, group purity or ARI/NMI, issue
coverage, top-k ranking, confidence calibration, analyst time, cross-domain generalization,
profile stability, runtime, and peak memory.

For reproducibility, record the commit, complete configuration, dataset checksum, machine,
profile, module cap, and seed for every experiment.

## Determinism and failures

Deterministic mechanisms include seed-42 statistical samples, configured graph seeds,
content-derived stable IDs, stable artifact ordering/timestamps, and deterministic queue sorting.
Identical serialization still requires identical data, configuration, dependencies, modules, and
compatible numeric behavior.

Failure behavior:

- failed `can_run()` checks create skipped records;
- module exceptions create failed records;
- `fail_fast=false` continues remaining modules;
- integrity subdomains are isolated;
- invalid findings are rejected during validation;
- loader exceptions become `DataLoadError` with source context.

Modules currently execute sequentially. `parallel_modules` is experimental and is not distributed
execution.

## Testing

```bash
pytest
```

Expected current result:

```text
78 passed
```

Focused tests:

```bash
python -m pytest tests/test_integrity_investigator.py
python -m pytest tests/test_relationship_investigator.py
python -m pytest tests/test_graph_based_evidence_compression.py
python -m pytest tests/test_evidence_fusion.py
python -m pytest tests/test_explainable_reasoning.py
python -m pytest tests/test_phase3_research_platform.py
python -m pytest tests/test_performance_profiles.py
```

Static checks:

```bash
ruff check .
mypy investigation_engine
```

These are development targets, not currently clean merge gates. At the time of this README
consolidation, the full test suite passes, while `ruff check .` reports pre-existing style issues
across the repository.

## Known limitations

1. This is an alpha research prototype, not a production observability service.
2. Processing is pandas-based and in memory; peak RAM may greatly exceed source file size.
3. Only two investigators are registered.
4. Finding caps can exclude observations before fusion.
5. Balanced sampling can miss rare or weak relationships.
6. Confidence and priority weights are heuristic and uncalibrated.
7. Semantic rules are partly domain-specific.
8. Similar graph nodes do not establish a shared causal origin.
9. Current evaluation emphasizes internal metrics rather than labelled correctness.
10. No temporal state, streaming, feedback learning, distributed execution, or interactive UI is
    implemented.
11. PostgreSQL table interpolation expects trusted input.
12. Results guide review; they are not automatic remediation or final domain conclusions.

## Repository structure

```text
INVESTIGATION/
├── investigation_engine/
│   ├── cli/                         # Typer commands and Rich output
│   ├── config/                      # Pydantic settings
│   ├── core/                        # Loader, registry, orchestration
│   ├── fusion/                      # Reasoning orchestration
│   ├── models/                      # Dataset, finding, result models
│   ├── modules/                     # Built-in investigators
│   ├── ranking/                     # Legacy priority support
│   ├── reasoning/
│   │   ├── evidence/                # Similarity, graphs, communities, compression
│   │   ├── knowledge/               # Semantic construction
│   │   ├── hypothesis/              # Hypothesis generation
│   │   ├── prioritization/          # Refinement and ranking
│   │   ├── provenance/              # Lineage
│   │   ├── ablation.py
│   │   ├── comparison.py
│   │   ├── confidence.py
│   │   └── metrics.py
│   ├── reports/                     # Exports, graphs, paper assets
│   ├── evaluation.py
│   └── utils/
├── tests/
├── benchmark.py
├── evaluation.py
├── pyproject.toml
└── README.md
```

## Development history and merging

`PHASE-1` and `PHASE-2` are sibling branches that diverged from common commit `8804f99`;
neither branch contains the other branch's tip.

### Phase 1

`PHASE-1` established the package, CLI, configuration, logging, loader, plugin framework, two
investigators, typed findings, initial hierarchical reasoning, tests, and provenance/determinism
requirements.

Its README listed distribution, anomaly, cluster, importance, drift, feedback, GraphRAG, and a
dashboard as future work. Branch names are historical; those items were not all implemented in
the next branch.

### Phase 2

`PHASE-2` added graph similarity and compression, three community methods, expanded evidence and
knowledge, hypothesis generation, confidence propagation, semantic refinement, likely-cause and
evidence-strength explanations, metrics, provenance rendering, comparisons, ablations, dataset
evaluation, visualizations, paper assets, richer CLI output, and deterministic regression tests.

The current working tree adds balanced/full profiles, bounded relationship scans, grouped
missingness findings, loader-metadata reuse, and profile tests.

### Merge checklist

```bash
# First inspect and commit the current PHASE-2 implementation and README together.
git status
git diff --check
pytest
git add README.md investigation_engine tests
git commit -m "docs: consolidate phase documentation and performance profiles"

# Create an integration branch from the more complete PHASE-2 working state.
git switch -c integration/phase-merge
git merge --no-ff PHASE-1
pytest
```

Resolve the expected README conflict in favor of this consolidated document. Review every code
conflict rather than accepting one branch wholesale, because the branches are siblings. Check that
datasets, credentials, generated benchmarks, and paper assets were not accidentally staged. After
verification, rename or merge the integration branch into the intended default branch.

## Phase 4 roadmap

The following are proposed and **not implemented**.

### Cross-investigation reasoning

- Link related investigations without duplicating evidence.
- Construct higher-level narratives while retaining complete lineage.
- Track shared causes, contradictions, and validation dependencies.

### Temporal and streaming investigations

- Incrementally update profiles and evidence as batches arrive.
- Track creation, escalation, resolution, and recurrence.
- Define bounded state, event-time, and late-data behavior.

### Root-cause graph inference

- Separate candidate causes from observations.
- Rank diagnostic causes without presenting correlation as causality.
- Evaluate with known or injected root causes and validation interventions.

### Domain knowledge packs

- Externalize terminology, semantic rules, constraints, and actions.
- Version packs and provide a domain-neutral fallback.
- Evaluate zero-shot and configured cross-domain behavior.

### Human feedback

- Capture accept, reject, merge, split, reprioritize, and resolve actions.
- Preserve the original deterministic result alongside adaptive state.
- Make learned ranking/similarity changes auditable.

### Optional LLM narrative

- Consume only finalized structured investigations.
- Never alter evidence, severity, confidence, or priority.
- Cite artifact IDs and evaluate consistency, omissions, hallucinations, cost, and reproducibility.

### Distributed execution

- Partition large scans and graph construction.
- Aggregate deterministically across workers.
- Measure communication, memory, latency, recovery, and single-node equivalence.

Phase 4 should be delivered as focused, testable milestones. Patent-sensitive technical designs
should be reviewed before being added to a public repository, paper, presentation, or issue.

## License

The package metadata currently declares the MIT License. A standalone `LICENSE` file is not
present in the repository and should be added before public distribution.
