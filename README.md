# AI Investigation Intelligence Engine

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Tests](https://img.shields.io/badge/tests-73%20passed-green.svg)](#testing-and-validation)
[![License](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)

The AI Investigation Intelligence Engine is an autonomous dataset-investigation system for analyst workflows. Instead of stopping at descriptive profiling, it detects structural and statistical issues, compresses redundant findings, constructs semantic knowledge objects, frames evidence-backed hypotheses, and produces a prioritized investigation queue with provenance, metrics, and analyst-readable summaries.

The engine is designed for deterministic, explainable investigative reasoning:

- no LLM dependency
- typed Pydantic v2 models
- structured logging
- reproducible graph-based reasoning
- analyst-facing CLI output
- machine-readable export paths for downstream systems

## What the engine does

Given a dataset, the engine:

1. loads and profiles the source data
2. runs registered investigator modules
3. emits atomic `Finding` objects
4. compresses related findings into `EvidenceUnit` objects
5. constructs semantic `KnowledgeObject` concepts
6. generates evidence-backed `Hypothesis` objects
7. prioritizes final `Investigation` queue entries
8. renders explainability, reasoning metrics, benchmarks, and publication assets

This yields a review flow that is much closer to how an analyst actually works: fewer duplicate alerts, clearer concepts, claim-style hypotheses, explicit evidence strength, likely causes with provenance, and actionable validation guidance.

## Finalized reasoning pipeline

The architecture is intentionally layered and stable:

```mermaid
flowchart TD
    A[Dataset] --> B[Investigators]
    B --> C[Findings]
    C --> D[Graph-Based Evidence Compression]
    D --> E[Evidence Units]
    E --> F[Knowledge Objects]
    F --> G[Hypotheses]
    G --> H[Investigations]
    H --> I[Explainability + Metrics + Benchmarking + Reporting]
```

## Core architecture

### 1. Dataset ingestion

Supported input paths:

- CSV
- XLSX
- PostgreSQL connection strings
- in-memory `pandas.DataFrame` objects via the Python API

`DatasetLoader` converts the source into a DataFrame and derives dataset metadata such as row count, column count, memory footprint, and per-column profiling signals used by downstream investigators.

### 2. Autonomous investigators

The engine discovers registered modules at runtime and executes all compatible investigators unless filtered by configuration or CLI options.

Current built-in investigators:

#### Integrity Investigator

Focuses on structural health and data-quality failures:

- missing values
- duplicate rows
- duplicate features
- identifier integrity
- constant features
- near-constant features
- datatype integrity issues
- abnormal cardinality
- correlated missingness patterns
- overall structural health scoring

#### Relationship Investigator

Focuses on feature-to-feature and feature-to-target relationships:

- Pearson correlation
- Spearman correlation
- mutual information
- target dependency analysis
- multicollinearity via VIF
- feature communities and redundant groups

### 3. Multi-layer reasoning

The reasoning system turns many noisy findings into a smaller, analyst-usable queue.

#### Findings

`Finding` is the atomic anomaly object emitted directly by an investigator. It preserves:

- the source module
- severity
- confidence
- affected columns
- evidence payload
- metadata category

#### Graph-Based Evidence Compression

The engine builds a finding graph and links findings using structural, statistical, and semantic similarity. Community detection then compresses related findings into `EvidenceUnit` objects.

This reduces review load and preserves traceability:

- each evidence unit retains supporting finding IDs
- graph cohesion and community strength remain available
- compression is deterministic under the configured seed

#### Knowledge construction

`KnowledgeObject` instances convert compressed evidence into semantic concepts an analyst can interpret, such as:

- Neighbor Cell Availability
- Neighbor Cell Telemetry
- Radio Configuration
- Radio Signal Quality
- Identifier Integrity
- Feature Engineering Artifact

The constructor merges duplicate semantic concepts deterministically so the output does not fragment into placeholder or repeated knowledge objects.

#### Hypothesis generation

`Hypothesis` objects frame claim-style analytical statements supported by evidence and knowledge. They include:

- supporting evidence
- contradicting evidence
- unknown evidence
- plausible causes
- validation steps
- confidence breakdown

The hypothesis layer is designed to produce analyst-readable claims rather than generic prompts. For example:

- “Neighbor-cell measurements are systematically unavailable.”
- “Neighbor-cell telemetry is conditionally collected.”
- “Radio-configuration parameters show little operational variation.”

#### Investigation prioritization

`Investigation` is the final analyst-facing queue entry. Each investigation includes:

- title
- summary
- hypothesis
- priority
- confidence
- likely causes
- recommended next steps
- evidence strength assessment
- full provenance back to findings

Priority is computed from evidence severity and reasoning support rather than raw finding count alone, which helps prevent noisy categories from outranking smaller but more critical issues.

## Explainability and analyst experience

The current engine surfaces much more than a flat list of findings.

### Executive summary

The CLI now renders a top-level executive summary when investigations exist, including:

- number of investigations generated
- high- and medium-priority counts
- primary issue
- estimated cause
- affected feature count
- lead confidence
- recommended first action

### Investigation dashboards

Verbose and explain modes render analyst-facing investigation panels with:

- summary
- evidence strength
- confidence
- supporting evidence
- knowledge objects
- hypothesis titles
- likely causes
- impact
- recommended validation
- provenance chain
- confidence breakdown
- compact reasoning tree

### Structured evidence strength

Evidence strength is presented as derived support rather than a standalone label. The dashboard exposes:

- supporting findings
- supporting evidence units
- contributing investigators
- supporting knowledge objects
- graph cohesion
- community strength
- overall strength rating

### Provenance and lineage

Every final investigation can be traced back through:

- supporting hypotheses
- supporting knowledge objects
- supporting evidence units
- original findings
- contributing investigators

The `--explain` workflow renders full reasoning lineage trees for detailed auditability.

### Reasoning metrics

The engine records explainability and compression metrics such as:

- counts per reasoning layer
- compression ratios
- graph density
- average community size
- reasoning depth
- average evidence per hypothesis
- average knowledge per hypothesis
- runtime per layer
- memory metrics

## Public models

Important top-level models exposed by the engine:

| Model | Purpose |
| --- | --- |
| `Finding` | Atomic investigator output |
| `EvidenceUnit` | Compressed analytical fact built from related findings |
| `KnowledgeObject` | Semantic concept derived from evidence |
| `Hypothesis` | Evidence-backed analytical claim |
| `Investigation` | Prioritized analyst-facing investigation entry |
| `InvestigationResult` | Full run result including findings, reasoning layers, metrics, and provenance |

## Installation

The project requires Python 3.12 or later.

```bash path=null start=null
git clone https://github.com/username/investigation-engine.git
cd investigation-engine
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Configuration

The engine uses hierarchical Pydantic Settings with:

1. programmatic overrides
2. environment variables
3. `.env`
4. code defaults

Environment variables use the `IE_` prefix and nested fields use `__`.

Example:

```ini
IE_LOG_LEVEL=INFO
IE_ENGINE__MAX_ROWS_SAMPLE=500000
IE_ENGINE__MAX_FINDINGS_PER_MODULE=50
IE_ENGINE__INVESTIGATION_SEMANTIC_COHESION_THRESHOLD=0.46
IE_ENGINE__MAX_INVESTIGATION_EVIDENCE_SHARE=0.60

IE_INTEGRITY__MISSING_MEDIUM_THRESHOLD=0.10
IE_INTEGRITY__NEAR_CONSTANT_DOMINANCE_THRESHOLD=0.95

IE_RELATIONSHIP__PEARSON_THRESHOLD=0.80
IE_RELATIONSHIP__VIF_HIGH_THRESHOLD=5.0

IE_EVIDENCE_COMPRESSION__ALGORITHM=louvain
IE_EVIDENCE_COMPRESSION__EDGE_WEIGHT_THRESHOLD=0.18
IE_EVIDENCE_COMPRESSION__RANDOM_SEED=42
```

Notable configurable areas:

- engine execution behavior
- investigator thresholds
- evidence-compression algorithm and similarity weights
- semantic cohesion and evidence-share guardrails
- logging

## CLI usage

The Typer CLI is the main entry point.

### General help

```bash path=null start=null
investigation-engine --help
```

### Inspect engine metadata

```bash path=null start=null
investigation-engine info
investigation-engine schema
```

### Run an investigation

High-level summary:

```bash path=null start=null
investigation-engine investigate data/dataset.csv
```

Verbose analyst dashboard:

```bash path=null start=null
investigation-engine investigate data/dataset.csv --verbose
```

Full explainability and metrics:

```bash path=null start=null
investigation-engine investigate data/dataset.csv --explain --metrics
```

Benchmark reasoning methods during the run:

```bash path=null start=null
investigation-engine investigate data/dataset.csv --benchmark
```

Run only selected modules:

```bash path=null start=null
investigation-engine investigate data/dataset.csv --modules integrity_investigator,relationship_investigator
```

Investigate PostgreSQL data:

```bash path=null start=null
investigation-engine investigate "postgresql://user:password@host:5432/dbname" --table events
```

Or:

```bash path=null start=null
investigation-engine investigate "postgresql://user:password@host:5432/dbname" --query "SELECT * FROM events LIMIT 10000"
```

Write the full run result to disk:

```bash path=null start=null
investigation-engine investigate data/dataset.csv --output result.json
```

### Standalone research/reporting commands

Compression benchmark:

```bash path=null start=null
investigation-engine benchmark data/dataset.csv --output benchmark.md
```

Ablation study:

```bash path=null start=null
investigation-engine ablate data/dataset.csv --output ablation.csv
```

Dataset evaluation suite:

```bash path=null start=null
investigation-engine evaluate --output evaluation.md
```

Reasoning graph export:

```bash path=null start=null
investigation-engine visualize data/dataset.csv --graph-type evidence --output reasoning_graph.mmd
```

Publication-ready tables and assets:

```bash path=null start=null
investigation-engine paper-assets data/dataset.csv --output-dir paper_assets
```

## Output formats

Depending on the command and flags, the engine can emit:

- terminal dashboards via Rich
- full JSON result payloads
- Markdown benchmark/evaluation tables
- CSV summaries
- LaTeX tables
- Mermaid graphs
- GraphML exports
- SVG graph exports
- publication-asset bundles

## Python API

Programmatic usage is supported through `InvestigationEngine`.

```python path=null start=null
import pandas as pd

from investigation_engine.config.settings import Settings
from investigation_engine.core.engine import InvestigationEngine

df = pd.read_csv("data/dataset.csv")

engine = InvestigationEngine(Settings())
result = engine.investigate_dataframe(df, name="dataset")

print(result.total_findings)
print(len(result.evidence_units))
print(len(result.knowledge_objects))
print(len(result.hypotheses))
print(len(result.investigations))
```

The returned `InvestigationResult` includes:

- dataset metadata
- all module findings
- evidence units
- knowledge objects
- hypotheses
- prioritized investigations
- module execution records
- reasoning metrics
- provenance trees
- benchmark reports when requested

## Determinism and validation philosophy

The engine emphasizes deterministic execution:

- graph compression uses deterministic settings and seed control
- reasoning artifacts use stable IDs and stable timestamps where needed
- repeated runs with identical inputs are expected to produce stable serialized outputs

Validation includes:

- investigator behavior
- reasoning-layer construction
- explainability rendering
- deterministic serialization
- benchmarking and ablation workflows
- CLI behavior
- visualization and paper-asset generation

## Testing and validation

Run the full test suite:

```bash path=null start=null
pytest
```

Current validation status:

- `73 passed`

Targeted reasoning-focused validation examples:

```bash path=null start=null
python -m pytest tests/test_explainable_reasoning.py tests/test_evidence_fusion.py tests/test_phase3_research_platform.py
```

Optional local quality checks:

```bash path=null start=null
ruff check .
mypy .
```

## Repository structure

High-level layout:

- `investigation_engine/core/` — engine orchestration, loading, plugin discovery
- `investigation_engine/modules/` — investigator modules
- `investigation_engine/models/` — top-level public result models
- `investigation_engine/fusion/` — reasoning artifact orchestration
- `investigation_engine/reasoning/evidence/` — graph compression and evidence models
- `investigation_engine/reasoning/knowledge/` — semantic concept construction
- `investigation_engine/reasoning/hypothesis/` — hypothesis generation
- `investigation_engine/reasoning/prioritization/` — investigation refinement and ranking
- `investigation_engine/reasoning/provenance/` — lineage rendering and tracking
- `investigation_engine/reports/` — exporters, visualizations, paper assets
- `tests/` — regression and integration coverage

## Design constraints

The current system is built around these constraints:

- no architecture redesign of the reasoning stages
- no LLM dependency
- no breaking public API changes
- deterministic execution
- strong analyst readability
- explainability and provenance first

## License

MIT. See `LICENSE`.
