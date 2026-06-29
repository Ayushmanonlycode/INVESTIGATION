# AI Investigation Intelligence Engine

An autonomous investigation intelligence engine designed to automate the exploratory phase that analysts perform before decision-making. Rather than describing the entire dataset or generating large descriptive reports, the engine proactively discovers statistically important anomalies and fuses related findings into cohesive high-level investigations.

---

## Architecture

```
investigation_engine/
├── core/         # Engine orchestrator, plugin discovery, data loader
├── models/       # Pydantic schemas (Finding, DatasetInfo, Investigation, InvestigationResult)
├── modules/      # Investigation plugins (autonomous investigators)
├── fusion/       # Evidence Fusion Engine (graph-based clustering & rule reasoning)
├── ranking/      # Investigation Priority Score (IPS) interfaces
├── reports/      # Reserved for future report generation
├── utils/        # Logging, timing, and shared statistical helpers
├── config/       # Pydantic Settings configuration (hierarchical)
└── cli/          # Typer CLI
```

---

## Core Components

### 1. Integrity Investigator (`modules/integrity.py`)
Our first autonomous investigator behaves like an experienced data quality engineer. It evaluates the dataset's structural integrity across 10 distinct domains:
- **Missing Value Investigation**: Identifies column-level dropout rates and completely empty columns.
- **Duplicate Row Investigation**: Vectorized exact duplicate row detection.
- **Duplicate Feature Investigation**: Compares fast hash fingerprints to find identical/synonymous columns.
- **Identifier Investigation Heuristics**: Validates key columns for duplicates or missing values.
- **Constant Feature Investigation**: Identifies variables with zero variance.
- **Near-Constant Feature Investigation**: Highlights features dominated almost entirely by a single value (e.g., >95% dominance).
- **Datatype Integrity Investigation**: Detects numeric data stored as strings and mixed data types in object/string columns.
- **Cardinality Investigation**: Flags abnormally high or low cardinality columns.
- **Missingness Relationship Investigation**: Computes correlations between column missingness patterns.
- **Structural Health Score**: A composite, weighted score (0–100) representing dataset structural health.

### 2. Evidence Fusion Engine (`fusion/`)
The first reasoning layer of the engine. Rather than leaving findings isolated, it connects them:
- **Evidence Graph (`fusion/graph.py`)**: Uses NetworkX to build an internal graph. Nodes represent Findings, and weighted edges represent relationships (overlapping columns, correlated issues, shared feature families, same metadata category).
- **Extensible Rule Engine (`fusion/rules.py`)**: Groups connected subgraphs using rules (`MissingValuesRule`, `ConstantFeatureRule`, `DuplicateFeatureRule`, `IdentifierRule`, and `GenericGroupRule` fallback) and fuses them into a single high-level `Investigation` model.
- **Cumulative Confidence & Priority**: Applies corroboration bonuses to confidence and calculates priority using severity, count, and confidence.

---

## Installation

Ensure you are using Python 3.12, then run:

```bash
pip install -e ".[dev]"
```

---

## Usage

### CLI Commands

1. **Investigate a dataset (with high-level fused investigations summary):**
   ```bash
   investigation-engine investigate data/data.xlsx
   ```

2. **Investigate and show detailed tables of fused investigations and structural health:**
   ```bash
   investigation-engine investigate data/data.xlsx --verbose
   ```

3. **Run specific investigation modules:**
   ```bash
   investigation-engine investigate data/data.xlsx --modules integrity_investigator
   ```

4. **Output results to a JSON file:**
   ```bash
   investigation-engine investigate data/data.xlsx --output results.json
   ```

5. **Show all registered investigators:**
   ```bash
   investigation-engine info
   ```

6. **View the universal Finding JSON schema:**
   ```bash
   investigation-engine schema
   ```

---

## Testing

Run the full pytest suite to verify investigators and the fusion engine:

```bash
pytest tests/ -v
```
