"""Dataset loader with format auto-detection.

Handles loading structured datasets from multiple sources:
    - CSV files (via pandas + pyarrow)
    - XLSX/XLS files (via pandas + openpyxl)
    - PostgreSQL databases (via pandas + psycopg2)

The loader produces both a DataFrame and a DatasetInfo metadata object,
pre-computing structural information that investigation modules need.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from investigation_engine.config.settings import Settings
from investigation_engine.models.dataset import ColumnProfile, DatasetInfo
from investigation_engine.utils.timing import timed


class DataLoadError(Exception):
    """Raised when a dataset cannot be loaded."""


class DatasetLoader:
    """Loads structured datasets and produces metadata.

    Supports CSV, XLSX, and PostgreSQL sources with automatic format
    detection based on file extension or connection string prefix.

    Attributes:
        config: Engine configuration with sampling settings.
    """

    # File extension to loader mapping
    _EXTENSION_MAP: dict[str, str] = {
        ".csv": "csv",
        ".tsv": "csv",
        ".xlsx": "xlsx",
        ".xls": "xlsx",
    }

    def __init__(self, config: Settings | None = None) -> None:
        self._config = config or Settings()

    @timed
    def load(
        self,
        source: str,
        *,
        table_name: str | None = None,
        query: str | None = None,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, DatasetInfo]:
        """Load a dataset from the specified source.

        Args:
            source: File path or PostgreSQL connection string.
            table_name: Table name for PostgreSQL sources.
            query: Custom SQL query for PostgreSQL sources.
            **kwargs: Additional keyword arguments passed to the pandas reader.

        Returns:
            Tuple of (DataFrame, DatasetInfo).

        Raises:
            DataLoadError: If the source cannot be loaded or format is unsupported.
        """
        source_type = self._detect_source_type(source)
        logger.info("Loading dataset from '{}' (detected type: {})", source, source_type)

        try:
            if source_type == "csv":
                df = self._load_csv(source, **kwargs)
            elif source_type == "xlsx":
                df = self._load_xlsx(source, **kwargs)
            elif source_type == "postgresql":
                df = self._load_postgresql(source, table_name=table_name, query=query, **kwargs)
            else:
                raise DataLoadError(f"Unsupported source type: {source_type}")
        except DataLoadError:
            raise
        except Exception as e:
            raise DataLoadError(f"Failed to load dataset from '{source}': {e}") from e

        # Apply sampling if necessary
        df = self._apply_sampling(df)

        # Build dataset info
        dataset_info = self._build_dataset_info(df, source, source_type)

        logger.info(
            "Dataset loaded: {} rows × {} columns ({:.1f} MB)",
            dataset_info.row_count,
            dataset_info.column_count,
            dataset_info.memory_usage_bytes / (1024 * 1024),
        )

        return df, dataset_info

    def _detect_source_type(self, source: str) -> str:
        """Detect the source type from file extension or connection string.

        Args:
            source: File path or connection string.

        Returns:
            Source type string: 'csv', 'xlsx', or 'postgresql'.

        Raises:
            DataLoadError: If the format cannot be determined.
        """
        # Check for PostgreSQL connection string
        if source.startswith(("postgresql://", "postgres://", "postgresql+psycopg2://")):
            return "postgresql"

        # Check file extension
        path = Path(source)
        ext = path.suffix.lower()

        if ext in self._EXTENSION_MAP:
            return self._EXTENSION_MAP[ext]

        raise DataLoadError(
            f"Cannot determine format for '{source}'. "
            f"Supported extensions: {list(self._EXTENSION_MAP.keys())}. "
            f"For PostgreSQL, use a 'postgresql://' connection string."
        )

    def _load_csv(self, path: str, **kwargs: Any) -> pd.DataFrame:
        """Load a CSV file with pyarrow backend for performance.

        Args:
            path: Path to the CSV file.
            **kwargs: Additional arguments for pd.read_csv.

        Returns:
            Loaded DataFrame.
        """
        default_kwargs: dict[str, Any] = {
            "engine": "pyarrow",
            "on_bad_lines": "warn",
        }
        default_kwargs.update(kwargs)

        logger.debug("Reading CSV with pyarrow: {}", path)
        return pd.read_csv(path, **default_kwargs)

    def _load_xlsx(self, path: str, **kwargs: Any) -> pd.DataFrame:
        """Load an Excel file.

        Args:
            path: Path to the Excel file.
            **kwargs: Additional arguments for pd.read_excel.

        Returns:
            Loaded DataFrame.
        """
        default_kwargs: dict[str, Any] = {
            "engine": "openpyxl",
        }
        default_kwargs.update(kwargs)

        logger.debug("Reading Excel file: {}", path)
        return pd.read_excel(path, **default_kwargs)

    def _load_postgresql(
        self,
        connection_string: str,
        *,
        table_name: str | None = None,
        query: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load data from a PostgreSQL database.

        Args:
            connection_string: PostgreSQL connection string.
            table_name: Name of the table to load.
            query: Custom SQL query (takes precedence over table_name).
            **kwargs: Additional arguments for pd.read_sql.

        Returns:
            Loaded DataFrame.

        Raises:
            DataLoadError: If neither table_name nor query is provided.
        """
        if query:
            sql = query
        elif table_name:
            sql = f"SELECT * FROM {table_name}"  # noqa: S608
        else:
            raise DataLoadError(
                "PostgreSQL source requires either 'table_name' or 'query' parameter."
            )

        logger.debug("Executing SQL query on PostgreSQL")
        return pd.read_sql(sql, connection_string, **kwargs)

    def _apply_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply random sampling if the dataset exceeds the configured maximum.

        Args:
            df: The loaded DataFrame.

        Returns:
            Original or sampled DataFrame.
        """
        max_rows = self._config.engine.max_rows_sample

        if len(df) > max_rows:
            logger.warning(
                "Dataset has {} rows, exceeding max_rows_sample={}. "
                "Sampling {} rows randomly.",
                len(df),
                max_rows,
                max_rows,
            )
            return df.sample(n=max_rows, random_state=42).reset_index(drop=True)

        return df

    def _build_dataset_info(
        self,
        df: pd.DataFrame,
        source: str,
        source_type: str,
    ) -> DatasetInfo:
        """Build a DatasetInfo object from a loaded DataFrame.

        Args:
            df: The loaded DataFrame.
            source: Original source path/string.
            source_type: Detected source type.

        Returns:
            DatasetInfo with pre-computed metadata.
        """
        # Build column profiles
        columns: list[ColumnProfile] = []
        for col_name in df.columns:
            col = df[col_name]
            dtype_str = str(col.dtype)

            is_numeric = pd.api.types.is_numeric_dtype(col)
            is_bool = pd.api.types.is_bool_dtype(col)
            is_datetime = pd.api.types.is_datetime64_any_dtype(col)
            is_categorical = (
                pd.api.types.is_categorical_dtype(col)
                or (pd.api.types.is_object_dtype(col) and not is_datetime)
                or (is_numeric and col.nunique() < 20 and len(df) > 100)
            )

            # Sample non-null values
            non_null = col.dropna()
            sample_values = (
                non_null.head(5).tolist() if len(non_null) > 0 else []
            )

            columns.append(
                ColumnProfile(
                    name=str(col_name),
                    dtype=dtype_str,
                    null_count=int(col.isnull().sum()),
                    unique_count=int(col.nunique()),
                    is_numeric=is_numeric,
                    is_categorical=is_categorical,
                    is_datetime=is_datetime,
                    is_boolean=is_bool,
                    sample_values=sample_values,
                )
            )

        # Derive dataset name from source
        name = Path(source).stem if not source.startswith("postgresql") else source.split("/")[-1]

        # Compute memory usage
        memory_bytes = int(df.memory_usage(deep=True).sum())

        # Check for missing values
        has_missing = bool(df.isnull().any().any())

        # Count duplicates
        duplicate_count = int(df.duplicated().sum())

        return DatasetInfo(
            name=name,
            source=source,
            source_type=source_type,
            row_count=len(df),
            column_count=len(df.columns),
            column_types={str(col): str(df[col].dtype) for col in df.columns},
            columns=columns,
            memory_usage_bytes=memory_bytes,
            has_missing_values=has_missing,
            duplicate_row_count=duplicate_count,
            loaded_at=datetime.now(timezone.utc),
        )
