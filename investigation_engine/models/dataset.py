"""Dataset metadata model.

Wraps raw DataFrame metadata to provide investigation context.
The DatasetInfo model travels with every investigation run, providing
modules with structural information about the dataset without requiring
them to recompute it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, computed_field


class ColumnProfile(BaseModel):
    """Lightweight profile for a single dataset column.

    Captures type information and basic statistics computed during
    dataset loading, so investigation modules don't need to redundantly
    calculate them.
    """

    name: str = Field(..., description="Column name.")
    dtype: str = Field(..., description="Pandas dtype as string.")
    null_count: int = Field(default=0, ge=0, description="Number of null values.")
    unique_count: int = Field(default=0, ge=0, description="Number of unique values.")
    is_numeric: bool = Field(default=False, description="Whether the column is numeric.")
    is_categorical: bool = Field(default=False, description="Whether the column is categorical.")
    is_datetime: bool = Field(default=False, description="Whether the column is datetime.")
    is_boolean: bool = Field(default=False, description="Whether the column is boolean.")
    sample_values: list[Any] = Field(
        default_factory=list,
        description="Sample of up to 5 non-null values for quick inspection.",
    )


class DatasetInfo(BaseModel):
    """Metadata about a loaded dataset.

    Attributes:
        name: Human-readable name (typically the filename).
        source: File path or connection string used to load the data.
        source_type: Type of source (csv, xlsx, postgresql).
        row_count: Total number of rows.
        column_count: Total number of columns.
        column_types: Mapping of column name to pandas dtype string.
        columns: Detailed column profiles.
        memory_usage_bytes: Approximate memory footprint in bytes.
        has_missing_values: Whether any column has null values.
        duplicate_row_count: Number of fully duplicated rows.
        loaded_at: UTC timestamp when the dataset was loaded.
        metadata: Extensible metadata for loader-specific information.
    """

    name: str = Field(..., min_length=1, description="Dataset name.")
    source: str = Field(..., min_length=1, description="File path or connection string.")
    source_type: str = Field(
        ...,
        description="Source format: 'csv', 'xlsx', or 'postgresql'.",
    )
    row_count: int = Field(..., ge=0, description="Total row count.")
    column_count: int = Field(..., ge=0, description="Total column count.")
    column_types: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of column names to dtype strings.",
    )
    columns: list[ColumnProfile] = Field(
        default_factory=list,
        description="Detailed column profiles.",
    )
    memory_usage_bytes: int = Field(
        default=0,
        ge=0,
        description="Approximate memory usage in bytes.",
    )
    has_missing_values: bool = Field(
        default=False,
        description="Whether the dataset contains any null values.",
    )
    duplicate_row_count: int = Field(
        default=0,
        ge=0,
        description="Number of fully duplicated rows.",
    )
    loaded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the dataset was loaded.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible metadata from the data loader.",
    )

    model_config = {
        "frozen": False,
        "json_schema_extra": {
            "title": "DatasetInfo",
            "description": "Structural metadata about a loaded dataset.",
        },
    }

    @computed_field  # type: ignore[prop-decorator]
    @property
    def numeric_columns(self) -> list[str]:
        """List of numeric column names."""
        return [c.name for c in self.columns if c.is_numeric]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def categorical_columns(self) -> list[str]:
        """List of categorical column names."""
        return [c.name for c in self.columns if c.is_categorical]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def datetime_columns(self) -> list[str]:
        """List of datetime column names."""
        return [c.name for c in self.columns if c.is_datetime]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def missing_value_ratio(self) -> float:
        """Ratio of total missing values to total cells."""
        total_cells = self.row_count * self.column_count
        if total_cells == 0:
            return 0.0
        total_missing = sum(c.null_count for c in self.columns)
        return total_missing / total_cells

    def get_column_profile(self, column_name: str) -> ColumnProfile | None:
        """Retrieve a column profile by name.

        Args:
            column_name: The column to look up.

        Returns:
            The ColumnProfile if found, None otherwise.
        """
        for col in self.columns:
            if col.name == column_name:
                return col
        return None
