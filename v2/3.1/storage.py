"""Date-partitioned feature lake — the 10TB-shaped storage layer.

Your current "one parquet per ticker" (~4000 tiny files) is the textbook small-files
anti-pattern: terrible for cross-sectional date-slice reads and crushing on metadata
overhead at scale. This lake instead:

  * partitions by DATE BUCKET (Hive layout) so a single-date cross-section reads a few
    files, not all 4000, and new trading days append as new partitions;
  * casts feature columns to float32 (XGBoost `hist` bins to 256 buckets -> zero
    accuracy loss, ~half the bytes);
  * compresses with zstd and sizes row groups for streaming scans;
  * exposes predicate pushdown on Date / Ticker so the XGBoost trainer streams only
    what it asks for.

Backed by pyarrow.dataset. Read returns pandas; write takes the panel the engine emits.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

# internal partition column (date bucket). MUST NOT start with "_" or "." —
# pyarrow.dataset discovery silently ignores dirs with those prefixes (hive layout
# would then read back an empty schema).
_PART = "part_id"


class FeatureLake:
    def __init__(
        self,
        root: str | Path,
        *,
        date_col: str = "Date",
        ticker_col: str = "Ticker",
        bucket_size: int = 21,        # ~one trading month of dates per partition
        compression: str = "zstd",
        row_group_rows: int = 128_000,
        float32: bool = True,
    ):
        self.root = Path(root)
        self.date_col = date_col
        self.ticker_col = ticker_col
        self.bucket_size = bucket_size
        self.compression = compression
        self.row_group_rows = row_group_rows
        self.float32 = float32

    # -- precision -------------------------------------------------------------
    def _downcast(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.float32:
            return df
        df = df.copy()
        for c in df.columns:
            if c in (self.date_col, self.ticker_col):
                continue
            if pd.api.types.is_float_dtype(df[c]):
                df[c] = df[c].astype(np.float32)
        return df

    def _bucket(self, dates: pd.Series) -> np.ndarray:
        # dense rank of distinct dates // bucket_size -> contiguous partition ids
        codes = pd.factorize(np.sort(dates.unique()))[0]
        rank = dict(zip(np.sort(dates.unique()), codes))
        return (dates.map(rank).to_numpy() // self.bucket_size).astype(np.int64)

    # -- write -----------------------------------------------------------------
    def write(self, panel: pd.DataFrame, *, overwrite: bool = True) -> dict:
        df = self._downcast(panel)
        df[_PART] = self._bucket(df[self.date_col])
        table = pa.Table.from_pandas(df, preserve_index=False)
        fmt = ds.ParquetFileFormat()
        ds.write_dataset(
            table,
            base_dir=str(self.root),
            format=fmt,
            partitioning=ds.partitioning(pa.schema([(_PART, pa.int64())]), flavor="hive"),
            file_options=fmt.make_write_options(compression=self.compression),
            max_rows_per_group=self.row_group_rows,
            existing_data_behavior="delete_matching" if overwrite else "overwrite_or_ignore",
        )
        return self.stats()

    def append(self, panel: pd.DataFrame) -> dict:
        """Add new rows as new partition files without rewriting existing ones."""
        return self.write(panel, overwrite=False)

    # -- read (predicate pushdown) --------------------------------------------
    def _dataset(self) -> ds.Dataset:
        return ds.dataset(str(self.root), format="parquet", partitioning="hive")

    def read(
        self,
        *,
        columns: list[str] | None = None,
        date_range: tuple | None = None,
        tickers: list[str] | None = None,
    ) -> pd.DataFrame:
        dataset = self._dataset()
        expr = None
        if date_range is not None:
            lo, hi = date_range
            e = (pc.field(self.date_col) >= lo) & (pc.field(self.date_col) <= hi)
            expr = e if expr is None else expr & e
        if tickers is not None:
            e = pc.field(self.ticker_col).isin(list(tickers))
            expr = e if expr is None else expr & e
        if columns is not None:
            columns = list(dict.fromkeys([self.date_col, self.ticker_col, *columns]))
        table = dataset.to_table(columns=columns, filter=expr)
        df = table.to_pandas()
        return df.drop(columns=[_PART], errors="ignore")

    # -- introspection ---------------------------------------------------------
    def stats(self) -> dict:
        files = list(self.root.rglob("*.parquet"))
        total_bytes = sum(f.stat().st_size for f in files)
        parts = sorted({p.parent.name for p in files})
        nrows = 0
        for f in files:
            try:
                nrows += pq.ParquetFile(f).metadata.num_rows
            except Exception:
                pass
        return {
            "files": len(files),
            "partitions": len(parts),
            "total_mb": round(total_bytes / 1e6, 3),
            "rows": nrows,
            "root": str(self.root),
        }
