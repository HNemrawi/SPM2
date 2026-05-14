"""Tempdir-backed Arrow IPC store for the active session's DataFrame.

The pandas DataFrame held in ``st.session_state[SessionKeys.DF]`` is the
largest single object in memory. We mirror it to a column-major Arrow IPC
file in the OS tempdir (typically 5-10x smaller than the pandas frame on
disk thanks to dictionary-encoded categoricals and the lack of Python-object
overhead) and expose two views over it:

  - ``ArrowStore.to_pandas()`` — round-trips back to a pandas DataFrame.
  - ``ArrowStore.duckdb_view()`` — registers the Arrow Table with an
    in-process DuckDB connection so SQL filter/aggregate pushdown is
    available (see ``dashboard/filters.py:_apply_filters_cached``).

Lifetime: the store is built lazily on first access and cached via
``@st.cache_resource`` keyed on the file's content hash, so it is reused
across reruns of the same uploaded file. When the cache entry is evicted
(``max_entries=2``) or the process exits (``atexit`` hook), the tempdir is
removed. The data never leaves the host process.
"""

from __future__ import annotations

import atexit
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as paipc
import streamlit as st

try:
    import duckdb  # type: ignore[import-not-found]

    HAS_DUCKDB = True
except ImportError:  # pragma: no cover - duckdb is a runtime dependency
    duckdb = None  # type: ignore[assignment]
    HAS_DUCKDB = False

logger = logging.getLogger(__name__)


class ArrowStore:
    """Holds an Arrow Table plus its on-disk IPC representation and a lazily
    opened DuckDB connection over the same table."""

    __slots__ = ("_path", "_table", "file_hash", "_duckdb_con", "_view_name")

    def __init__(self, path: Path, table: pa.Table, file_hash: str) -> None:
        self._path = path
        self._table = table
        self.file_hash = file_hash
        self._duckdb_con: Optional["duckdb.DuckDBPyConnection"] = None
        self._view_name = "data"

    @property
    def path(self) -> Path:
        return self._path

    def table(self) -> pa.Table:
        return self._table

    def to_pandas(self) -> pd.DataFrame:
        """Round-trip back to a pandas DataFrame and re-stamp the file hash
        on ``df.attrs`` so downstream caches behave identically."""
        df = self._table.to_pandas(zero_copy_only=False)
        df.attrs["hmis_file_hash"] = self.file_hash
        return df

    def duckdb_view(self, name: Optional[str] = None) -> "duckdb.DuckDBPyConnection":
        """Return an in-process DuckDB connection with the Arrow Table
        registered as a view (default name ``data``). The connection is
        re-used across calls so query planning is amortized."""
        if not HAS_DUCKDB:
            raise RuntimeError(
                "duckdb is not installed — fall back to the pandas filter path."
            )
        view_name = name or self._view_name
        if self._duckdb_con is None:
            con = duckdb.connect(":memory:")
            con.register(view_name, self._table)
            self._duckdb_con = con
            self._view_name = view_name
        elif name and name != self._view_name:
            self._duckdb_con.register(name, self._table)
            self._view_name = name
        return self._duckdb_con

    def close(self) -> None:
        """Tear down the DuckDB connection and remove the tempdir. Safe to
        call multiple times."""
        if self._duckdb_con is not None:
            try:
                self._duckdb_con.close()
            except Exception:  # pragma: no cover - defensive
                pass
            self._duckdb_con = None
        try:
            if self._path.exists():
                shutil.rmtree(self._path.parent, ignore_errors=True)
        except OSError:  # pragma: no cover - defensive
            pass


def _build_from_df(df: pd.DataFrame, file_hash: str) -> ArrowStore:
    """Convert ``df`` to Arrow, write an IPC file under the OS tempdir, and
    return a fresh ``ArrowStore``. Registers an ``atexit`` cleanup hook."""
    tmpdir = Path(tempfile.mkdtemp(prefix="hmis_arrow_"))
    path = tmpdir / "data.arrow"
    table = pa.Table.from_pandas(df, preserve_index=False)
    with pa.OSFile(str(path), "wb") as sink:
        with paipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    store = ArrowStore(path=path, table=table, file_hash=file_hash)
    atexit.register(store.close)
    logger.debug(
        "ArrowStore built: rows=%d cols=%d file_hash=%s path=%s",
        len(df),
        len(df.columns),
        file_hash[:8],
        path,
    )
    return store


@st.cache_resource(show_spinner=False, max_entries=2)
def get_arrow_store(file_hash: str, _df: pd.DataFrame) -> ArrowStore:
    """Session-scoped cached builder. The ``file_hash`` IS the cache key
    (one Arrow Table per uploaded file); ``_df`` is underscore-prefixed so
    Streamlit does not try to hash it. ``max_entries=2`` keeps the current
    file + one rollback slot."""
    return _build_from_df(_df, file_hash)
