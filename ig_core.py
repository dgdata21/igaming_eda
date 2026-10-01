# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 09:17:02 2026
@author: flotp

ig_core.py
"""

import duckdb
import pandas as pd
import os

from igaming_eda.ig_vars import CATS_ARG, ROWS, MEMORY, TABLES, THREADS, FILES


# ####################################################
# PRINT OPTIONS
# ####################################################
def print_opt():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.float_format", "{:,.3f}".format)


def section(title):
    width = 110
    print()
    print("=" * width)
    print(title.center(width))
    print("=" * width)


# ####################################################
# PROCESS DATA
# ####################################################
def cats(
    df: pd.DataFrame,
    TEXTS: list[str] | None = None,
    NUMS: list[str] | None = None,
    FLOATS: list[str] | None = None,
    N64: list[str] | None = None,
    DATES_CAT: list[str] | None = None,
    DT: list[str] | None = None,
) -> pd.DataFrame:
    df = df.copy()
    if TEXTS is not None:
        for col in TEXTS:
            if col in df.columns:
                df[col] = pd.Categorical(df[col])
    if NUMS is not None:
        for col in NUMS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], downcast="integer")
    if FLOATS is not None:
        for col in FLOATS:
            if col in df.columns:
                df[col] = df[col].astype("float32")
    if N64 is not None:
        for col in N64:
            if col in df.columns:
                df[col] = df[col].astype("int64")
    if DATES_CAT is not None:
        for col in DATES_CAT:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).dt.date.astype("category")
    if DT is not None:
        for col in DT:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
    return df


def _to_category(df):
    for col in df.columns:
        s = df[col]
        if isinstance(s.dtype, pd.CategoricalDtype):
            continue  # уже category
        if pd.api.types.is_string_dtype(s) and s.nunique() <= max(
            1, len(s) * 0.05
        ):
            df[col] = s.astype("category")
    return df


# ####################################################
# LOAD DATA
# ####################################################
def _connect():
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{MEMORY}'")
    con.execute(f"SET threads={THREADS}")
    for name, path in TABLES.items():
        if path.exists():
            con.execute(
                f"CREATE VIEW {name} AS SELECT *"
                f"FROM read_parquet('{path.as_posix()}')"
            )
    return con


def schema(table):
    return q(f"DESCRIBE {table}")


def q(sql, params=None):
    """Пример:
    q("SELECT COUNT(*) AS n FROM casino")
    q("SELECT player_id, SUM(total_bet) AS s
       FROM casino WHERE ts >= ? GROUP BY 1",
      ["2025-01-01"])
    """
    con = _connect()
    try:
        return con.execute(sql, params or []).df()
    finally:
        con.close()


def load(
    table,
    columns=None,
    where=None,
    limit=None,
    params=None,
    use_cats=True,
    info=True,
    head=True,
    max_rows=ROWS,
):
    cols = ", ".join(f'"{c}"' for c in columns) if columns else "*"
    where_sql = f" WHERE {where}" if where else ""
    params = params or []

    con = _connect()
    try:
        if limit is None and max_rows is not None:
            n = con.execute(
                f"SELECT COUNT(*) FROM {table}{where_sql}", params
            ).fetchone()[0]
            if n > max_rows:
                raise MemoryError(
                    f"Найдено {n:,} строк, это больше лимита {max_rows:,}. "
                    "Добавьте where, уберите лишние колонки, задайте max_rows"
                    " или посчитайте через q()."
                )
        sql = f"SELECT {cols} FROM {table}{where_sql}"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        df = con.execute(sql, params).df()
    finally:
        con.close()

    if use_cats:
        df = cats(df, **CATS_ARG.get(table, {}))
    df = _to_category(df)

    if info:
        print(f"\n------ {table.upper()} INFO ------")
        df.info()
    if head:
        section(f"{table.upper()} HEAD")
        print(df.head(6))
    return df


def load_all(**kwargs):
    return tuple(load(name, max_rows=ROWS, **kwargs) for name in FILES)


def check_df(df: pd.DataFrame, name: str, id_columns: list[str] | None = None):
    print(f"\n{'='*20} CHECKING: {name.upper()} {'='*20}\n")
    missing_summary = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_percent": df.isna().mean() * 100,
        }
    )

    print(f" {name.upper()} MISSING VALUES ".center(50, "-"))
    print(f"checking columns - {id_columns}\n")
    active_missing = missing_summary[missing_summary["missing_count"] > 0]
    if not active_missing.empty:
        print(active_missing.round(2))
    else:
        print("No missing values.")

    print("\n", f" {name.upper()} DUPLICATES ".center(50, "-"))
    print(f"checking columns - {id_columns}\n")

    subset = [c for c in (id_columns or []) if c in df.columns] or None
    duplicate_rows = df[df.duplicated(subset=subset, keep=False)]

    if not duplicate_rows.empty:
        print(f"Duplicated rows: {len(duplicate_rows)}")
        if subset:
            print(duplicate_rows.round(2).sort_values(id_columns).head(10))
        else:
            print(duplicate_rows.round(2).head(10))
    else:
        print("No duplicates.")


def save_parquet(
    df: pd.DataFrame,
    name: str,
    float_cols: list[str] = None,
    decimals: int = 4,
) -> None:
    base_dir = r"D:/code/practice/igaming/ig_pq"
    file_path = os.path.join(base_dir, f"{name}.parquet")
    if float_cols:
        cols_to_round = [c for c in float_cols if c in df.columns]
        round_dict = {col: decimals for col in cols_to_round}
        df_to_save = df.round(round_dict)
    else:
        df_to_save = df

    df_to_save.to_parquet(
        path=file_path,
        engine="pyarrow",
        compression="brotli",
        index=False,
    )
    print(f"Файл успешно сохранен: {file_path}")


# save_parquet(df_full, "bonus", float_cols=BONUS_COLS.get("FLOATS"))
