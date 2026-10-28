from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from .data_preprocessing import PreparedData, SPENDING_COLUMNS, engineer_customer_data
from .registration import REGISTERED_COLUMNS


DATABASE_COLUMNS  = [
    "ID",
    *REGISTERED_COLUMNS,
    "Age",
    "customer_Tenure_Days",
    "Total_Spending",
    "Total_Children",
    "Living_status",
    "Cluster",
    "Created_At",
    "Source",
]

NUMERIC_COLUMNS = [
    "ID",
    "Year_Birth",
    "Income",
    "Kidhome",
    "Teenhome",
    "Recency",
    *SPENDING_COLUMNS,
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
    "Complain",
    "Response",
    "Age",
    "customer_Tenure_Days",
    "Total_Spending",
    "Total_Children",
    "Cluster",
]

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS customers (
    ID INTEGER PRIMARY KEY,
    Year_Birth INTEGER NOT NULL,
    Education TEXT NOT NULL,
    Marital_Status TEXT NOT NULL,
    Income REAL,
    Kidhome INTEGER NOT NULL,
    Teenhome INTEGER NOT NULL,
    Dt_Customer TEXT NOT NULL,
    Recency INTEGER NOT NULL,
    MntWines INTEGER NOT NULL,
    MntFruits INTEGER NOT NULL,
    MntMeatProducts INTEGER NOT NULL,
    MntFishProducts INTEGER NOT NULL,
    MntSweetProducts INTEGER NOT NULL,
    MntGoldProds INTEGER NOT NULL,
    NumDealsPurchases INTEGER NOT NULL,
    NumWebPurchases INTEGER NOT NULL,
    NumCatalogPurchases INTEGER NOT NULL,
    NumStorePurchases INTEGER NOT NULL,
    NumWebVisitsMonth INTEGER NOT NULL,
    Complain INTEGER NOT NULL,
    Response INTEGER NOT NULL,
    Age INTEGER NOT NULL,
    customer_Tenure_Days INTEGER NOT NULL,
    Total_Spending REAL NOT NULL,
    Total_Children INTEGER NOT NULL,
    Living_status TEXT NOT NULL,
    Cluster INTEGER NOT NULL CHECK (Cluster BETWEEN 0 AND 3),
    Created_At TEXT NOT NULL,
    Source TEXT NOT NULL CHECK (Source IN ('original', 'registered'))
)
"""


@contextmanager
def get_connection(path: str | Path):
    connection = sqlite3.connect(Path(path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db(path: str | Path) -> None:
    """Create the database schema idempotently."""
    try:
        with get_connection(path) as connection:
            connection.execute(CREATE_TABLE_SQL)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_customers_cluster ON customers (Cluster)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_customers_source ON customers (Source)"
            )
    except sqlite3.Error as error:
        raise RuntimeError(f"Unable to initialize SQLite database: {error}") from error


def _as_database_row(
    engineered: pd.Series,
    customer_id: int,
    cluster: int,
    source: str,
    created_at: str | None = None,
) -> dict:
    row = {column: engineered.get(column) for column in REGISTERED_COLUMNS}
    if isinstance(row["Dt_Customer"], pd.Timestamp):
        row["Dt_Customer"] = row["Dt_Customer"].strftime("%d-%m-%Y")
    elif pd.notna(row["Dt_Customer"]):
        row["Dt_Customer"] = str(row["Dt_Customer"])
    row.update(
        {
            "ID": int(customer_id),
            "Age": int(engineered["Age"]),
            "customer_Tenure_Days": int(engineered["customer_Tenure_Days"]),
            "Total_Spending": float(engineered["Total_Spending"]),
            "Total_Children": int(engineered["Total_Children"]),
            "Living_status": engineered["Living_status"],
            "Cluster": int(cluster),
            "Created_At": created_at or datetime.now(timezone.utc).isoformat(),
            "Source": source,
        }
    )
    if pd.isna(row["Income"]):
        row["Income"] = None
    return row


def _insert_rows(
    connection: sqlite3.Connection,
    rows: list[dict],
    ignore_duplicates: bool,
) -> int:
    if not rows:
        return 0
    placeholders = ", ".join("?" for _ in DATABASE_COLUMNS)
    verb = "INSERT OR IGNORE" if ignore_duplicates else "INSERT"
    sql = f"INSERT {'' if verb == 'INSERT' else 'OR IGNORE'} INTO customers ({', '.join(DATABASE_COLUMNS)}) VALUES ({placeholders})"
    inserted = 0
    for row in rows:
        cursor = connection.execute(sql, [row.get(column) for column in DATABASE_COLUMNS])
        inserted += cursor.rowcount
    return inserted


def seed_database_from_csv(
    path: str | Path,
    csv_path: str | Path,
    prepared: PreparedData,
    labels: list[int],
    registered_csv_path: str | Path | None = None,
) -> int:
    """Seed original data once and migrate legacy registered rows if available."""
    init_db(path)
    with get_connection(path) as connection:
        existing = connection.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        if existing:
            return 0

        original = prepared.customer_data.copy()
        rows = []
        for position, (_, customer) in enumerate(original.iterrows()):
            rows.append(
                _as_database_row(
                    customer,
                    int(customer["ID"]),
                    int(labels[position]),
                    "original",
                )
            )

        if registered_csv_path and Path(registered_csv_path).exists():
            registered = pd.read_csv(registered_csv_path)
            if not registered.empty:
                engineered = engineer_customer_data(
                    registered,
                    prepared.reference_date,
                    prepared.reference_year,
                    prepared.income_median,
                )
                for _, customer in engineered.iterrows():
                    rows.append(
                        _as_database_row(
                            customer,
                            int(customer["ID"]),
                            int(customer["Cluster"]),
                            "registered",
                        )
                    )
        return _insert_rows(connection, rows, ignore_duplicates=True)


def get_all_customers(path: str | Path) -> pd.DataFrame:
    init_db(path)
    with get_connection(path) as connection:
        customers = pd.read_sql_query(
            "SELECT * FROM customers ORDER BY ID",
            connection,
        )
    for column in NUMERIC_COLUMNS:
        if column in customers:
            customers[column] = pd.to_numeric(customers[column], errors="coerce")
    return customers


def get_customer_by_id(path: str | Path, customer_id: int) -> pd.Series | None:
    init_db(path)
    with get_connection(path) as connection:
        row = connection.execute(
            "SELECT * FROM customers WHERE ID = ?",
            (int(customer_id),),
        ).fetchone()
    return pd.Series(dict(row)) if row else None


def get_cluster_summary(path: str | Path) -> pd.DataFrame:
    init_db(path)
    with get_connection(path) as connection:
        return pd.read_sql_query(
            """
            SELECT Cluster, COUNT(*) AS Customers,
                   AVG(Income) AS Average_Income,
                   AVG(Total_Spending) AS Average_Spending
            FROM customers
            GROUP BY Cluster
            ORDER BY Cluster
            """,
            connection,
        )


def get_total_customer_count(path: str | Path) -> int:
    init_db(path)
    with get_connection(path) as connection:
        return int(connection.execute("SELECT COUNT(*) FROM customers").fetchone()[0])


def get_database_status(path: str | Path) -> dict:
    init_db(path)
    with get_connection(path) as connection:
        total = int(connection.execute("SELECT COUNT(*) FROM customers").fetchone()[0])
        registered = int(
            connection.execute(
                "SELECT COUNT(*) FROM customers WHERE Source = ?",
                ("registered",),
            ).fetchone()[0]
        )
        latest = connection.execute(
            "SELECT MAX(Created_At) FROM customers WHERE Source = ?",
            ("registered",),
        ).fetchone()[0]
        counts = pd.read_sql_query(
            "SELECT Cluster, COUNT(*) AS Customers FROM customers GROUP BY Cluster ORDER BY Cluster",
            connection,
        )
    return {"total": total, "registered": registered, "latest": latest, "counts": counts}


def insert_customer(
    path: str | Path,
    record: pd.DataFrame,
    prepared: PreparedData,
    cluster: int,
) -> int:
    """Insert a validated raw record in one transaction and return its ID."""
    if cluster not in range(4):
        raise ValueError("A registered customer must have a cluster from 0 to 3.")
    from .registration import validate_customer_record

    errors = validate_customer_record(record)
    if errors:
        raise ValueError(" ".join(errors))
    engineered = engineer_customer_data(
        record,
        prepared.reference_date,
        prepared.reference_year,
        prepared.income_median,
    ).iloc[0]
    with get_connection(path) as connection:
        existing_ids = connection.execute(
            "SELECT ID FROM customers ORDER BY ID DESC LIMIT 1"
        ).fetchone()
        row = _as_database_row(engineered, 0, cluster, "registered")
        try:
            duplicate_conditions = " AND ".join(
                f"{column} IS ?" if row[column] is None else f"{column} = ?"
                for column in REGISTERED_COLUMNS
            )
            duplicate = connection.execute(
                f"SELECT ID FROM customers WHERE {duplicate_conditions} LIMIT 1",
                tuple(row[column] for column in REGISTERED_COLUMNS),
            ).fetchone()
            if duplicate:
                raise ValueError(
                    f"This customer matches existing customer ID {int(duplicate[0])}."
                )
            customer_id = int(existing_ids[0]) + 1 if existing_ids else 1
            row["ID"] = customer_id
            _insert_rows(connection, [row], ignore_duplicates=False)
        except sqlite3.IntegrityError as error:
            raise ValueError(f"Customer could not be inserted: {error}") from error
    return customer_id
