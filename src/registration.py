from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_preprocessing import SPENDING_COLUMNS


REGISTERED_COLUMNS = [
    "Year_Birth",
    "Education",
    "Marital_Status",
    "Income",
    "Kidhome",
    "Teenhome",
    "Dt_Customer",
    "Recency",
    *SPENDING_COLUMNS,
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
    "Complain",
    "Response",
]


def build_customer_record(values: dict) -> pd.DataFrame:
    """Build a one-row raw record using the source CSV column names."""
    record = {column: values[column] for column in REGISTERED_COLUMNS}
    record["Dt_Customer"] = pd.Timestamp(values["Dt_Customer"]).strftime("%d-%m-%Y")
    return pd.DataFrame([record], columns=REGISTERED_COLUMNS)


def validate_customer_record(record: pd.DataFrame) -> list[str]:
    """Return user-facing validation errors for a new raw customer record."""
    errors = []
    missing = [column for column in REGISTERED_COLUMNS if column not in record.columns]
    if missing:
        return [f"Missing required columns: {', '.join(missing)}"]
    row = record.iloc[0]
    if not 1900 <= int(row["Year_Birth"]) <= 2026:
        errors.append("Year of birth must be between 1900 and 2026.")
    if float(row["Income"]) < 0:
        errors.append("Income cannot be negative.")
    if int(row["Kidhome"]) < 0 or int(row["Teenhome"]) < 0:
        errors.append("Household child counts cannot be negative.")
    numeric_fields = [
        "Recency",
        *SPENDING_COLUMNS,
        "NumDealsPurchases",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
        "NumWebVisitsMonth",
        "Complain",
        "Response",
    ]
    if any(float(row[field]) < 0 for field in numeric_fields):
        errors.append("Purchase, recency, visit, and campaign values cannot be negative.")
    if int(row["Complain"]) not in (0, 1) or int(row["Response"]) not in (0, 1):
        errors.append("Complain and Response must be 0 or 1.")
    if pd.isna(pd.to_datetime(row["Dt_Customer"], dayfirst=True, errors="coerce")):
        errors.append("Enrollment date is invalid.")
    return errors


def load_registered_customers(path: str | Path) -> pd.DataFrame:
    """Load persisted registrations, returning an empty frame when none exist."""
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame(columns=[*REGISTERED_COLUMNS, "ID", "Cluster", "Is_Registered"])
    registered = pd.read_csv(file_path)
    missing = [column for column in [*REGISTERED_COLUMNS, "ID", "Cluster"] if column not in registered]
    if missing:
        raise ValueError(f"Updated customer file is missing columns: {', '.join(missing)}")
    return registered


def _fingerprint(record: pd.DataFrame) -> tuple:
    values = record[REGISTERED_COLUMNS].iloc[0].copy()
    values["Dt_Customer"] = pd.Timestamp(values["Dt_Customer"]).strftime("%d-%m-%Y")
    return tuple(values.tolist())


def save_registered_customer(
    record: pd.DataFrame,
    cluster: int,
    updated_path: str | Path,
    original: pd.DataFrame,
) -> tuple[bool, str, pd.DataFrame]:
    """Persist a registration without modifying the original CSV."""
    errors = validate_customer_record(record)
    if errors:
        return False, " ".join(errors), pd.DataFrame()
    path = Path(updated_path)
    existing = load_registered_customers(path)
    candidate_key = _fingerprint(record)
    duplicate = any(
        _fingerprint(existing.loc[[index], REGISTERED_COLUMNS])
        == candidate_key
        for index in existing.index
    )
    if duplicate:
        return False, "This customer is already registered.", existing

    all_ids = pd.to_numeric(
        pd.concat([original["ID"], existing["ID"]], ignore_index=True),
        errors="coerce",
    ).dropna()
    next_id = int(all_ids.max()) + 1 if not all_ids.empty else 1
    saved = record.copy()
    saved["ID"] = next_id
    saved["Cluster"] = int(cluster)
    saved["Is_Registered"] = True
    columns = [*REGISTERED_COLUMNS, "ID", "Cluster", "Is_Registered"]
    combined = pd.concat([existing, saved[columns]], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return True, f"Customer {next_id} was added to the working dataset.", combined
