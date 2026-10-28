from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SPENDING_COLUMNS = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
]

DROP_COLUMNS = [
    "ID",
    "Year_Birth",
    "Marital_Status",
    "Kidhome",
    "Teenhome",
    "Dt_Customer",
    *SPENDING_COLUMNS,
]

CATEGORICAL_FEATURES = ["Education", "Living_status"]


@dataclass(frozen=True)
class PreparedData:
    """All data products needed by the dashboard and clustering models."""

    raw: pd.DataFrame
    customer_data: pd.DataFrame
    cleaned: pd.DataFrame
    encoded: pd.DataFrame
    scaled: np.ndarray
    pca_coordinates: np.ndarray
    pca_model: PCA
    scaler: StandardScaler
    encoder: OneHotEncoder
    reference_date: pd.Timestamp
    reference_year: int
    income_median: float


def load_customer_data(path: str | Path) -> pd.DataFrame:
    """Load the source CSV and raise a useful error when it is unavailable."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Customer data file was not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError("The customer data file is empty.")
    return frame


def prepare_data(raw: pd.DataFrame, reference_year: int = 2026) -> PreparedData:
    """Reproduce the notebook's transformations and create model features."""
    required = {
        "Year_Birth",
        "Education",
        "Marital_Status",
        "Income",
        "Kidhome",
        "Teenhome",
        "Dt_Customer",
        *SPENDING_COLUMNS,
    }
    missing_columns = sorted(required.difference(raw.columns))
    if missing_columns:
        raise ValueError(
            "The customer data is missing required columns: "
            + ", ".join(missing_columns)
        )

    income_median = float(raw["Income"].median())
    parsed_dates = pd.to_datetime(raw["Dt_Customer"], dayfirst=True, errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("One or more Dt_Customer values could not be parsed.")
    reference_date = parsed_dates.max()
    working = engineer_customer_data(
        raw,
        reference_date=reference_date,
        reference_year=reference_year,
        income_median=income_median,
    )

    customer_data = working[
        (working["Age"] < 90) & (working["Income"] < 600000)
    ].copy()
    cleaned = customer_data.drop(columns=DROP_COLUMNS)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded_categories = encoder.fit_transform(cleaned[CATEGORICAL_FEATURES])
    encoded_category_frame = pd.DataFrame(
        encoded_categories,
        columns=encoder.get_feature_names_out(CATEGORICAL_FEATURES),
        index=cleaned.index,
    )
    encoded = pd.concat(
        [cleaned.drop(columns=CATEGORICAL_FEATURES), encoded_category_frame],
        axis=1,
    )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(encoded)
    pca_model = PCA(n_components=3)
    pca_coordinates = pca_model.fit_transform(scaled)

    return PreparedData(
        raw=raw.copy(),
        customer_data=customer_data,
        cleaned=cleaned,
        encoded=encoded,
        scaled=scaled,
        pca_coordinates=pca_coordinates,
        pca_model=pca_model,
        scaler=scaler,
        encoder=encoder,
        reference_date=reference_date,
        reference_year=reference_year,
        income_median=income_median,
    )


def engineer_customer_data(
    raw: pd.DataFrame,
    reference_date: pd.Timestamp,
    reference_year: int,
    income_median: float,
) -> pd.DataFrame:
    """Apply the notebook feature engineering to one or more customer records."""
    working = raw.copy()
    numeric_columns = [
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
    ]
    numeric_columns = [column for column in numeric_columns if column in working.columns]
    for column in numeric_columns:
        working[column] = working[column].map(
            lambda value: value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value
        )
        working[column] = pd.to_numeric(working[column], errors="coerce")
    if "Year_Birth" in working.columns:
        working["Year_Birth"] = working["Year_Birth"].fillna(working["Year_Birth"].median())
    if "Income" in working.columns:
        working["Income"] = working["Income"].fillna(income_median)
    for column in [
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
    ]:
        if column in working.columns:
            working[column] = working[column].fillna(0)
    working["Age"] = reference_year - working["Year_Birth"]
    working["Dt_Customer"] = pd.to_datetime(
        working["Dt_Customer"], dayfirst=True, errors="coerce"
    )
    if working["Dt_Customer"].isna().any():
        raise ValueError("One or more Dt_Customer values could not be parsed.")
    working["customer_Tenure_Days"] = (
        reference_date - working["Dt_Customer"]
    ).dt.days
    working["Total_Spending"] = working[SPENDING_COLUMNS].sum(axis=1)
    working["Total_Children"] = working["Kidhome"] + working["Teenhome"]
    working["Education"] = working["Education"].replace(
        {
            "Basic": "Undergraduate",
            "2n Cycle": "Undergraduate",
            "Graduation": "Graduate",
            "PhD": "Postgraduate",
            "Master": "Postgraduate",
        }
    )
    working["Living_status"] = working["Marital_Status"].replace(
        {
            "Married": "Partner",
            "Together": "Partner",
            "Single": "Alone",
            "Divorced": "Alone",
            "Widow": "Alone",
            "Absurd": "Alone",
            "YOLO": "Alone",
        }
    )
    return working


def transform_new_customer(record: pd.DataFrame, prepared: PreparedData) -> tuple[
    pd.DataFrame, np.ndarray, np.ndarray
]:
    """Transform new raw records with the already-fitted training transformers."""
    engineered = engineer_customer_data(
        record,
        reference_date=prepared.reference_date,
        reference_year=prepared.reference_year,
        income_median=prepared.income_median,
    )
    feature_frame = engineered.drop(columns=DROP_COLUMNS, errors="ignore")
    encoded_categories = prepared.encoder.transform(feature_frame[CATEGORICAL_FEATURES])
    encoded_category_frame = pd.DataFrame(
        encoded_categories,
        columns=prepared.encoder.get_feature_names_out(CATEGORICAL_FEATURES),
        index=feature_frame.index,
    )
    encoded = pd.concat(
        [feature_frame.drop(columns=CATEGORICAL_FEATURES), encoded_category_frame],
        axis=1,
    ).reindex(columns=prepared.encoded.columns, fill_value=0)
    scaled = prepared.scaler.transform(encoded)
    coordinates = prepared.pca_model.transform(scaled)
    return engineered, scaled, coordinates
