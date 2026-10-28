from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

from .data_preprocessing import PreparedData, transform_new_customer


CLUSTER_COLORS = {0: "#E53935", 1: "#1E88E5", 2: "#FDD835", 3: "#43A047"}
CLUSTER_INTERPRETATIONS = {
    0: "Low/moderate income and low/moderate spending",
    1: "High income and high spending",
    2: "Low income and low spending",
    3: "Moderate income and high spending",
}


@dataclass(frozen=True)
class ClusteringResult:
    algorithm: str
    n_clusters: int
    labels: np.ndarray
    customers: pd.DataFrame
    profile: pd.DataFrame
    wcss: pd.DataFrame
    silhouette: pd.DataFrame
    optimal_k: int | None
    model: KMeans | AgglomerativeClustering
    training_coordinates: np.ndarray


def evaluate_cluster_counts(
    coordinates: np.ndarray, min_k: int = 2, max_k: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame, int | None]:
    """Calculate notebook-style WCSS and silhouette diagnostics."""
    sample_count = len(coordinates)
    upper_k = min(max_k, sample_count - 1)
    if upper_k < min_k:
        raise ValueError("There are not enough customers to evaluate cluster counts.")

    wcss_rows = []
    for k in range(1, upper_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(coordinates)
        wcss_rows.append({"clusters": k, "wcss": model.inertia_})

    wcss = pd.DataFrame(wcss_rows)
    knee = KneeLocator(
        wcss["clusters"],
        wcss["wcss"],
        curve="convex",
        direction="decreasing",
    )
    optimal_k = int(knee.elbow) if knee.elbow is not None else None

    silhouette_rows = []
    for k in range(min_k, upper_k + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(
            coordinates
        )
        silhouette_rows.append(
            {"clusters": k, "silhouette_score": silhouette_score(coordinates, labels)}
        )
    return wcss, pd.DataFrame(silhouette_rows), optimal_k


def fit_clustering(
    prepared: PreparedData,
    algorithm: str,
    n_clusters: int,
    wcss: pd.DataFrame | None = None,
    silhouette: pd.DataFrame | None = None,
    optimal_k: int | None = None,
) -> ClusteringResult:
    """Fit the selected clustering algorithm and create business profiles."""
    if n_clusters < 2:
        raise ValueError("The number of clusters must be at least 2.")
    if n_clusters > len(prepared.pca_coordinates):
        raise ValueError("The number of clusters cannot exceed the customer count.")

    if algorithm == "K-Means":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif algorithm == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    else:
        raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

    labels = model.fit_predict(prepared.pca_coordinates)
    labels = canonicalize_labels(labels, prepared.customer_data, n_clusters)
    customers = prepared.customer_data.copy()
    customers["Cluster"] = labels
    profile = customers.groupby("Cluster", sort=True).mean(numeric_only=True)

    if wcss is None or silhouette is None:
        wcss, silhouette, optimal_k = evaluate_cluster_counts(
            prepared.pca_coordinates
        )

    return ClusteringResult(
        algorithm=algorithm,
        n_clusters=n_clusters,
        labels=labels,
        customers=customers,
        profile=profile,
        wcss=wcss,
        silhouette=silhouette,
        optimal_k=optimal_k,
        model=model,
        training_coordinates=prepared.pca_coordinates,
    )


def canonicalize_labels(
    labels: np.ndarray, customers: pd.DataFrame, n_clusters: int
) -> np.ndarray:
    """Keep the notebook's four business meanings stable across model refits."""
    if n_clusters != 4:
        return labels
    labeled = customers[["Living_status", "Total_Spending"]].copy()
    labeled["label"] = labels
    means = labeled.groupby("label").agg(
        spending=("Total_Spending", "mean"),
        partner_rate=("Living_status", lambda values: (values == "Partner").mean()),
    )
    partner_labels = means.sort_values(["partner_rate", "spending"]).index.tolist()
    partner_labels = sorted(
        partner_labels,
        key=lambda label: (
            0 if means.loc[label, "partner_rate"] >= 0.5 else 1,
            means.loc[label, "spending"],
        ),
    )
    partner = [label for label in partner_labels if means.loc[label, "partner_rate"] >= 0.5]
    alone = [label for label in partner_labels if means.loc[label, "partner_rate"] < 0.5]
    if len(partner) != 2 or len(alone) != 2:
        return labels
    mapping = {partner[0]: 0, partner[1]: 1, alone[0]: 2, alone[1]: 3}
    return np.array([mapping[label] for label in labels])


def assign_new_customer(
    record: pd.DataFrame,
    prepared: PreparedData,
    result: ClusteringResult,
) -> tuple[int, pd.DataFrame, np.ndarray]:
    """Assign a new record using the fitted model or Agglomerative centroids."""
    engineered, _, coordinates = transform_new_customer(record, prepared)
    if isinstance(result.model, KMeans):
        raw_label = int(result.model.predict(coordinates)[0])
        labels = canonicalize_labels(
            result.labels,
            prepared.customer_data,
            result.n_clusters,
        )
        source_labels = result.model.labels_
        mapping = {}
        for source, canonical in zip(source_labels, labels):
            mapping.setdefault(int(source), int(canonical))
        cluster = mapping[raw_label]
    else:
        training_labels = result.model.labels_
        centroids = np.vstack(
            [
                result.training_coordinates[training_labels == label].mean(axis=0)
                for label in range(result.n_clusters)
            ]
        )
        raw_label = int(np.argmin(((coordinates[0] - centroids) ** 2).sum(axis=1)))
        canonical_labels = canonicalize_labels(
            training_labels, prepared.customer_data, result.n_clusters
        )
        cluster = int(canonical_labels[training_labels == raw_label][0])
    engineered["Cluster"] = cluster
    return cluster, engineered, coordinates


def add_registered_customers(
    result: ClusteringResult,
    prepared: PreparedData,
    registered: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Append persisted registrations to dashboard data without retraining."""
    if registered.empty:
        return result.customers.copy(), result.training_coordinates
    engineered, _, coordinates = transform_new_customer(registered, prepared)
    if "Cluster" not in registered.columns:
        raise ValueError("Registered data must contain a persisted Cluster column.")
    engineered["Cluster"] = registered["Cluster"].astype(int).to_numpy()
    return (
        pd.concat([result.customers, engineered], ignore_index=True),
        np.vstack([result.training_coordinates, coordinates]),
    )
