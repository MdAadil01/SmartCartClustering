from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .clustering import CLUSTER_COLORS


def _color_map() -> dict[str, str]:
    return {str(cluster): color for cluster, color in CLUSTER_COLORS.items()}

def pca_figure(customers: pd.DataFrame, coordinates, title: str) -> go.Figure:
    frame = pd.DataFrame(
        {
            "PCA 1": coordinates[:, 0],
            "PCA 2": coordinates[:, 1],
            "PCA 3": coordinates[:, 2],
            "Cluster": customers["Cluster"].astype(str).to_numpy(),
        }
    )
    if "ID" in customers:
        frame["Customer ID"] = customers["ID"].to_numpy()
    return px.scatter_3d(
        frame,
        x="PCA 1",
        y="PCA 2",
        z="PCA 3",
        color="Cluster",
        hover_data=["Customer ID"] if "Customer ID" in frame else None,
        color_discrete_map=_color_map(),
        title=title,
    )


def income_spending_figure(customers: pd.DataFrame) -> go.Figure:
    return px.scatter(
        customers,
        x="Total_Spending",
        y="Income",
        color=customers["Cluster"].astype(str),
        hover_data=[column for column in ["ID", "Age", "Education"] if column in customers],
        labels={
            "color": "Cluster",
            "Total_Spending": "Total spending (INR)",
            "Income": "Income (INR)",
        },
        color_discrete_map=_color_map(),
        title="Income versus total spending (INR)",
    )


def cluster_distribution_figure(customers: pd.DataFrame) -> go.Figure:
    counts = customers["Cluster"].value_counts().sort_index().rename_axis("Cluster")
    counts = counts.reset_index(name="Customers")
    counts["Cluster"] = counts["Cluster"].astype(str)
    return px.bar(
        counts,
        x="Cluster",
        y="Customers",
        color="Cluster",
        color_discrete_map=_color_map(),
        title="Customers per cluster",
    )


def evaluation_figure(wcss: pd.DataFrame, silhouette: pd.DataFrame) -> go.Figure:
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=wcss["clusters"],
            y=wcss["wcss"],
            mode="lines+markers",
            name="WCSS",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=silhouette["clusters"],
            y=silhouette["silhouette_score"],
            mode="lines+markers",
            name="Silhouette score",
        ),
        secondary_y=True,
    )
    figure.update_xaxes(title_text="Number of clusters")
    figure.update_yaxes(title_text="WCSS", secondary_y=False)
    figure.update_yaxes(title_text="Silhouette score", secondary_y=True)
    figure.update_layout(title="Cluster-count evaluation")
    return figure


def profile_figure(profile: pd.DataFrame) -> go.Figure:
    metrics = ["Income", "Total_Spending", "Age", "Total_Children", "Response"]
    available = [metric for metric in metrics if metric in profile.columns]
    normalized = profile[available].copy()
    for column in available:
        minimum, maximum = normalized[column].min(), normalized[column].max()
        normalized[column] = (
            (normalized[column] - minimum) / (maximum - minimum)
            if maximum != minimum
            else 0
        )
    long_profile = normalized.reset_index().melt(id_vars="Cluster")
    long_profile["Cluster"] = long_profile["Cluster"].astype(str)
    return px.line(
        long_profile,
        x="variable",
        y="value",
        color="Cluster",
        color_discrete_map=_color_map(),
        markers=True,
        labels={"variable": "Metric", "value": "Relative cluster mean"},
        title="Relative cluster profile",
    )
