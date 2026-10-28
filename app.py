from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import streamlit as st

from src.clustering import (
    CLUSTER_COLORS,
    CLUSTER_INTERPRETATIONS,
    ClusteringResult,
    assign_new_customer,
    evaluate_cluster_counts,
    fit_clustering,
)
from src.database import (
    get_all_customers,
    get_database_status,
    init_db,
    insert_customer,
    seed_database_from_csv,
)
from src.data_preprocessing import (
    load_customer_data,
    prepare_data,
    transform_new_customer,
)
from src.registration import (
    REGISTERED_COLUMNS,
    build_customer_record,
    validate_customer_record,
)
from src.visualizations import (
    cluster_distribution_figure,
    evaluation_figure,
    income_spending_figure,
    pca_figure,
    profile_figure,
)


st.set_page_config(
    page_title="SmartCart Customer Segmentation",
    page_icon="🛒",
    layout="wide",
)

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "smartcart_customers.csv"
UPDATED_PATH = ROOT / "smartcart_customers_updated.csv"
DATABASE_PATH = ROOT / "smartcart.db"


def format_inr(value: float | int) -> str:
    """Format a numeric value as Indian rupees with lakh/crore grouping."""
    amount = f"{float(value):.0f}"
    sign = "-" if amount.startswith("-") else ""
    digits = amount.lstrip("-")
    if len(digits) <= 3:
        grouped = digits
    else:
        last_three = digits[-3:]
        leading = digits[:-3]
        groups = []
        while leading:
            groups.insert(0, leading[-2:])
            leading = leading[:-2]
        grouped = ",".join(groups + [last_three])
    return f"{sign}₹{grouped}"


@st.cache_data(show_spinner=False)
def get_prepared_data(path: str):
    return prepare_data(load_customer_data(path))


@st.cache_data(show_spinner=False)
def get_evaluation(coordinates):
    return evaluate_cluster_counts(coordinates)


@st.cache_data(show_spinner=False)
def get_database_customers(path: str, database_revision: int) -> pd.DataFrame:
    return get_all_customers(path)


def csv_download(frame: pd.DataFrame, label: str, filename: str) -> None:
    st.download_button(
        label=label,
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def cluster_style(value: object) -> str:
    try:
        color = CLUSTER_COLORS[int(value)]
    except (KeyError, TypeError, ValueError):
        return ""
    return f"background-color: {color}; color: {'black' if int(value) == 2 else 'white'}"


def displayed_result(
    result: ClusteringResult,
    prepared,
    database_customers: pd.DataFrame,
) -> tuple[ClusteringResult, pd.DataFrame]:
    customers = database_customers.copy()
    customers["Is_Registered"] = customers["Source"].eq("registered")
    _, _, coordinates = transform_new_customer(
        customers[REGISTERED_COLUMNS], prepared
    )
    profile = customers.groupby("Cluster", sort=True).mean(numeric_only=True)
    return replace(result, customers=customers, profile=profile), coordinates


def render_filters(customers: pd.DataFrame, key_prefix: str):
    with st.sidebar:
        st.subheader("Customer filters")
        clusters = sorted(customers["Cluster"].unique())
        selected_clusters = st.multiselect(
            "Cluster",
            clusters,
            default=clusters,
            key=f"{key_prefix}_clusters",
        )
        age_range = st.slider(
            "Age",
            int(customers["Age"].min()),
            int(customers["Age"].max()),
            (int(customers["Age"].min()), int(customers["Age"].max())),
            key=f"{key_prefix}_age",
        )
        income_range = st.slider(
            "Income",
            float(customers["Income"].min()),
            float(customers["Income"].max()),
            (float(customers["Income"].min()), float(customers["Income"].max())),
            key=f"{key_prefix}_income",
        )
        spending_range = st.slider(
            "Total spending",
            float(customers["Total_Spending"].min()),
            float(customers["Total_Spending"].max()),
            (
                float(customers["Total_Spending"].min()),
                float(customers["Total_Spending"].max()),
            ),
            key=f"{key_prefix}_spending",
        )
    return customers[
        customers["Cluster"].isin(selected_clusters)
        & customers["Age"].between(*age_range)
        & customers["Income"].between(*income_range)
        & customers["Total_Spending"].between(*spending_range)
    ].copy()


def render_overview(result: ClusteringResult) -> None:
    st.header("Overview")
    customers = result.customers
    columns = st.columns(4)
    columns[0].metric("Customers in working dataset", f"{len(customers):,}")
    columns[1].metric("Average income", format_inr(customers["Income"].mean()))
    columns[2].metric("Average spending", format_inr(customers["Total_Spending"].mean()))
    columns[3].metric("Registered customers", f"{int(customers.get('Is_Registered', pd.Series(dtype=bool)).sum()):,}")
    distribution = (
        customers.groupby("Cluster")
        .agg(
            Customers=("Cluster", "size"),
            Percentage=("Cluster", lambda values: len(values) / len(customers)),
            Average_Income=("Income", "mean"),
            Average_Spending=("Total_Spending", "mean"),
        )
        .reset_index()
    )
    st.plotly_chart(cluster_distribution_figure(customers), use_container_width=True)
    st.dataframe(
        distribution.style.format(
            {
                "Percentage": "{:.1%}",
                "Average_Income": format_inr,
                "Average_Spending": format_inr,
            }
        ).map(cluster_style, subset=["Cluster"]),
        use_container_width=True,
        hide_index=True,
    )
    st.plotly_chart(income_spending_figure(customers), use_container_width=True)


def render_segmentation(result: ClusteringResult, coordinates) -> None:
    st.header("Customer Segmentation")
    st.caption(
        f"{result.algorithm} with {result.n_clusters} clusters on the three-component PCA representation."
    )
    st.plotly_chart(
        pca_figure(result.customers, coordinates, "3D PCA customer segments"),
        use_container_width=True,
    )
    st.plotly_chart(income_spending_figure(result.customers), use_container_width=True)


def render_insights(result: ClusteringResult) -> None:
    st.header("Cluster Insights")
    st.plotly_chart(profile_figure(result.profile), use_container_width=True)
    profile = result.profile.reset_index()
    profile.insert(1, "Interpretation", [CLUSTER_INTERPRETATIONS.get(int(i), "Configured segment") for i in profile["Cluster"]])
    st.dataframe(
        profile.style.format("{:,.2f}", subset=profile.select_dtypes("number").columns)
        .map(cluster_style, subset=["Cluster"]),
        use_container_width=True,
    )
    highest_spending = result.profile["Total_Spending"].idxmax()
    highest_response = result.profile["Response"].idxmax()
    st.info(
        f"Cluster {highest_spending} has the highest average spending "
        f"({format_inr(result.profile.loc[highest_spending, 'Total_Spending'])}). "
        f"Cluster {highest_response} has the highest campaign response "
        f"({result.profile.loc[highest_response, 'Response']:.1%})."
    )


def render_explorer(result: ClusteringResult) -> None:
    st.header("Customer Explorer")
    customers = result.customers.copy()
    with st.sidebar:
        search_id = st.text_input("Search by customer ID", key="explorer_search_id").strip()
        selected_clusters = st.multiselect(
            "Explorer clusters",
            sorted(customers["Cluster"].unique()),
            default=sorted(customers["Cluster"].unique()),
            key="explorer_cluster_filter",
        )
        income = st.slider(
            "Explorer income",
            float(customers["Income"].min()),
            float(customers["Income"].max()),
            (float(customers["Income"].min()), float(customers["Income"].max())),
            key="explorer_income",
        )
        spending = st.slider(
            "Explorer spending",
            float(customers["Total_Spending"].min()),
            float(customers["Total_Spending"].max()),
            (float(customers["Total_Spending"].min()), float(customers["Total_Spending"].max())),
            key="explorer_spending",
        )
    if search_id:
        customers = customers[customers["ID"].astype(str).str.contains(search_id, na=False)]
    customers = customers[
        customers["Cluster"].isin(selected_clusters)
        & customers["Income"].between(*income)
        & customers["Total_Spending"].between(*spending)
    ].copy()
    customers["Cluster Label"] = customers["Cluster"].map(
        lambda value: f"Cluster {int(value)}"
    )
    if "Is_Registered" not in customers:
        customers["Is_Registered"] = False
    display_columns = [
        "ID",
        "Cluster",
        "Cluster Label",
        "Is_Registered",
        "Age",
        "Education",
        "Living_status",
        "Income",
        "Total_Spending",
        "Total_Children",
        "Recency",
        "Response",
    ]
    st.write(f"{len(customers):,} customers match the selected filters.")
    st.dataframe(
        customers[display_columns].style.map(cluster_style, subset=["Cluster"]),
        use_container_width=True,
        hide_index=True,
    )
    csv_download(customers, "Download filtered customer records", "smartcart_filtered_customers.csv")


def render_evaluation(result: ClusteringResult) -> None:
    st.header("Model Evaluation")
    st.metric("Detected elbow", result.optimal_k if result.optimal_k else "Not detected")
    st.plotly_chart(evaluation_figure(result.wcss, result.silhouette), use_container_width=True)
    best = result.silhouette.loc[result.silhouette["silhouette_score"].idxmax()]
    st.success(
        f"Best tested silhouette score: {best['silhouette_score']:.3f} at "
        f"{int(best['clusters'])} clusters."
    )


def render_data_explorer(prepared, customers: pd.DataFrame, status: dict) -> None:
    st.header("Data Explorer")
    st.write(
        f"Original rows: {len(prepared.raw):,}. "
        f"Rows after notebook filters: {len(prepared.cleaned):,}. "
        f"Database rows: {status['total']:,}."
    )
    st.subheader("Database status")
    status_columns = st.columns(4)
    status_columns[0].metric("Connection", "Connected")
    status_columns[1].metric("Database records", f"{status['total']:,}")
    status_columns[2].metric("New registrations", f"{status['registered']:,}")
    status_columns[3].metric(
        "Latest registration",
        status["latest"][:19].replace("T", " ") if status["latest"] else "None",
    )
    st.dataframe(status["counts"], use_container_width=True, hide_index=True)
    st.subheader("Customers from SQLite")
    st.dataframe(customers.head(100), use_container_width=True)
    csv_download(customers, "Download database customer data", "smartcart_database_customers.csv")
    st.subheader("Missing values in original CSV")
    st.dataframe(
        prepared.raw.isna().sum().rename("missing_values").to_frame(),
        use_container_width=True,
    )


def render_prediction_page(prepared, result: ClusteringResult) -> None:
    st.header("Customer Prediction & Registration")
    st.caption(
        "The form uses the original dataset columns and the fitted preprocessing pipeline. "
        "Registration is optional and never changes the original CSV."
    )
    if result.n_clusters != 4:
        st.warning("Prediction and registration require the four-cluster model. Set Number of clusters to 4.")
        return
    education_options = ["Basic", "2n Cycle", "Graduation", "Master", "PhD"]
    marital_options = ["Absurd", "Alone", "Divorced", "Married", "Single", "Together", "Widow", "YOLO"]
    with st.form("customer_prediction_form"):
        st.subheader("Customer information")
        first, second, third = st.columns(3)
        year_birth = first.number_input("Year of birth", 1900, 2026, 1980)
        education = first.selectbox("Education", education_options, index=2)
        marital_status = first.selectbox("Marital status", marital_options, index=3)
        income = second.number_input("Income (INR)", min_value=0.0, value=50000.0, step=1000.0)
        kidhome = second.number_input("Children at home", min_value=0, max_value=10, value=0, step=1)
        teenhome = second.number_input("Teenagers at home", min_value=0, max_value=10, value=0, step=1)
        enrollment = third.date_input("Customer enrollment date", value=pd.Timestamp("2013-01-01"))
        recency = third.number_input("Days since last purchase", min_value=0, max_value=3650, value=30)
        complain = third.selectbox("Customer complaint", [0, 1], format_func=lambda value: "No" if value == 0 else "Yes")
        st.subheader("Purchases and campaign fields")
        purchase_columns = st.columns(3)
        wines = purchase_columns[0].number_input("Wines", min_value=0, value=100, step=10)
        fruits = purchase_columns[0].number_input("Fruits", min_value=0, value=20, step=10)
        meat = purchase_columns[0].number_input("Meat products", min_value=0, value=100, step=10)
        fish = purchase_columns[1].number_input("Fish products", min_value=0, value=20, step=10)
        sweets = purchase_columns[1].number_input("Sweet products", min_value=0, value=20, step=10)
        gold = purchase_columns[1].number_input("Gold products", min_value=0, value=20, step=10)
        deals = purchase_columns[2].number_input("Deal purchases", min_value=0, value=2, step=1)
        web = purchase_columns[2].number_input("Web purchases", min_value=0, value=4, step=1)
        catalog = purchase_columns[2].number_input("Catalog purchases", min_value=0, value=2, step=1)
        store = purchase_columns[2].number_input("Store purchases", min_value=0, value=4, step=1)
        visits = st.number_input("Web visits per month", min_value=0, value=5, step=1)
        response = st.selectbox("Campaign accepted", [0, 1], format_func=lambda value: "No" if value == 0 else "Yes")
        submitted = st.form_submit_button("Predict Customer Cluster")

    if submitted:
        values = {
            "Year_Birth": year_birth, "Education": education, "Marital_Status": marital_status,
            "Income": income, "Kidhome": kidhome, "Teenhome": teenhome,
            "Dt_Customer": enrollment, "Recency": recency, "MntWines": wines,
            "MntFruits": fruits, "MntMeatProducts": meat, "MntFishProducts": fish,
            "MntSweetProducts": sweets, "MntGoldProds": gold, "NumDealsPurchases": deals,
            "NumWebPurchases": web, "NumCatalogPurchases": catalog,
            "NumStorePurchases": store, "NumWebVisitsMonth": visits,
            "Complain": complain, "Response": response,
        }
        record = build_customer_record(values)
        errors = validate_customer_record(record)
        if errors:
            for error in errors:
                st.error(error)
        else:
            cluster, engineered, coordinates = assign_new_customer(record, prepared, result)
            st.session_state["pending_record"] = record
            st.session_state["pending_cluster"] = cluster
            st.session_state["pending_engineered"] = engineered
            st.session_state["pending_coordinates"] = coordinates
            st.session_state["pending_algorithm"] = result.algorithm

    if "pending_record" not in st.session_state:
        return
    cluster = int(st.session_state["pending_cluster"])
    record = st.session_state["pending_record"]
    total_spending = int(record.iloc[0][["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]].sum())
    color = CLUSTER_COLORS[cluster]
    st.markdown(
        f'<div style="border-left: 12px solid {color}; padding: 1rem; border-radius: .5rem; '
        'color: #000000; '
        f'background: #f5f5f5"><h3>Cluster {cluster}</h3>'
        f'<p>{CLUSTER_INTERPRETATIONS[cluster]}</p></div>',
        unsafe_allow_html=True,
    )
    metrics = st.columns(3)
    metrics[0].metric("Predicted cluster", cluster)
    metrics[1].metric("Income", format_inr(record.iloc[0]["Income"]))
    metrics[2].metric("Total spending", format_inr(total_spending))
    profile = result.profile.loc[cluster]
    st.write(
        f"This assignment is consistent with the training profile: the cluster averages "
        f"{format_inr(profile['Income'])} income and "
        f"{format_inr(profile['Total_Spending'])} spending."
    )
    if st.button("Add Customer to Dataset", type="primary"):
        try:
            customer_id = insert_customer(
                DATABASE_PATH,
                record,
                prepared,
                cluster,
            )
            st.session_state["registration_message"] = (
                f"Customer {customer_id} was saved to SQLite."
            )
            st.cache_data.clear()
            st.session_state.pop("pending_record", None)
            st.session_state.pop("pending_cluster", None)
            st.session_state.pop("pending_engineered", None)
            st.session_state.pop("pending_coordinates", None)
            st.rerun()
        except (OSError, ValueError, RuntimeError) as error:
            st.error(f"Could not save the customer: {error}")


def main() -> None:
    st.title("🛒 SmartCart")
    st.caption("E-commerce customer segmentation and business insights")
    try:
        prepared = get_prepared_data(str(DATA_PATH))
        wcss, silhouette, optimal_k = get_evaluation(prepared.pca_coordinates)
        seed_model = fit_clustering(
            prepared,
            "Agglomerative",
            4,
            wcss=wcss,
            silhouette=silhouette,
            optimal_k=optimal_k,
        )
        init_db(DATABASE_PATH)
        seed_database_from_csv(
            DATABASE_PATH,
            DATA_PATH,
            prepared,
            seed_model.labels.tolist(),
            registered_csv_path=UPDATED_PATH,
        )
        database_revision = DATABASE_PATH.stat().st_mtime_ns
        database_customers = get_database_customers(
            str(DATABASE_PATH),
            database_revision,
        )
        database_status = get_database_status(DATABASE_PATH)
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as error:
        st.error(f"Unable to prepare the SmartCart data: {error}")
        st.stop()

    if "registration_message" in st.session_state:
        st.success(st.session_state.pop("registration_message"))

    with st.sidebar:
        st.header("Model controls")
        algorithm = st.selectbox("Clustering algorithm", ["Agglomerative", "K-Means"])
        n_clusters = st.slider("Number of clusters", 2, 10, 4)
        section = st.radio(
            "Dashboard section",
            [
                "Overview",
                "Customer Segmentation",
                "Cluster Insights",
                "Customer Explorer",
                "Model Evaluation",
                "Data Explorer",
                "Customer Prediction & Registration",
            ],
        )

    try:
        result = fit_clustering(
            prepared, algorithm, n_clusters, wcss=wcss, silhouette=silhouette, optimal_k=optimal_k
        )
        working_result, coordinates = displayed_result(
            result,
            prepared,
            database_customers,
        )
    except ValueError as error:
        st.error(f"Unable to fit the selected clustering model: {error}")
        st.stop()

    if section == "Overview":
        render_overview(working_result)
    elif section == "Customer Segmentation":
        render_segmentation(working_result, coordinates)
    elif section == "Cluster Insights":
        render_insights(working_result)
    elif section == "Customer Explorer":
        render_explorer(working_result)
    elif section == "Model Evaluation":
        render_evaluation(working_result)
    elif section == "Data Explorer":
        render_data_explorer(prepared, database_customers, database_status)
    else:
        render_prediction_page(prepared, result)


if __name__ == "__main__":
    main()
