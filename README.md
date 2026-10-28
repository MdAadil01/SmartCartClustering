# SmartCart Customer Segmentation

SmartCart is a Streamlit dashboard for exploring and segmenting the customers in
`smartcart_customers.csv`. It preserves the notebook's data preparation, feature
engineering, PCA reduction, K-Means evaluation, and Agglomerative/K-Means
clustering workflow while adding interactive exploration and downloads.

## Features

- Overview KPIs and cluster distribution
- Interactive 3D PCA segmentation view
- K-Means or Agglomerative clustering
- Configurable cluster count
- Income, spending, age, education, and cluster filters
- Cluster profiles and business insights
- Elbow/WCSS and silhouette evaluation charts
- Customer and cleaned-data CSV downloads
- Four-cluster prediction and optional customer registration
- Income and spending displayed in Indian rupees (INR)
- SQLite-backed customer storage and database status monitoring

## Run locally

From the project directory, create or activate a virtual environment and install
the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Then start Streamlit:

```powershell
streamlit run app.py
```

The app expects `smartcart_customers.csv` in the same directory as `app.py`.
On first startup it creates `smartcart.db`, creates the `customers` table, and
seeds the filtered original records once. If the legacy
`smartcart_customers_updated.csv` exists, its registered rows are migrated during
that first seed. Later reruns read dashboard customer data from SQLite and do not
reimport the CSV.

## Customer prediction and registration

The **Customer Prediction & Registration** section transforms a single form
submission with the fitted training encoder, scaler, and PCA model. K-Means uses
the fitted model's `predict` method. Agglomerative clustering has no native
unseen-record prediction, so the app assigns a new record to the nearest
training cluster centroid in PCA space; this is shown as a nearest-centroid
assignment rather than a retrained Agglomerative model.

Registrations are written to `smartcart.db` in a transaction. The original CSV is
never overwritten. On restart, the database remains the dashboard source of truth
while the clustering model remains trained on the original customers. A duplicate
raw submission is rejected and IDs are generated above the maximum database ID.

For four clusters, labels are stabilized from the measured training profiles:
0 is the partner low/moderate-spending segment, 1 is the partner high-spending
segment, 2 is the alone low-spending segment, and 3 is the alone high-spending
segment. Their colors are red, blue, yellow, and green respectively.

The database stores all original CSV fields, engineered age, tenure, total
spending, total children, living status, the assigned cluster, UTC creation time,
and a `Source` value of `original` or `registered`. New K-Means customers use the
fitted model's `predict` method. Agglomerative customers use the documented
nearest training-centroid assignment in PCA space. Registrations never retrain
the model implicitly.

## Notebook compatibility notes

The app retains the notebook's `2026 - Year_Birth` age calculation, median
income imputation, date-based tenure, spending and children totals, grouped
education/living-status categories, age/income filters, one-hot encoding,
standardization, three-component PCA, and random-state-42 clustering. Customer
identifiers and raw source fields dropped by the notebook are not used in the
model feature matrix.
