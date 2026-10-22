# DataCo Supply Chain dataset

Required:
- `DataCoSupplyChainDataset.csv`: main supply-chain/order-item data. The backend creates a sanitized DuckDB table named `dataco_orders`.
- `DescriptionDataCoSupplyChain.csv`: field definitions and dataset documentation.

Optional local-only file:
- `tokenized_access_logs.csv`: click/access events. When present locally, the backend ingests only Product, Category, Date, Month, Hour and Department. The IP and URL fields are deliberately not materialized into the agent-queryable table or returned to the model. Review/anonymize this file before any public sharing.

The read-only agent supports SQL analytics, order lookup, country delivery statistics, shipping-mode comparisons and aggregate access-log analytics. The country flow map is available at `/map?dataset=dataco` with API endpoints under `/api/dataco/map/*`.

**Important data limitation:** the route map uses `Customer Country → Order Country` as a proxy based on dataset fields. It does not represent a verified warehouse/seller origin or a validated physical shipping route. `Late_delivery_risk` is an existing dataset label, not an ML prediction. The DataCo path does not reuse Olist-trained models or the Olist Brazil state map.
