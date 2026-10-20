# DataCo Supply Chain dataset

Place the dataset files you maintain in this folder. The application reads the main CSV at:

- `DataCoSupplyChainDataset.csv` (required for enabling DataCo in the dataset selector)
- `DescriptionDataCoSupplyChain.csv` (documentation/reference)
- `tokenized_access_logs.csv` (optional; not currently used by the chat agent)

Do not commit credentials or unmasked personal information. The access-log file contains IP addresses; only add it if needed and after considering whether to anonymize/remove IPs. The agent currently uses only the main supply-chain CSV.

The DataCo adapter is read-only and exposes a DuckDB table named `dataco_orders`. It supports SQL analytics, order lookup and regional late-risk summaries. It does not yet reuse the Olist ML models or Olist-specific flow map.
