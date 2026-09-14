# AI Logistics Agent — Olist E-commerce Dataset

The project builds an **AI Agent for logistics operations** on an e-commerce platform, running as a **local web app**, combining:

1. **Machine Learning models** for delivery-risk prediction, trained on real-world data from the Olist Brazilian E-commerce Dataset (~99K orders) from Kaggle.
2. A **conversational Agent using the Gemini API (Google GenAI, function calling)** — users can ask questions in natural language through the web interface, and the agent automatically calls the appropriate tool (ML model or statistical query) to generate an answer.

## 1. Problem & Dataset

The original dataset consists of 9 CSV tables (orders, order_items, payments, reviews, products, customers, sellers, geolocation, and product_category_translation). After merging and processing, the result is **`processed_dataset.csv`** (99,442 rows, 41 columns) — an order-level feature table containing:

* Seller → customer distance (Haversine distance based on average ZIP-code coordinates): `distance_km`, `cust_lat/lng`, `seller_lat/lng`
* Product volume/weight, number of items, total value/freight/payment amounts
* Order-time features: `purchase_dow`, `purchase_month`, `purchase_hour`
* **Labels:**

  * `is_late`: whether the order was delivered later than the estimated delivery date (`order_delivered_customer_date > order_estimated_delivery_date`)
  * `actual_delivery_days`: number of days from purchase to customer delivery

All features used for training consist only of information **available at the time of order placement**, simulating a real-world scenario of evaluating delivery risk *before* fulfillment.

`processed_dataset.parquet` is a parallel version of the same dataset (faster to read). `product_category_name_translation.csv` contains the Portuguese → English product-category translations used during the original data processing step.

## 2. ML Models (XGBoost)

| **Model**                  | **File**                          | **Objective**                                     | **Test Set Results**                                                                    |
| -------------------------- | --------------------------------- | ------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `late_delivery_classifier` | `late_delivery_classifier.joblib` | Late-delivery probability (binary classification) | ROC-AUC **0.788**, Accuracy **0.762**, Precision **0.20**, Recall **0.67**, F1 **0.31** |
| `delivery_days_regressor`  | `delivery_days_regressor.joblib`  | Actual delivery time in days                      | MAE **4.74 days**, RMSE **7.58 days**                                                   |

Both models are implemented as `sklearn.Pipeline` objects (ColumnTransformer preprocessing + `XGBClassifier`/`XGBRegressor`) and trained on 96,470 delivered orders using an 80/20 train/test split.

Because the data is imbalanced (only ~8% of orders are late), the classifier uses `scale_pos_weight` to prioritize higher recall over precision — prioritizing the detection of potentially late orders while accepting more false alarms.

Detailed feature lists and metrics are available in `model_metadata.json`.

## 3. Agent Architecture (Target)

```text
User (natural-language Vietnamese questions through the local web UI)
        │
        ▼
   Gemini API (Google GenAI SDK, function calling) ──► selects the appropriate tool
        │
        ▼
   tools.py: predict_new_order / get_order_info /
             get_state_stats / get_seller_stats /
             get_category_stats / top_risky_states /
             top_risky_categories
        │
        ▼
   Tool returns JSON → Gemini summarizes the result in Vietnamese → displayed in the web UI
```

The backend is planned to use Flask to serve the web UI and expose the tools through HTTP APIs.

## 4. Current Repository Status

The repository currently contains:

```text
.
├── processed_dataset.csv                    # processed dataset (99,442 rows)
├── processed_dataset.parquet                # Parquet version of the dataset
├── product_category_name_translation.csv    # Portuguese → English category translation
├── late_delivery_classifier.joblib          # trained classifier
├── delivery_days_regressor.joblib            # trained regressor
├── model_metadata.json                       # complete feature list + metrics
├── test_model.py                             # quick classifier test script
├── test_tools.py                             # quick test for tools.predict_new_order
├── requirements.txt
├── README.md
└── README_DATA.md
```

**Not yet available in the repository (to be implemented):** `src/data_pipeline.py`, `src/train_models.py`, `src/tools.py`, `src/agent.py`, and `src/api.py` — covering data processing, model training, business-logic tools, Gemini Agent integration, and the web/API server.

`test_model.py` and `test_tools.py` currently import these modules (`models/late_delivery_classifier.joblib`, `from src import tools`), so they will not run until the required source files are implemented.

## 5. How to Run (Once Source Code Is Complete)

```bash
pip install -r requirements.txt

# 1) Build the processed dataset (if rebuilding from the original data — see README_DATA.md)
python3 -m src.data_pipeline

# 2) Train the models
python3 -m src.train_models

# 3) Run the web app (Flask) — agent uses Gemini API
export GEMINI_API_KEY="..."
python3 -m src.api   # http://localhost:5000
```

### API Examples

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"distance_km": 1200, "customer_state": "BA", "seller_state": "SP"}'

curl http://localhost:5000/stats/top-risky-states?n=5
```

### Example Agent Conversation (Web UI)

```text
User: An order from a seller in SP to a customer in AL is about 2500 km away. Is there a risk of late delivery?

Agent: [calls predict_new_order] → Late-delivery probability ~XX%, estimated delivery time ~YY days.
       Recommendation: AL has historically had a high late-delivery rate, so consider
       choosing a seller closer to the customer or providing a longer estimated
       delivery time to the customer.
```

## 6. Future Improvements

* Implement `src/tools.py`, `src/agent.py` (Gemini function calling), `src/api.py` (Flask), and the local web interface to complete the agent workflow described above.
* Improve the precision/recall trade-off through threshold tuning or cost-sensitive learning based on the actual cost of late deliveries versus false alarms.
* Add a `recommend_seller` tool to select sellers with strong on-time delivery performance for a given region.
* Add long-term conversation memory / prediction logging for periodic retraining and model-drift monitoring.
* Visualize delivery-risk maps by state using Folium/Plotly.

---

*Dataset: [Brazilian E-Commerce Public Dataset by Olist (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)*
