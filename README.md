# AI Logistics Agent — Olist E-commerce Dataset

A local web app combining **Machine Learning** (delivery risk prediction) and a **multi-model conversational AI Agent** (Gemini / GPT-4o / Grok — automatically switching models when a provider runs out of quota) to support logistics operations for an e-commerce platform, built on the **Olist Brazilian E-commerce Dataset** (~99K orders from Kaggle).

## 1. Key Features

* **Predict delivery delay risk + delivery time** for a new order (before shipment)
* **Vietnamese conversational Q&A** through the web interface, with the agent automatically calling the appropriate tool to answer questions
* **Flexible model selection** directly from the interface: Gemini Flash/Flash-Lite/2.0, GPT-4o/mini, Grok 4 Fast/4.6 — switch models when one provider runs out of quota without losing conversation context
* **Automatically generate SQL** to answer analytical questions without a dedicated tool (e.g. comparisons by month, payment method, product weight...)
* **Train temporary models on demand** for targets/subsets other than the two main models (e.g. predicting review_score or training specifically for one state)
* **Dockerized** — run with a single command without manually installing Python/dependencies

## 2. Architecture

```text
Browser (web interface, templates/index.html)
        │  model selection via dropdown
        ▼
Flask API (src/api.py)  ──►  src/providers.py (multi-model abstraction layer)
        │                         │
        │                         ├─ Gemini (google-genai)
        │                         ├─ OpenAI (GPT-4o / GPT-4o mini)
        │                         └─ xAI/Grok (OpenAI-compatible endpoint)
        │
        ▼
   src/tools.py — business tools available to the agent:
     predict_new_order · get_order_info · get_state_stats · get_seller_stats
     get_category_stats · top_risky_states · top_risky_categories
     describe_dataset · query_dataset_sql (DuckDB, SELECT only) · train_custom_model
        │
        ▼
   data/processed_dataset.csv  +  models/*.joblib (XGBoost)
```

**Conversation history is stored in a "canonical" format** (text only, independent of the provider) in `src/providers.py`, so switching models in the middle of a conversation does not break the conversation format. The trade-off is that the new model does not remember internal tool calls from previous turns and only receives the conversation content in text form.

## 3. Problem & Data

The original dataset contains 9 CSV tables (orders, order_items, payments, reviews, products, customers, sellers, geolocation, product_category_translation), which are merged into **`data/processed_dataset.csv`** (99,442 rows, 41 columns) through `src/data_pipeline.py`:

* Seller → customer distance (haversine distance based on average coordinates for zip codes)
* Product volume/weight, item count, total product value/shipping fee/payment amount
* Order-time features: weekday, month, hour
* **Labels:** `is_late` (whether the order was delivered later than the estimated delivery date) and `actual_delivery_days` (actual delivery time in days) — only computable for delivered orders

All features used for training contain only information **available at the time the order is placed**, simulating a real-world scenario where delivery risk is assessed *before* shipment.

## 4. ML Models (XGBoost)

| Model                      | File                                     | Objective                             | Test Set Results                                |
| -------------------------- | ---------------------------------------- | ------------------------------------- | ----------------------------------------------- |
| `late_delivery_classifier` | `models/late_delivery_classifier.joblib` | Probability of late delivery (binary) | ROC-AUC **0.788**, Recall **0.67**, F1 **0.31** |
| `delivery_days_regressor`  | `models/delivery_days_regressor.joblib`  | Actual delivery time in days          | MAE **4.74 days**, RMSE **7.58 days**           |

Both models are implemented as `sklearn.Pipeline` pipelines (ColumnTransformer + XGBClassifier/XGBRegressor), trained on 96,470 delivered orders and split 80/20 into training and test sets.

Because the data is imbalanced (~8% late orders), the classifier uses `scale_pos_weight` to prioritize higher recall over precision.

Full details are available in `models/model_metadata.json`.

## 5. Chat Models

| Model                        | Key Required     | Notes                                                              |
| ---------------------------- | ---------------- | ------------------------------------------------------------------ |
| Gemini 2.5 Flash *(default)* | `GEMINI_API_KEY` | Free — [get it here](https://aistudio.google.com/apikey)           |
| Gemini 2.5 Flash-Lite        | `GEMINI_API_KEY` | Free, higher quota — used when Flash runs out of quota             |
| Gemini 2.0 Flash             | `GEMINI_API_KEY` | Free, previous-generation fallback                                 |
| GPT-4o mini                  | `OPENAI_API_KEY` | Paid — [platform.openai.com](https://platform.openai.com/api-keys) |
| GPT-4o                       | `OPENAI_API_KEY` | Paid, highest quality                                              |
| Grok 4 Fast                  | `XAI_API_KEY`    | Paid — [console.x.ai](https://console.x.ai) (requires credits)     |
| Grok 4.6                     | `XAI_API_KEY`    | Paid, xAI's most capable model                                     |

Only **one API key is required** (Gemini, free) to use the full application. The other keys are optional and can be used as fallbacks when Gemini reaches its quota.

Models without a configured key are shown as `(key not configured)` in the dropdown and will display a clear error message if selected.

## 6. Project Structure

```text
logistics_agent/
├── data/                          # processed_dataset.csv + category translation table
├── models/                        # trained .joblib models + model_metadata.json
├── src/
│   ├── data_pipeline.py           # load, merge, feature engineering
│   ├── train_models.py            # train classifier + regressor
│   ├── tool_schemas.py            # TOOLS (JSON Schema) + shared SYSTEM_PROMPT
│   ├── tools.py                   # business tools (ML prediction, statistics, SQL, temporary training)
│   ├── providers.py                # multi-model abstraction layer (Gemini/OpenAI/xAI)
│   ├── api.py                      # Flask API + web UI — MAIN ENTRY POINT
│   ├── agent.py, agent_gemini.py   # legacy CLI versions (terminal only, no web UI)
├── templates/index.html            # web chat interface
├── requirements.txt                 # local installation (open versions, flexible across Python versions)
├── requirements-docker.txt          # Docker dependencies (versions pinned to match model training)
├── Dockerfile, docker-compose.yml
├── .env.example                     # API key configuration template
└── README.md
```

## 7. Running Locally (without Docker)

```bash
pip install -r requirements.txt
```

Set the API key (PowerShell — only Gemini is required):

```powershell
$env:GEMINI_API_KEY="AIza..."
```

Run the web app:

```bash
python -m src.api
```

Open your browser at: **http://localhost:5000**

To rebuild the dataset/models from scratch:

```bash
python -m src.data_pipeline
python -m src.train_models
```

## 8. Running with Docker

```bash
cp .env.example .env
# open .env and enter GEMINI_API_KEY (required);
# OPENAI_API_KEY and XAI_API_KEY are optional

docker compose up --build
```

Open your browser at: **http://localhost:5000**

Next time, simply run:

```bash
docker compose up
```

No `--build` is required unless the code or dependencies have changed.

`requirements-docker.txt` pins the exact versions of scikit-learn/xgboost used when the models were trained, avoiding `InconsistentVersionWarning` issues that can occur when the local environment uses different versions.

## 9. API Endpoints

```text
GET  /                              Web chat interface
GET  /health
GET  /models                        List of models + API key configuration status
POST /predict                       {distance_km, customer_state, seller_state, ...}
GET  /order/<order_id>
GET  /stats/state/<state>
GET  /stats/seller/<seller_id>
GET  /stats/category/<category>
GET  /stats/top-risky-states?n=5
GET  /stats/top-risky-categories?n=5
POST /chat                          {"message": "...", "model": "gemini-2.5-flash"}
POST /chat/reset                    Clear conversation history for the current session
```

Example:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"distance_km": 1200, "customer_state": "BA", "seller_state": "SP"}'
```

## 10. Future Development

* Improve the precision/recall trade-off through threshold tuning or cost-sensitive learning
* Add a `recommend_seller` tool to select sellers with the best on-time delivery history for a given region
* Store conversation history in a database instead of RAM to support multiple concurrent users
* Visualize risk levels by state using folium/plotly
* Remove the unused `nvidia-nccl-cu13` dependency (~250MB), which is unnecessary for this CPU-only project, to reduce Docker image size

---

*Dataset: [Brazilian E-Commerce Public Dataset by Olist (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)*
