# AI Logistics Agent — Olist E-commerce Dataset

A local web app combining **Machine Learning** (delivery risk prediction) and a **multi-model conversational AI Agent** (Gemini / GPT-4o / Grok — automatically switch models when one provider runs out of quota) to support logistics operations for an e-commerce platform, built on the **Olist Brazilian E-commerce** dataset (~99K orders, Kaggle).

## 1. Key Features

* **Predict delivery risk + delivery time** for a new order (before shipment)
* **Vietnamese conversational Q&A** through the web interface, with the agent automatically calling the appropriate tool
* **Flexible model selection** directly from the UI: Gemini Flash/Flash-Lite/2.0, GPT-4o/mini, Grok 4 Fast/4.6, and **Local (rule-based, free, offline)** — switch models when one provider runs out of quota without losing conversation context
* **Automatic SQL generation** to answer analytical questions without a dedicated tool (monthly comparisons, payment methods, product weight...)
* **Quickly train temporary models** for other targets/subsets beyond the two main models (e.g. predicting `review_score`, or training only for one state)
* **Dockerized deployment** — run with one command without manually installing Python/dependencies

## 2. Architecture

```text
Browser (web UI, templates/index.html)
        │  select model via dropdown
        ▼
Flask API (src/api.py)  ──►  src/providers.py (multi-model abstraction layer)
        │                         │
        │                         ├─ Gemini (google-genai)
        │                         ├─ OpenAI (GPT-4o / GPT-4o mini)
        │                         ├─ xAI/Grok (OpenAI-compatible endpoint)
        │                         └─ Local (regex/rule-based, no API call — always available)
        │
        ▼
   src/tools.py — business functions available to the agent:
     predict_new_order · get_order_info · get_state_stats · get_seller_stats
     get_category_stats · top_risky_states · top_risky_categories
     describe_dataset · query_dataset_sql (DuckDB, SELECT only) · train_custom_model
        │
        ▼
   data/processed_dataset.csv  +  models/*.joblib (XGBoost)
```

**Conversation history is stored in "canonical" format** (text only, provider-independent) in `src/providers.py`, so switching models in the middle of a conversation does not break the format — the trade-off is that the new model will not "remember" internal tool calls from previous turns, only the text-based conversation history.

## 3. Problem & Data

The original dataset contains 9 CSV tables (orders, order_items, payments, reviews, products, customers, sellers, geolocation, product_category_translation), merged into **`data/processed_dataset.csv`** (99,442 rows, 41 columns) through `src/data_pipeline.py`:

* Seller → customer distance (Haversine, based on average zip-code coordinates)
* Product volume/weight, number of items, total order value/shipping/payment amounts
* Order-time features: weekday, month, hour
* **Labels:** `is_late` (delivered later than the estimated date) and `actual_delivery_days` (actual delivery duration in days) — only available for delivered orders (`delivered`)

All features used for training contain only information **available at order placement time** — accurately simulating a real-world scenario where delivery risk is evaluated *before* shipment.

## 4. ML Models (XGBoost)

| Model                      | File                                     | Objective                             | Test Set Results                                |
| -------------------------- | ---------------------------------------- | ------------------------------------- | ----------------------------------------------- |
| `late_delivery_classifier` | `models/late_delivery_classifier.joblib` | Probability of late delivery (binary) | ROC-AUC **0.788**, Recall **0.67**, F1 **0.31** |
| `delivery_days_regressor`  | `models/delivery_days_regressor.joblib`  | Actual delivery time in days          | MAE **4.74 days**, RMSE **7.58 days**           |

Both are `sklearn.Pipeline` models (ColumnTransformer + XGBClassifier/XGBRegressor), trained on 96,470 delivered orders using an 80/20 train/test split. Due to class imbalance (~8% late orders), the classifier uses `scale_pos_weight` to prioritize higher recall over precision. Full details: `models/model_metadata.json`.

## 5. Chat Model Support

| Model                        | Key Required      | Notes                                                                                                     |
| ---------------------------- | ----------------- | --------------------------------------------------------------------------------------------------------- |
| Gemini 2.5 Flash *(default)* | `GEMINI_API_KEY`  | Free — [get it here](https://aistudio.google.com/apikey)                                                  |
| Gemini 2.5 Flash-Lite        | `GEMINI_API_KEY`  | Free, higher quota — used when Flash runs out of quota                                                    |
| Gemini 2.0 Flash             | `GEMINI_API_KEY`  | Free, previous-generation fallback                                                                        |
| GPT-4o mini                  | `OPENAI_API_KEY`  | Paid — [platform.openai.com](https://platform.openai.com/api-keys)                                        |
| GPT-4o                       | `OPENAI_API_KEY`  | Paid, highest quality                                                                                     |
| Grok 4 Fast                  | `XAI_API_KEY`     | Paid — [console.x.ai](https://console.x.ai) (requires credits)                                            |
| Grok 4.6                     | `XAI_API_KEY`     | Paid, xAI's strongest model                                                                               |
| Local (rule-based)           | *no key required* | Free, fully offline — final fallback when all other free/paid providers run out of quota. See Section 5.1 |

Only **1 key** (Gemini, free) is required to use the full feature set. Other keys are optional and serve as fallbacks when Gemini runs out of quota. Models without a configured key will appear as `(no key)` in the dropdown and return a clear error if selected. **Local** is always available and does not depend on any API key.

### 5.1. Local Mode (rule-based) — How It Works

It does not call any LLM/API — it only detects fixed keywords/patterns (regex) to determine which tool should be called. Therefore, it **does not understand natural language as flexibly** as real AI models.

In return, it is always available, even when the external network is unavailable or all other free/paid providers have exhausted their quotas.

Supported query patterns:

| Category                             | Example Query                                                   |
| ------------------------------------ | --------------------------------------------------------------- |
| New order prediction                 | "Predict an order for customer RJ, seller SP, distance 900km"   |
| Look up an order                     | "Look up order <order_id>"                                      |
| Seller statistics                    | "Statistics for seller <seller_id>"                             |
| Specific product category statistics | "How does the audio category perform?"                          |
| Top N riskiest states / categories   | "Top 5 riskiest states", "Which category has the most delays?"  |
| Compare 2 states                     | "Compare RJ and SP"                                             |
| Compare 2 categories                 | "Compare audio and food"                                        |
| Statistics for a specific state      | "Statistics for state RJ"                                       |
| Late deliveries by month             | "Late delivery rate by month"                                   |
| Late deliveries by weekday           | "Which weekday has the most delays?"                            |
| Late deliveries by order hour        | "Which time of day is most likely to be late?"                  |
| Same-state vs cross-state            | "Are same-state deliveries less likely to be late?"             |
| Effect of installment count          | "Do more installments affect delivery time?"                    |
| Effect of product weight             | "Are heavier orders more likely to be late?"                    |
| Review score by state/category       | "Which state has the lowest customer ratings?"                  |
| Dataset overview                     | "How many orders are there?", "What years does the data cover?" |
| Dataset structure                    | "How many columns are in the dataset?"                          |

Queries that do not match any supported pattern will return a list of example queries above, together with a recommendation to switch to a real AI model (Gemini/GPT/Grok) for more flexible and complex questions.

## 6. Project Structure

```text
logistics_agent/
├── data/                          # processed_dataset.csv + category translation table
├── models/                        # trained .joblib models + model_metadata.json
├── src/
│   ├── data_pipeline.py           # load, merge, feature engineering
│   ├── train_models.py            # train classifier + regressor
│   ├── tool_schemas.py            # TOOLS (JSON Schema) + shared SYSTEM_PROMPT
│   ├── tools.py                   # business functions (ML prediction, statistics, SQL, temporary training)
│   ├── providers.py                # multi-model abstraction layer (Gemini/OpenAI/xAI)
│   ├── api.py                      # Flask API + web UI server — MAIN ENTRY POINT
│   ├── agent.py, agent_gemini.py   # old CLI versions (terminal-based, no web UI)
├── templates/index.html            # web chat interface
├── requirements.txt                # local installation (open versions, flexible across Python versions)
├── requirements-docker.txt          # Docker dependencies (versions PINNED to match model training)
├── Dockerfile, docker-compose.yml
├── .env.example                    # API key configuration template
└── README.md
```

## 7. Run Locally (Without Docker)

```bash
pip install -r requirements.txt
```

Set the API key (PowerShell — only Gemini is required):

```powershell
$env:GEMINI_API_KEY="AIza..."
```

Start the web app:

```bash
python -m src.api
```

Open your browser: **http://localhost:5000**

To rebuild the dataset/model from scratch:

```bash
python -m src.data_pipeline
python -m src.train_models
```

## 8. Run with Docker

```bash
cp .env.example .env
# open .env and enter GEMINI_API_KEY (required); OPENAI_API_KEY, XAI_API_KEY (optional)

docker compose up --build
```

Open your browser: **http://localhost:5000**

Next time, simply run:

```bash
docker compose up
```

No `--build` is required unless the code or dependencies have changed.

`requirements-docker.txt` pins the exact `scikit-learn/xgboost` versions used when the models were trained — avoiding `InconsistentVersionWarning`, which can occur when the local machine uses different versions.

## 9. API Endpoints

```text
GET  /                              Web chat interface
GET  /health
GET  /models                        List available models + API key configuration status
POST /predict                       {distance_km, customer_state, seller_state, ...}
GET  /order/<order_id>
GET  /stats/state/<state>
GET  /stats/seller/<seller_id>
GET  /stats/category/<category>
GET  /stats/top-risky-states?n=5
GET  /stats/top-risky-categories?n=5
POST /chat                          {"message": "...", "model": "gemini-2.5-flash"}
POST /chat/reset                    Clear the current session's conversation history
```

Example:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"distance_km": 1200, "customer_state": "BA", "seller_state": "SP"}'
```

## 10. Future Development

* Improve the precision/recall balance through threshold tuning or cost-sensitive learning
* Add a `recommend_seller` tool (select sellers with strong on-time delivery history for a specific region)
* Store conversation history in a database instead of RAM to support multiple concurrent users
* Visualize state-level delivery risk maps using Folium/Plotly
* Remove the unnecessary `nvidia-nccl-cu13` dependency (~250MB, not required for this CPU-only project) to reduce Docker image size

---

*Dataset: [Brazilian E-Commerce Public Dataset by Olist (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)*
