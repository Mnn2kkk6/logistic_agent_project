# AI Logistics Agent — Olist E-commerce Dataset

A local web app combining **Machine Learning** (delivery risk prediction) and a **multi-model conversational AI Agent** (Gemini / GPT-4o / Grok / Groq — automatically switch models when one provider runs out of quota) to support logistics operations for an e-commerce platform, built on the **Olist Brazilian E-commerce** dataset (~99K orders, Kaggle).

## 1. Key Features

* **Predict delivery risk + delivery time** for a new order (before shipment)
* **Vietnamese conversational Q&A** through the web interface, with the agent automatically calling the appropriate tool
* **Flexible model selection** directly from the UI: Gemini Flash/Flash-Lite/2.0, GPT-4o/mini, Grok 4 Fast/4.6, GPT-OSS 120B/20B (Groq, free), and **Local (rule-based, free, offline)** — switch models mid-conversation without losing context
* **Automatic SQL generation** (DuckDB, read-only) to answer analytical questions without a dedicated tool — monthly comparisons, payment methods, product weight, multi-table joins, etc.
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
        │                         ├─ Groq (OpenAI-compatible endpoint, free tier)
        │                         └─ Local (regex/rule-based, no API call — always available)
        │
        ▼
   src/tools.py — business functions available to the agent:
     predict_new_order · get_order_info · get_state_stats · get_seller_stats
     get_category_stats · top_risky_states · top_risky_categories
     describe_dataset · query_dataset_sql (DuckDB, SELECT only) · train_custom_model
        │
        ▼
   data/processed_dataset.csv + raw CSVs  +  models/*.joblib (XGBoost)
```

**Conversation history is stored in a "canonical" format** (text only, provider-independent) in `src/providers.py`, so switching models mid-conversation doesn't break the format. Trade-off: the new model won't "remember" internal tool calls from previous turns, only the text-based conversation history.

## 3. Problem & Data

The original dataset contains 9 CSV tables (orders, order_items, payments, reviews, products, customers, sellers, geolocation, product_category_translation), merged into **`data/processed_dataset.csv`** (99,442 rows, 41 columns) through `src/data_pipeline.py`:

* Seller → customer distance (Haversine, based on average zip-code coordinates)
* Product volume/weight, number of items, total order value/shipping/payment amounts
* Order-time features: weekday, month, hour
* **Labels:** `is_late` (delivered later than the estimated date) and `actual_delivery_days` (actual delivery duration in days) — only available for delivered orders

All training features contain only information **available at order placement time** — simulating a real-world scenario where delivery risk is evaluated *before* shipment.

For row-level analysis (per-product, per-seller, per-payment) that the merged `orders` table can't represent, the raw CSVs are also registered as DuckDB views (`order_items`, `order_payments`, `order_reviews`, `products`, `sellers`, `customers`, `category_translation`) and are queryable through `query_dataset_sql`.

> **Note:** `olist_geolocation_dataset.csv` is used by `data_pipeline.py` to compute `distance_km`, but is **not currently registered** as a queryable DuckDB view. Zip-prefix-level geographic questions (e.g. "compare late-delivery rate by distance bucket using raw geolocation data") can't be answered via `query_dataset_sql` yet — see Known Limitations.

## 4. ML Models (XGBoost)

| Model                       | File                                      | Objective                               | Test Set Results                                |
|------------------------------|-------------------------------------------|------------------------------------------|---------------------------------------------------|
| `late_delivery_classifier`   | `models/late_delivery_classifier.joblib`  | Probability of late delivery (binary)     | ROC-AUC **0.788**, Recall **0.67**, F1 **0.31**    |
| `delivery_days_regressor`    | `models/delivery_days_regressor.joblib`   | Actual delivery time in days              | MAE **4.74 days**, RMSE **7.58 days**              |

Both are `sklearn.Pipeline` models (`ColumnTransformer` + `XGBClassifier`/`XGBRegressor`), trained on 96,470 delivered orders (80/20 split). Due to class imbalance (~8% late orders), the classifier uses `scale_pos_weight` to prioritize recall over precision. Full details: `models/model_metadata.json`.

## 5. Chat Model Support

| Model                          | Key Required       | Cost / Notes                                                                 |
|----------------------------------|---------------------|--------------------------------------------------------------------------------|
| Gemini 2.5 Flash *(default)*     | `GEMINI_API_KEY`     | Free — [aistudio.google.com/apikey](https://aistudio.google.com/apikey)         |
| Gemini 2.5 Flash-Lite             | `GEMINI_API_KEY`     | Free, higher quota — fallback when Flash runs out                               |
| Gemini 2.0 Flash                  | `GEMINI_API_KEY`     | Free, previous-generation fallback                                              |
| GPT-4o mini                        | `OPENAI_API_KEY`     | Paid, low cost                                                                   |
| GPT-4o                              | `OPENAI_API_KEY`     | Paid, highest quality                                                            |
| Grok 4 Fast                         | `XAI_API_KEY`         | Paid — [console.x.ai](https://console.x.ai) (requires credits)                  |
| Grok 4.6                            | `XAI_API_KEY`         | Paid, xAI's strongest model                                                      |
| GPT-OSS 120B (Groq)                  | `GROQ_API_KEY`         | Free (~14,400 req/day), runs on Groq's fast inference hardware                   |
| GPT-OSS 20B (Groq)                    | `GROQ_API_KEY`         | Free, smaller/faster — fallback when 120B is rate-limited                       |
| Local (rule-based)                     | *no key required*      | Free, fully offline — final fallback. See §5.1 and Known Limitations            |

Only **1 key** (Gemini, free) is required for full functionality. Other keys are optional fallbacks. Models without a configured key show `(no key)` in the dropdown and return a clear error if selected.

### 5.1. Local Mode (rule-based) — How It Works

Calls no LLM/API — only detects fixed keywords/patterns (regex) to pick which tool to call. It does **not** understand free-form natural language, and always available even offline or when every paid/free provider is out of quota.

Supported query patterns include: new-order prediction, order/seller/category lookup, top-N riskiest states/categories, compare 2 states/categories, late-delivery rate by month/weekday/hour, same-state vs cross-state, installment count effect, product weight effect, review score by state/category, dataset overview/schema — plus 8 hardcoded multi-table analytical queries (2018 top categories, seller revenue-vs-review outliers, state spend-vs-delivery-time, top products outside top categories, freight-bucket delay comparison, seller processing time, high-price/low-review products).

Queries that don't match any pattern return a list of example queries and a suggestion to switch to a real AI model for more flexible questions.

## 6. Known Limitations

* **Local mode keyword matching can mis-fire on complex analytical questions.** Rules like "contains `review`" or "contains `freight` + `nhóm`" are broad enough to catch questions that are topically adjacent but actually asking something different (e.g. a question about payment methods that happens to mention "review" gets answered with an unrelated state-review ranking). If you add new analytical stress-test questions, verify Local mode's response actually matches the question before trusting it — when in doubt, switch to a real model.
* **`query_dataset_sql` depends on the LLM writing valid DuckDB SQL.** Complex multi-CTE joins occasionally trip up smaller/free models, and very long tool-call arguments (long SQL strings) can occasionally come back as malformed JSON from OpenAI-compatible endpoints (observed with Groq's free models on nested-query questions), which currently isn't caught and can surface as a generic server error instead of a graceful message.
* **`olist_geolocation_dataset.csv` isn't exposed via `query_dataset_sql`** — only the pre-aggregated `distance_km` derived from it during the data pipeline is available.
* **In-memory chat history** (`_CHAT_HISTORY` dict in `api.py`) is per-process — restarting the container clears everyone's conversations, and it isn't safe for multiple concurrent real users.

## 7. Project Structure

```text
logistics_agent/
├── data/                            # raw CSVs + processed_dataset.csv
├── models/                          # trained .joblib models + model_metadata.json
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py             # load, merge, feature engineering
│   ├── train_models.py              # train classifier + regressor
│   ├── tool_schemas.py              # TOOLS (JSON Schema) + shared SYSTEM_PROMPT
│   ├── tools.py                     # business functions (ML prediction, statistics, SQL, temp training)
│   ├── providers.py                 # multi-model abstraction layer (Gemini/OpenAI/xAI/Groq/Local)
│   ├── api.py                       # Flask API + web UI server — MAIN ENTRY POINT
│   ├── agent.py, agent_gemini.py    # legacy CLI-only prototypes (terminal-based, predate providers.py — not used by api.py)
├── templates/index.html             # web chat interface
├── requirements.txt                 # local installation (open version ranges)
├── requirements-docker.txt          # Docker dependencies (versions pinned to match trained models)
├── Dockerfile, docker-compose.yml
├── .env.example                     # API key configuration template
└── README.md
```

## 8. Run Locally (Without Docker)

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

## 9. Run with Docker

```bash
cp .env.example .env
# open .env and enter GEMINI_API_KEY (required); OPENAI_API_KEY, XAI_API_KEY, GROQ_API_KEY (optional)

docker compose up --build
```

Open your browser: **http://localhost:5000**

Next time, simply run:

```bash
docker compose up
```

No `--build` needed unless code or dependencies changed.

`requirements-docker.txt` pins the exact `scikit-learn`/`xgboost` versions used to train the models — avoiding `InconsistentVersionWarning` that can occur when the local machine uses different versions.

## 10. API Endpoints

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

## 11. Future Development

* Fix `api.py`/`providers.py` error handling so malformed tool-call JSON or unexpected provider errors return a clean JSON error instead of a raw HTML 500 page
* Tighten Local mode's keyword rules (or fall back to "I don't understand" more conservatively) to avoid confidently answering the wrong question
* Register `olist_geolocation_dataset.csv` as a DuckDB view for zip-level geographic analysis
* Improve precision/recall balance via threshold tuning or cost-sensitive learning
* Add a `recommend_seller` tool (sellers with the strongest on-time delivery history for a given region)
* Store conversation history in a database instead of in-process RAM, to support multiple concurrent users
* Visualize state-level delivery risk maps (Folium/Plotly)

---

*Dataset: [Brazilian E-Commerce Public Dataset by Olist (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)*
