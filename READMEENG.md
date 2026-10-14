# AI Logistics Agent — Olist E-commerce

AI Logistics Agent is a logistics analytics web application for e-commerce operations. The project combines **Machine Learning, LLM Function Calling, SQL analytics, and an interactive order-flow map** to answer operational questions from the Olist Brazilian E-Commerce dataset.

> **Dataset:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

**Vietnamese documentation:** [README.md](README.md)

## 1. Highlights

- **Late-delivery prediction:** estimate the probability that a new order will be delivered late.
- **Delivery-time prediction:** estimate the actual delivery time in days.
- **AI Agent:** users ask natural-language questions; the selected LLM chooses the appropriate business tool.
- **Multi-provider LLM:** Gemini, OpenAI, xAI/Grok, Groq, and a local rule-based fallback.
- **DuckDB analytics:** run read-only `SELECT` / `WITH ... SELECT` queries across the dataset.
- **Operational lookup and statistics:** order lookup, seller/state/category performance, and risk rankings.
- **Interactive flow map:** visualize seller-state → customer-state flows, drill into sample orders, and animate deliveries.
- **Optional Tavily integration:** web search utility for information outside the Olist dataset.
- **Dockerized deployment:** Flask service with healthcheck and multi-stage Docker build.

## 2. Problem and Data

The project uses **9 original Olist CSV tables** covering orders, order items, payments, reviews, products, customers, sellers, geolocation, and category translation.

Main pipeline:

~~~text
Raw Olist CSV
     |
     v
Load + Merge + Feature Engineering
     |
     +--> Historical features
     |
     v
processed_dataset.csv
     |
     +-------------------+
     |                   |
     v                   v
ML Training          DuckDB Analytics
     |                   |
     v                   v
2 Production Models   Agent Tools / SQL
     |                   |
     +---------+---------+
               |
               v
        Flask Web Application
          |             |
          v             v
       Chat UI  <-->  Flow Map
~~~

### Main features

The feature set is designed around information available **at order placement time**, avoiding post-delivery information leakage.

Examples:

- Haversine distance between seller and customer.
- Product price, freight, payment value, and installments.
- Number of items, distinct sellers, and distinct categories.
- Total weight, volume, price-per-item, and log-transformed variables.
- Seller state, customer state, category, and payment type.
- Purchase weekday, month, hour, weekend and peak-season indicators.
- Historical features for seller, category, state pair, and customer state.

**Targets:**
- `is_late`: whether the order was delivered after the estimated delivery date.
- `actual_delivery_days`: elapsed days from purchase to customer delivery.

## 3. Machine Learning

Two production models are trained with **XGBoost** inside `sklearn.Pipeline`, using separate numeric and categorical preprocessing.

| Model | Target | Test result |
|---|---|---|
| XGBClassifier | `is_late` | ROC-AUC **0.814** |
| XGBRegressor | `actual_delivery_days` | MAE **4.449 days**, RMSE **7.28 days** |

### Classifier operating points

The test late-rate is about **8.11%**, so the classifier threshold can be selected according to business priorities:

| Mode | Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Recall-priority (deployed) | **0.38** | 0.193 | **0.750** | 0.307 |
| F1-optimal | **0.6493** | 0.335 | 0.488 | **0.397** |

This allows the system to trade off **catching more risky orders** against **reducing false alarms**, instead of always using threshold = 0.5.

Model metadata is stored in:

~~~text
models/model_metadata.json
~~~

## 4. AI Agent

Conversation flow:

~~~text
User
  |
  v
Flask Chat API
  |
  v
Selected LLM Provider
  |
  +--> Function Calling
  |       |
  |       +--> get_order_info
  |       +--> predict_new_order
  |       +--> get_state_stats
  |       +--> get_seller_stats
  |       +--> get_category_stats
  |       +--> top_risky_states
  |       +--> top_risky_categories
  |       +--> describe_dataset
  |       +--> query_dataset_sql
  |       +--> train_custom_model
  |
  v
Tool result
  |
  v
LLM final answer
~~~

### Tool groups

**Prediction**
- `predict_new_order`

Uses information available at order placement to return late-delivery risk and expected delivery days. When `seller_id` is known, historical seller performance can also be incorporated.

**Lookup / statistics**
- `get_order_info`
- `get_state_stats`
- `get_seller_stats`
- `get_category_stats`
- `top_risky_states`
- `top_risky_categories`

**General analytics**
- `describe_dataset`
- `query_dataset_sql`
- `train_custom_model`
- `show_flow_on_map`

`query_dataset_sql` is restricted to read-only queries and can access both the aggregated `orders` table and the raw tables needed for detailed joins.

## 5. Multi-provider LLM

The web UI lets the user select a model directly:

| Provider | Models in current project |
|---|---|
| Google Gemini | Gemini 2.5 Flash, 2.5 Flash-Lite, 2.0 Flash |
| OpenAI | GPT-4o Mini, GPT-4o |
| xAI | Grok 4 Fast, Grok 4.6 |
| Groq | GPT-OSS 120B, GPT-OSS 20B |
| Local | Rule-based offline fallback |

Conversation history is stored in a **canonical provider-independent format**, so the user can switch models between turns without storing provider-specific tool-call formats.

## 6. Interactive Logistics Flow Map

Open:

```
http://localhost:5000/map
```

The map is built with **Leaflet** (CARTO dark basemap or OpenStreetMap, switchable at the top right) and provides:

- **State-to-state flows:** curved arcs `seller_state → customer_state`; thickness = order volume, colour = the selected metric.
- **Four colour metrics:** historical late rate, **predicted risk (XGBoost)**, average delivery days, order count. The legend adapts to the data.
- **State choropleth** (by destination or origin state) from a 27-state GeoJSON (`data/brazil_states.geojson`). Click a state to show only its flows.
- **Filters:** month range, product category, minimum orders per flow.
- **Top 10 highlighted flows** for the selected metric (click to drill down).
- **Hover:** tooltip with stats plus a monthly late-rate sparkline.
- **Drill-down:** 40 **stratified** sample orders (both late and on-time), dots move along the arcs at a speed inversely proportional to real delivery days; pause button and 0.5×–4× speed control.
- **New-order prediction:** pick seller/customer states, category, price, freight, weight → the route is drawn with late probability, risk level and expected delivery days.
- **Agent integration:** the `show_flow_on_map` tool — ask "show me the SP → RJ flow on the map" and the chat UI opens the map beside the conversation and zooms to that flow. In the other direction, the **"Ask agent"** button in the drill-down view sends a ready-made question to the chat.
- Deep links: `/map?from=SP&to=RJ&mode=pred` (`mode`: `late` | `pred` | `days` | `orders`).

> **Note:** map coordinates are **area/ZIP-code averages used for visualization**, not real-time GPS. "Predicted risk" is the model output for a **typical order** of each flow (distance, price, freight, weight and estimated days taken from the flow's real median/mean), so it is not directly comparable 1-to-1 with the historical late rate.

Aggregates are cached in memory and pre-computed at server start (disable with `MAP_WARMUP=0`).

### Map endpoints

```
GET  /map
GET  /api/map/filters
GET  /api/map/flows?start=2017-01&end=2018-08&category=bed_bath_table&min_orders=5
GET  /api/map/sample?from=SP&to=RJ&limit=40[&start=&end=&category=]
GET  /api/map/trend?from=SP&to=RJ[&start=&end=&category=]
POST /api/map/predict-route
GET  /api/map/geojson
```

## 7. Tavily Web Search

The project includes a `web_search` utility backed by **Tavily** for information outside the Olist dataset, such as general logistics knowledge or external company information.

Configure:

~~~text
TAVILY_API_KEY=
~~~

For production usage, internal Olist data should remain the first source for questions that can be answered by the project's analytics tools.

## 8. Web API

Main endpoints:

~~~text
GET  /                         Web chat UI
GET  /health                   Healthcheck
GET  /models                   Available LLM models + key status

POST /chat                     Chat with selected model
POST /chat/reset               Reset current conversation

POST /predict                  Predict a new order
GET  /order/<order_id>         Lookup an order

GET  /stats/state/<state>
GET  /stats/seller/<seller_id>
GET  /stats/category/<category>
GET  /stats/top-risky-states?n=5
GET  /stats/top-risky-categories?n=5

GET  /map
GET  /api/map/filters
GET  /api/map/flows
GET  /api/map/sample?from=SP&to=RJ&limit=40
GET  /api/map/trend?from=SP&to=RJ
POST /api/map/predict-route
GET  /api/map/geojson
~~~

## 9. Project structure

~~~text
.
├── data/
│   ├── olist_*_dataset.csv
│   ├── product_category_name_translation.csv
│   ├── brazil_states.geojson
│   └── processed_dataset.csv
│
├── models/
│   ├── late_delivery_classifier.joblib
│   ├── delivery_days_regressor.joblib
│   ├── history_lookup.json
│   └── model_metadata.json
│
├── src/
│   ├── api.py
│   ├── agent.py
│   ├── agent_gemini.py
│   ├── providers.py
│   ├── tool_schemas.py
│   ├── tools.py
│   ├── data_pipeline.py
│   ├── history_features.py
│   ├── train_models.py
│   └── set_recall_threshold.py
│
├── templates/
│   ├── index.html
│   └── map.html
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-docker.txt
├── .env.example
├── README.md
└── READMEENG.md
~~~

## 10. Local setup

### 10.1. Install dependencies

~~~bash
pip install -r requirements.txt
~~~

### 10.2. Configure environment

Copy:

~~~text
.env.example
~~~

to:

~~~text
.env
~~~

Then fill in the API keys you need.

> Never commit the real `.env` file to GitHub.

### 10.3. Run Flask

PowerShell:

~~~powershell
$env:GEMINI_API_KEY="your-key"
python -m src.api
~~~

Open:

~~~text
http://localhost:5000
~~~

Map:

~~~text
http://localhost:5000/map
~~~

### 10.4. Run data pipeline / training

Rebuild the processed dataset:

~~~bash
python -m src.data_pipeline
~~~

Train both production models:

~~~bash
python -m src.train_models
~~~

Or train separately:

~~~bash
python -m src.train_models classifier
python -m src.train_models regressor
~~~

## 11. Docker

Build and start:

~~~bash
docker compose up --build
~~~

Check:

~~~bash
docker compose ps
~~~

Healthcheck:

~~~text
http://localhost:5000/health
~~~

Application:

~~~text
http://localhost:5000
~~~

Map:

~~~text
http://localhost:5000/map
~~~

The Dockerfile uses a multi-stage build: dependencies are installed in the builder stage, while the runtime stage keeps only the environment required to serve the application.

Docker Compose mounts the dataset and model directories read-only so they can be replaced without rebuilding the image.

## 12. Example questions

~~~text
What are the top 5 customer states with the highest late-delivery rate?

Does a new order from a seller in SP to a customer in RJ, 900 km away, have a high late-delivery risk?

How is this seller performing?

Compare late-delivery rates by month.

Which product categories have the longest delivery times?

Show the order flow from SP to RJ on the map.
~~~

## 13. Limitations and future work

- Olist is historical data; the map is intended for visualization rather than live tracking.
- Chat history is currently stored in Flask process memory; multi-instance deployments should use Redis or a database.
- Each external LLM provider requires its own API key; the local fallback only supports a limited set of patterns.
- Possible extensions include seller recommendation, route-aware prediction, model monitoring, drift detection, and long-term conversation memory.
- Classifier performance can be further optimized using business-cost-based thresholding or cost-sensitive learning.

---

**Stack:** Python, Pandas, scikit-learn, XGBoost, DuckDB, Flask, Google GenAI, OpenAI API, xAI API, Groq, Tavily, Leaflet, Docker