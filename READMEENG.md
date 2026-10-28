# AI Logistics Agent — Delivery Risk Prediction & Supply Chain Analytics

[Tiếng Việt](README.md) · [GitHub repository](https://github.com/Mnn2kkk6/logistic_agent_project)

AI Logistics Agent is a logistics analytics web application that combines **Machine Learning, an LLM-powered tool-calling agent, SQL analytics, and interactive order-flow maps**. Users ask operational questions in natural language and select a dataset; the backend routes the language model and invokes the relevant business tools.

The project supports two datasets: **Olist Brazilian E-Commerce** and **DataCo Supply Chain**. They use separate analytics paths. The Olist-trained XGBoost models and Brazil state map are not used to make predictions for DataCo.

## Contents

- [Features](#features)
- [Screenshots](#screenshots)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [Machine Learning on Olist](#machine-learning-on-olist)
- [AI Agent and Automatic LLM Routing](#ai-agent-and-automatic-llm-routing)
- [RAG Integrated into the AI Logistics Agent](#rag-integrated-into-the-ai-logistics-agent)
- [Logistics Maps](#logistics-maps)
- [API Reference](#api-reference)
- [Local Setup](#local-setup)
- [Docker Deployment](#docker-deployment)
- [Data Pipeline and Tests](#data-pipeline-and-tests)
- [Project Structure](#project-structure)
- [Current Limitations](#current-limitations)

## Features

### Olist E-commerce analytics (Brazil)

- Predict late-delivery probability and estimate actual delivery time in days for a new order.
- Look up orders, analyze performance by state/seller/category, and rank high-risk states and categories.
- Perform read-only SQL analytics with DuckDB.
- Analyze marketing funnels, channel performance, seller 360 profiles, and acquisition-to-logistics performance.
- Explore seller-state → customer-state order flows, filter by time/category, drill down into sample orders, and predict a route for a new order.

### DataCo Supply Chain analytics (global)

- Select DataCo from the dataset selector in the chat interface.
- Look up orders and analyze countries, regions, markets, shipping modes, and shipping-time aggregates.
- Query the normalized **dataco_orders** table with read-only SQL.
- Explore country flows by order count, the dataset's existing late-delivery-risk label, average shipping time, and sales.
- Optionally analyze access-log aggregates if a local file is provided. Raw IP and URL fields are not materialized into the agent-queryable table.

### Agent and application

- Tool/function calling connects natural-language requests to business functions.
- RAG with LangChain, Qdrant, and multilingual FastEmbed retrieves documentation, returns source citations, and abstains when evidence is insufficient.
- A deterministic intent classifier adds no LLM call: documentation-only questions use Direct RAG; data/ML questions hide the RAG tool schema; mixed questions retain both tool groups; general chat sends no tool schemas.
- One Olist/DataCo chat UI remains in place. The direct RAG API is still available for integrations and tests.
- The backend automatically selects an LLM based on task type, configured API keys, and fallback ordering, with bounded default retries.
- Backend history is limited to the latest 16 messages, and logs include total chat latency plus RAG retrieval/generation timings for diagnosis.
- A local rule-based fallback handles certain supported Olist question patterns without an external API call.
- Flask web application, health endpoint, and multi-stage Docker build.

### Chat và phân tích nghiệp vụ

<img width="1919" height="935" alt="image" src="https://github.com/user-attachments/assets/96db6384-1eae-4f79-a566-f7f266a1ab66" />


### Bản đồ luồng đơn hàng Olist

<img width="1919" height="935" alt="image" src="https://github.com/user-attachments/assets/81162530-deb7-43fc-9439-5408cc19dd5e" />

### Bản đồ luồng đơn hàng DataCo

<img width="1919" height="938" alt="image" src="https://github.com/user-attachments/assets/eaf0771e-85b4-4b25-8171-096a60528141" />

## Architecture

~~~text
                   Flask Web UI
                        |
              Select Olist or DataCo
                        |
                   Chat API /chat
                        |
              Automatic LLM Router
                /                \
     Local rules (Olist)     Online LLM providers
                \                /
                 Tool / Function Calling
                           |
           +---------------+----------------+
           |                                |
    Olist business tools              DataCo tools
    XGBoost + DuckDB                  DuckDB + sanitized fields
           |                                |
    Brazil state-flow map             Country-flow map
~~~

**RAG path:** allowlisted documents → LangChain chunking → local multilingual FastEmbed embeddings → Qdrant → retriever → configured LLM generates an answer with citations.

**Olist path:** source CSV files → order-level aggregation and feature engineering → processed dataset → XGBoost models and DuckDB tools → Flask UI/API.

**DataCo path:** DataCo CSV → encoding/column-name normalization and exclusion of selected sensitive fields from the queryable table → DuckDB analytics, order lookup, and country-flow map.

## Datasets

### 1. Olist Brazilian E-Commerce

Public source: [Brazilian E-Commerce Public Dataset by Olist — Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

The source tables cover orders, order items, payments, reviews, products, customers, sellers, geolocation, and category translation. The processed dataset is stored at **data/processed_dataset.csv**. Trained models and historical lookup features are in **models/**.

The pipeline builds features such as Haversine distance, order value, freight, item/seller/category counts, weight/volume, purchase time, seller/customer states, and historical statistics. Historical features should be carefully reviewed whenever retraining to ensure that future information or other forms of leakage do not enter the training process.

### 2. DataCo Supply Chain

The **data_2/** directory contains:

- **DataCoSupplyChainDataset.csv** — main order/line-item dataset.
- **DescriptionDataCoSupplyChain.csv** — field descriptions.
- **tokenized_access_logs.csv** — optional local-only file for access-event analytics; not required to run the core application.

The backend creates a queryable **dataco_orders** table and normalizes column names. Fields such as customer names, email, password, street address, postal code, and product-image URL are excluded from the agent's queryable table. This is not a guarantee that the entire source dataset is anonymous; review data permissions and sensitive content before sharing or deploying publicly.

**DataCo interpretation limits:** **late_delivery_risk** is a pre-existing dataset label, not a probability predicted by this project's ML model. The map uses **Customer Country → Order Country** as a proxy derived from available fields; it should not be interpreted as a verified warehouse/seller origin or a validated physical shipping route.

## Machine Learning on Olist

The project includes two XGBoost models saved as joblib artifacts and used with scikit-learn preprocessing pipelines.

| Model | Target | Result in current metadata |
|---|---|---|
| XGBClassifier | Late-delivery classification (is_late) | ROC-AUC: **0.814** |
| XGBRegressor | Actual delivery duration in days | MAE: **4.449 days** · RMSE: **7.28 days** |

These values come from **models/model_metadata.json**. They are not a performance guarantee on data outside Olist. Results may change when the dataset, train/test split, or pipeline changes.

### Classification threshold

The test late-delivery rate recorded in the metadata is approximately **8.11%**. The threshold can be adjusted to match operational priorities:

| Mode | Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Recall-priority — currently saved threshold | 0.38 | 0.193 | 0.750 | 0.307 |
| F1-optimal threshold in metadata | 0.6493 | 0.335 | 0.488 | 0.397 |

A recall-oriented threshold can identify more risky orders, but it also creates more false alarms. Production thresholds should reflect the actual cost of missed late deliveries versus unnecessary interventions.

Model files:

~~~text
models/
├── late_delivery_classifier.joblib
├── delivery_days_regressor.joblib
├── history_lookup.json
└── model_metadata.json
~~~

## AI Agent and Automatic LLM Routing

The backend routing logic is implemented in **src/model_router.py**. Users select a dataset; model selection is handled behind the scenes.

Current routing behavior:

1. For Olist, first check whether the rule-based agent recognizes a supported question pattern.
2. If an online LLM is needed, classify the task using lightweight rules as fast or reasoning-oriented.
3. Filter candidates by configured API keys, try them in priority order, and fall back to another provider after a failure.
4. Limit attempts using **MODEL_ROUTER_MAX_ATTEMPTS** (the code caps it at 8; `.env.example` defaults to 3).

The provider integrations in the code include **Google Gemini, OpenAI, xAI, Groq, and OpenRouter**, plus a local rule-based fallback. Actual availability depends on configuration, API keys, quotas, network access, and provider/model availability. The README does not imply that every model is always available or free.

### Representative tool groups

**Olist:** order lookup; new-order prediction; statistics by state/seller/category; high-risk rankings; dataset schema description; SQL queries; map-flow display; marketing analytics.

**DataCo:** dataset description; order lookup; region/country statistics; shipping-mode comparisons; read-only SQL over the sanitized table; optional access-log aggregates; country-flow map.

SQL outputs are capped. For DataCo, agent queries must use **dataco_orders**, and several keywords/functions that could modify data or read arbitrary files are blocked. These are application-level safeguards; they do not replace authentication, authorization, and infrastructure security in production.

## RAG Integrated into the AI Logistics Agent

RAG is integrated into the **single main chat UI** at **http://localhost:5000**. A deterministic intent classifier routes requests to the lightest suitable path without calling another LLM for classification.

- Documentation-only questions go directly to the existing RAG retrieval-and-generation path, skipping the agent's LLM tool-selection turn.
- Questions requiring order counts, metrics, or analytics use the selected dataset's SQL/ML tools, with the RAG tool schema omitted.
- Questions requiring both documentation and computed results use the hybrid agent with both tool groups available.
- Direct RAG cites source paths and abstains when the indexed evidence is not relevant enough. RAG does not replace SQL and must not invent numbers.

### Build or refresh the document index

Only approved sources are indexed: the bilingual READMEs, DataCo field descriptions, model metadata, and curated documents under `knowledge_base/`. Raw Olist/DataCo order CSVs are deliberately excluded. `knowledge_base/sample_delivery_sop.md` is a fictional demo SOP, not an official Olist or DataCo policy.

**Docker:**

~~~bash
docker compose up --build -d
docker compose exec logistics-agent python -m src.rag_ingest
~~~

Open **http://localhost:5000** and ask a question directly, for example: “Is DataCo's `late_delivery_risk` field a model-predicted probability?” The first indexing run downloads the multilingual embedding model through FastEmbed (ONNX on CPU) and caches it in the `fastembed_cache` volume.

**Native Python:**

~~~powershell
python -m src.rag_ingest
python -m src.api
~~~

Then open **http://localhost:5000**. Native Python execution does not automatically load `.env`; configure the LLM API key in your terminal or IDE. Embeddings run locally, while answer generation requires a configured LLM provider.

### RAG API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | /api/v1/rag/status | Check the index, chunk count, and backend configuration |
| POST | /api/v1/rag/query | Direct document-Q&A API with sources for integrations or tests |

Example direct API call (this is not a second chat interface):

~~~bash
curl -X POST http://localhost:5000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Is late_delivery_risk a model-predicted probability?","top_k":4}'
~~~

The request accepts `question` (or `query`) and `top_k` from 1 to 8. Sources include paths, similarity scores, and excerpts. The legacy `/knowledge` URL redirects to the main Agent chat.

### Configuration and limitations

- `RAG_EMBEDDING_MODEL` defaults to `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`; FastEmbed uses ONNX Runtime and does not require PyTorch.
- `RAG_LLM_PROVIDER=auto` selects a configured provider; supported values are `gemini`, `groq`, `openrouter`, `openai`, and `xai`. `RAG_LLM_MODEL` is optional.
- `RAG_MIN_RELEVANCE_SCORE` controls evidence abstention; evaluate it against representative questions.
- Index approved documents only. Never commit API secrets, personal data, or confidential company documents to a public repository.
- The demo has no per-user authentication or document-level ACLs yet. Do not expose a private knowledge base to the public Internet until access controls are implemented.

## Logistics Maps

### Olist: Brazil state-to-state order flows

Open **/map**. Features include:

- Arcs from seller state to customer state; line width and color represent the selected metric.
- Metrics such as order count, historical late rate, model-based risk, and average delivery duration.
- Month, category, and minimum-order filters, plus a state choropleth layer.
- Sample-order drill-down, monthly trend view, and new-order route prediction between two states.
- Navigation from the chat agent to a requested flow when the relevant tool is invoked.

Map coordinates aggregate or illustrate geographic areas; they are not GPS tracking or real-time vehicle locations. Model-based risk on a flow represents an estimate for a representative order created from aggregated inputs and is not directly comparable to the historical late rate.

### DataCo: country flows

Open **/map?dataset=dataco**. The map supports date, market, and shipping-mode filters, selectable metrics, and sample-order drill-down.

**Important:** it visualizes the relationship between customer-country and destination-country fields in the source data; it does not verify a physical transport route.

## API Reference

The application listens on **http://localhost:5000** by default.

### Core and Olist endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | / | Chat interface |
| GET | /health | Health check |
| GET | /models | Model information for diagnostics |
| GET | /datasets | Available datasets for the UI selector |
| POST | /chat | Send a message; JSON includes message and dataset_id |
| POST | /chat/restore | Restore chat history for a session |
| POST | /chat/reset | Clear the session's history for a dataset |
| POST | /predict | Predict a new Olist order from input features |
| GET | /order/{order_id} | Look up an Olist order |
| GET | /stats/state/{state} | State-level statistics |
| GET | /stats/seller/{seller_id} | Seller-level statistics |
| GET | /stats/category/{category} | Category-level statistics |
| GET | /stats/top-risky-states?n=5 | States with high late rates |
| GET | /stats/top-risky-categories?n=5 | Categories with high late rates |
| GET | /marketing/funnel | Marketing funnel summary |
| GET | /marketing/channels | Marketing channel performance |
| GET | /seller/{seller_id}/360 | Seller 360 view |
| GET | /marketing/acquisition-logistics | Acquisition and logistics performance |

### Olist map endpoints

~~~text
GET  /map
GET  /api/map/filters
GET  /api/map/flows
GET  /api/map/sample?from=SP&to=RJ&limit=40
GET  /api/map/trend?from=SP&to=RJ
POST /api/map/predict-route
GET  /api/map/geojson
~~~

### DataCo map endpoints

~~~text
GET  /api/dataco/map/health
GET  /api/dataco/map/filters
GET  /api/dataco/map/summary
GET  /api/dataco/map/flows
GET  /api/dataco/map/orders?from=Brazil&to=Mexico
~~~

Example chat request:

~~~bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Which five Brazilian states have the highest late-delivery rate?","dataset_id":"olist"}'
~~~

## Local Setup

### Prerequisites

- Git
- Python 3.12 is recommended to match the Docker environment
- Enough memory/disk bandwidth to read the CSV datasets; the first DataCo load may take time while the source is normalized and loaded into DuckDB
- An API key for at least one online LLM provider for questions not handled by Olist's rule-based agent. DataCo requires an online model.

### 1. Clone the repository

~~~bash
git clone https://github.com/Mnn2kkk6/logistic_agent_project.git
cd logistic_agent_project
python -m venv .venv
~~~

Activate the virtual environment:

~~~powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
~~~

~~~bash
# macOS / Linux
source .venv/bin/activate
~~~

### 2. Install dependencies

Flexible local dependencies:

~~~bash
pip install -r requirements.txt
~~~

To use the dependency versions pinned for the Docker image, use Python 3.12 and install:

~~~bash
pip install -r requirements-docker.txt
~~~

### 3. Configure an API key and run the app

Example for Windows PowerShell:

~~~powershell
$env:GEMINI_API_KEY="your-gemini-api-key"
$env:FLASK_SECRET_KEY="replace-with-a-random-secret"
python -m src.api
~~~

You can set the corresponding environment variables for OpenAI, xAI, Groq, or OpenRouter if you want to use those providers. **Running Python directly does not automatically load the .env file**; set variables in your terminal or IDE environment. Docker Compose automatically reads **.env** for variable interpolation.

Open:

- Chat: http://localhost:5000
- Health check: http://localhost:5000/health
- Dataset list: http://localhost:5000/datasets
- Olist map: http://localhost:5000/map
- DataCo map: http://localhost:5000/map?dataset=dataco

Common environment variables:

| Variable | Purpose |
|---|---|
| GEMINI_API_KEY or GOOGLE_API_KEY | Gemini |
| OPENAI_API_KEY | OpenAI |
| XAI_API_KEY | xAI |
| GROQ_API_KEY | Groq |
| OPENROUTER_API_KEY | OpenRouter |
| TAVILY_API_KEY | Optional web search |
| CARTO_API_KEY | Optional CARTO map tiles |
| FLASK_SECRET_KEY | Flask session signing |
| MODEL_ROUTER_FAST_MODEL | Preferred model for fast tasks |
| MODEL_ROUTER_REASONING_MODEL | Preferred model for reasoning tasks |
| MODEL_ROUTER_MAX_ATTEMPTS | Maximum model attempts |

Do not commit API keys or a secret-filled **.env** file to GitHub.

## Docker Deployment

Docker is the recommended route for an environment close to the project's defined runtime configuration.

1. Create **.env** from **.env.example** and fill in the API keys you need.
2. Build and start the service:

~~~bash
docker compose up --build -d
~~~

3. Check its status and logs:

~~~bash
docker compose ps
docker compose logs -f logistics-agent
~~~

4. Open **http://localhost:5000**. Health check: **http://localhost:5000/health**.

For RAG, run the ingest command above and open **http://localhost:5000**.

Stop the service:

~~~bash
docker compose down
~~~

The Dockerfile uses a multi-stage build, runs the runtime process as a non-root user, and defines a health check. Compose mounts **data/** and **models/** as read-only volumes. **data_2/** is copied into the image during build, so rebuild the image to update DataCo files inside the container.

## Data Pipeline and Tests

Rebuild the processed Olist dataset from the original CSVs:

~~~bash
python -m src.data_pipeline
~~~

Train the classifier and regressor:

~~~bash
python -m src.train_models
~~~

Train an individual model:

~~~bash
python -m src.train_models classifier
python -m src.train_models regressor
~~~

Place the required original Olist CSV files in **data/** before rebuilding the pipeline. The processed dataset and trained model artifacts are already stored in the repository for convenience; rerun training only when recreating or changing the pipeline.

DataCo tests are located at **tests/test_dataco_map.py**. Run:

~~~bash
python -m unittest discover -s tests -p "test_dataco_map.py" -v
~~~

~~~bash
python -m unittest discover -s tests -p "test_rag.py" -v
~~~

The GitHub Actions workflow **.github/workflows/dataco-backend-tests.yml** includes Python compilation, DataCo map JavaScript syntax validation, and the corresponding unit tests when triggered by its configured workflow conditions.

## Project Structure

~~~text
.
├── data/                     # Olist CSVs, processed dataset, Brazil GeoJSON
├── data_2/                   # DataCo CSV and field documentation
├── models/                   # XGBoost models, metadata, history lookup
├── knowledge_base/           # Curated knowledge sources and clearly labeled demo SOP
├── src/
│   ├── api.py                # Flask UI/API and map routes
│   ├── rag.py                # Ingestion, embeddings, Qdrant retrieval, citations, grounded generation
│   ├── rag_ingest.py         # Index creation/rebuild CLI
│   ├── agent.py              # Agent orchestration
│   ├── agent_gemini.py       # Gemini integration
│   ├── providers.py           # Provider adapters and tool calling
│   ├── model_router.py        # Automatic LLM selection and fallback
│   ├── dataset_registry.py    # Dataset availability and selector
│   ├── tools.py               # Olist tools, ML inference, DuckDB, map
│   ├── dataco_tools.py        # DataCo analytics and guarded SQL
│   ├── dataco_map.py          # Country-flow aggregation
│   ├── dataco_country_coords.py
│   ├── marketing_tools.py     # Marketing and seller analytics
│   ├── data_pipeline.py      # Olist preprocessing
│   ├── history_features.py   # Historical features
│   └── train_models.py       # Model training
├── templates/
│   ├── index.html            # Chat interface
│   ├── map.html              # Olist map
│   └── dataco_map.html       # DataCo map
├── tests/
│   └── test_dataco_map.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-docker.txt
├── .env.example
├── README.md
└── READMEENG.md
~~~

## Current Limitations

- Chat history is stored in the Flask process's RAM. It is lost when the process restarts; production or multi-instance deployment should use Redis or a database with an appropriate session strategy.
- Agent responses and tool calling depend on the provider, API keys, quotas, network connectivity, and model availability.
- The local rule-based fallback only covers explicitly programmed patterns and currently supports Olist only. DataCo requires an online provider.
- Olist and DataCo are historical datasets, not real-time logistics feeds.
- Map coordinates are for aggregate visualization, not GPS or evidence of a verified physical route.
- Before production use, validate models on newer data, assess leakage, add authentication/authorization, manage secrets carefully, and provide durable history, logging, and monitoring.

## Technology Stack

**Python · Pandas · NumPy · scikit-learn · XGBoost · DuckDB · Flask · Google GenAI · OpenAI API · xAI · Groq · OpenRouter · Tavily · Leaflet · Docker Compose**
