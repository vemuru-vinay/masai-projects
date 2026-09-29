# Masai Projects

A collection of projects built during the Masai program, moving from basic data collection to machine learning, LLM-powered agents, and a full end-to-end capstone.

| # | Project | Focus | Stack |
|---|---------|-------|-------|
| 1 | [Mini Project 1](#mini-project-1--hacker-news-trend-analysis) | API data collection, cleaning, analysis, visualization | Python, requests, pandas, matplotlib |
| 2 | [Mini Project 2](#mini-project-2--churnguard-customer-churn-prediction) | Customer churn prediction (ML) | pandas, scikit-learn, imbalanced-learn, Jupyter |
| 3 | [Mini Project 3](#mini-project-3--supportai-helpdesk-agent) | LLM-powered helpdesk agent | Python, Groq LLM API, TF-IDF |
| 4 | [Capstone Project](#capstone-project--aiml-end-to-end) | Data pipeline + analytics/ML + RAG service | SQLite, scikit-learn, LangGraph, ChromaDB, FastAPI, Docker |

> **Secrets:** API keys are never committed. `.env` files are git-ignored. To run Mini Project 3 / the optional LLM path, copy `Mini Project 3/.env.example` to `.env` and add your own Groq key.

---

## Mini Project 1 — Hacker News Trend Analysis

**Folder:** [`Mini Project 1/`](./Mini%20Project%201)

A four-stage data pipeline over live stories from the public Hacker News API, grouping stories into categories (technology, worldnews, sports, science, entertainment) by keyword.

| Task | File | What it does |
|------|------|--------------|
| 1 | `Task - 1/task1_data_collection.py` | Pulls top stories from the Hacker News API, assigns categories by keyword, saves raw JSON to `data/` |
| 2 | `Task - 2/task2_data_processing.py` | Inspects and cleans data (removes duplicates and nulls) and writes `trends_clean.csv` |
| 3 | `Task - 3/task3_analysis.py` | Computes statistics (average score, comments, per-category figures) and writes `trends_analyzed.csv` |
| 4 | `Task - 4/task4_visualization.py` | Produces charts in `outputs/`: top stories, top categories, popular vs. not popular |

**Run:** run the four task scripts in order (`pip install requests pandas numpy matplotlib`).

---

## Mini Project 2 — ChurnGuard: Customer Churn Prediction

**Folder:** [`Mini Project 2/`](./Mini%20Project%202)

An ML workflow that predicts which telecom customers are likely to churn, built as five Jupyter notebooks that each use `churnguard_data.csv`.

| Task | Notebook | What it does |
|------|----------|--------------|
| 1 | `Task-1/load_explore.ipynb` | Loads the data, checks shape, dtypes, and summary statistics |
| 2 | `Task-2/cleaned.ipynb` | Cleaning: drops `customerID`, removes duplicates, strips whitespace, fixes types and missing values |
| 3 | `Task-3/model_training.ipynb` | One-hot encodes, splits, trains Logistic Regression. Baseline accuracy **69.5%**, but churn recall only **0.29** |
| 4 | `Task-4/task4.ipynb` | Retrains on the full data with 5 features (tenure, MonthlyCharges, TotalCharges, SeniorCitizen, Contract) and builds an interactive tool for predicting a single customer's churn |
| 5 | `Task-5/model_improvement.ipynb` | Improves on the baseline: handles class imbalance (SMOTE), scales features, and compares before/after metrics |

**Run:** `pip install pandas numpy scikit-learn imbalanced-learn jupyter`, then open the notebooks in order.

---

## Mini Project 3 — SupportAI Helpdesk Agent

**Folder:** [`Mini Project 3/`](./Mini%20Project%203) (detailed README inside)

A conversational helpdesk agent built in four stages, each reusing the last:

1. **`knowledge_base.py`** — a static FAQ knowledge base with keyword search.
2. **`llm_integration.py`** — a Groq LLM client that rephrases the matched FAQ answer naturally, staying grounded in it.
3. **`intelligent_matching.py`** — a hybrid matcher combining keyword overlap and TF-IDF semantic similarity, with confidence scores.
4. **`helpdesk_agent.py`** — the `SupportAgent`: keeps conversation history, tracks confidence, and escalates to a human when nothing matches well.

**Setup:** `pip install groq scikit-learn python-dotenv`, copy `.env.example` to `.env`, set `GROQ_API_KEY`, then `python helpdesk_agent.py`.

---

## Capstone Project — AI/ML End-to-End

**Folder:** [`Capstone Project/`](./Capstone%20Project) (each module has its own README)

A three-module capstone covering the full path from raw data to a deployable application. Install everything with `pip install -r "Capstone Project/requirements.txt"`.

### Module 1 — Data Pipeline ([`data_pipeline/`](./Capstone%20Project/data_pipeline))
Scrapes book listings from books.toscrape.com (`requests` + BeautifulSoup), cleans and types the fields (price, rating, stock), converts price to INR at a fixed 1 GBP = 105.50 INR, and loads a normalized two-table SQLite database (`categories`, `books`). Runs five SQL queries (WHERE, ORDER BY, LIMIT, DISTINCT, BETWEEN, IN, JOIN) and checks that pandas (`read_sql`, `merge`) gives identical results.
Run: `python main.py`

### Module 2 — Analytics Pipeline ([`analytics/`](./Capstone%20Project/analytics))
A full data science workflow on the Titanic dataset.
- **EDA (`01_eda.py`):** profiling, threshold-based missing-value handling, univariate/bivariate/multivariate analysis, correlation heatmap, and four data-story charts.
- **Modeling (`02_modeling.py`):** stratified split, leak-free preprocessing pipeline, Logistic Regression vs. Decision Tree vs. Random Forest, class-imbalance comparison (baseline vs. class weights vs. SMOTE), grid-search tuning, and a fare-regression side task.
- **Result:** Random Forest recommended (accuracy 0.820, F1 0.758); SMOTE gave the best F1 (0.785). The best pipeline is saved as `best_pipeline.joblib`.

Run: `python 01_eda.py` then `python 02_modeling.py`

### Module 3 — Support Assistant ([`support_assistant/`](./Capstone%20Project/support_assistant))
A retrieval-augmented generation (RAG) service over 8 company policy documents.
- Documents are embedded locally (`all-MiniLM-L6-v2`) and stored in ChromaDB.
- A LangGraph flow classifies each question, then either retrieves context or answers directly.
- Responses are validated with Pydantic (`answer`, `sources`, `confidence`) and served through a FastAPI `POST /ask` endpoint.
- Runs fully offline by default (`MOCK_LLM=1`); set `MOCK_LLM=0` plus a `GROQ_API_KEY` to use a real LLM.
- Dockerized: `docker build -t zepto-support-assistant .` and `docker run -p 7860:7860 zepto-support-assistant`.

Run: `python ingestion.py` then `uvicorn main:app --port 7860`

---

## Repository structure
```
.
├── Mini Project 1/      Hacker News pipeline (4 tasks)
├── Mini Project 2/      ChurnGuard churn prediction (5 notebooks)
├── Mini Project 3/      SupportAI helpdesk agent
└── Capstone Project/    data_pipeline/, analytics/, support_assistant/
```
