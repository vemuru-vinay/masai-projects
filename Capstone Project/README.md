# AI/ML Capstone Project

A three-part capstone project covering data engineering, exploratory data analysis and machine learning, and a GenAI-powered service — built as an end-to-end demonstration of the full pipeline from raw data to a deployable application.

## Setup

This project uses **one consolidated `requirements.txt` at the repository root** covering all three modules — install everything at once with:
```
pip install -r requirements.txt
```
Each module's own README also documents any module-specific run steps (e.g. running scripts in a particular order, environment variables, Docker commands).

## Module 1: Data Pipeline

A web scraping and data engineering pipeline. Book listing data is scraped from a public practice site, cleaned and converted into proper types (price, rating, stock status), enriched with a currency conversion to INR, and loaded into a normalized SQLite database with a two-table schema. The pipeline then demonstrates querying that data both with raw SQL (covering filtering, sorting, joins, and more) and with pandas, including a side-by-side check confirming both approaches produce identical results.

**What it shows:** web scraping with `requests`/`BeautifulSoup`, data cleaning and type conversion, relational database design, and SQL/pandas interoperability.

Full details, setup steps, and results are in [`data_pipeline/README.md`](./data_pipeline/README.md).

## Module 2: Analytics Pipeline

A full data science workflow on the Titanic dataset, split into two stages that share one cleaned dataset. The first stage profiles the raw data, handles missing values with a defensible threshold-based strategy, and builds a visual "data story" around who was most likely to survive and why, using univariate, bivariate, and multivariate analysis plus a correlation heatmap. The second stage builds on that same cleaned data to train and rigorously evaluate three classifiers (Logistic Regression, Decision Tree, and Random Forest), compares strategies for handling class imbalance (including SMOTE), tunes the Random Forest with grid search, and runs a separate regression task predicting ticket fare. The best-performing model pipeline — preprocessing and classifier bundled together — is saved to disk in a form that can be reloaded and used directly on new, raw data.

**What it shows:** exploratory data analysis, data storytelling through visualization, leak-free preprocessing pipelines, multi-model evaluation, imbalance handling, hyperparameter tuning, and regression diagnostics.

Full details, setup steps, and results are in [`analytics/README.md`](./analytics/README.md).

## Module 3: Support Assistant

A small retrieval-augmented generation (RAG) service built around a set of company policy documents. The documents are embedded locally and stored in a vector database, and an orchestrated flow classifies each incoming question, decides whether it needs to retrieve supporting context, and returns a structured, schema-validated response. The whole service is wrapped behind a REST API and can run in a fully self-contained, offline mode with no external API calls — with an optional path to plug in a real language model for generation. The service is also containerized so it can be built and run anywhere with Docker.

**What it shows:** embedding and vector search, graph-based orchestration of a multi-step reasoning flow, structured output validation, API design, and containerization.

Full details, setup steps, and results are in [`support_assistant/README.md`](./support_assistant/README.md).

## Tech stack
Python, pandas, SQLite, scikit-learn, imbalanced-learn, seaborn, matplotlib, LangGraph, ChromaDB, sentence-transformers, FastAPI, Docker.

## Structure
```
.
├── requirements.txt
├── data_pipeline/
│   ├── main.py
│   └── README.md
├── analytics/
│   ├── data_prep.py
│   ├── 01_eda.py
│   ├── 02_modeling.py
│   └── README.md
├── support_assistant/
│   ├── docs/
│   ├── ingestion.py
│   ├── graph.py
│   ├── main.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README.md
└── README.md   (this file)
```
