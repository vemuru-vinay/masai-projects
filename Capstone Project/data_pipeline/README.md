# Module 1 — Data Pipeline

## What this does
Scrapes book listings from books.toscrape.com, cleans and types the fields, converts price to INR using a fixed baseline rate, loads everything into a normalized SQLite database, and runs SQL + pandas queries against it.

## How to run
1. Install dependencies:
   ```
   pip install requests beautifulsoup4 lxml pandas
   ```
2. Run the pipeline end to end:
   ```
   python main.py
   ```
   This will scrape the site, build `books.db` in this folder, run all 5 SQL queries, and print the pandas verification — all in one execution.

## Currency conversion
`price_inr` is computed using a fixed, project-defined baseline rate:
**1 GBP = 105.50 INR**
This is not a live/historical market rate and requires no external API call — it's a constant applied directly in the cleaning step.

## Data cleaning decisions
- **price_gbp**: extracted as a float from the raw price text (currency symbol stripped via regex). Rows where price can't be parsed are **dropped** — price is a structural field the row can't function without.
- **rating**: mapped from text (`One`–`Five`) to integer 1–5. If a rating fails to parse, the row is **not dropped** — instead the missing rating is **median-imputed** from the ratings that did parse successfully, since rating is a lower-stakes ordinal field and dropping otherwise-valid rows over it would waste data.
- **in_stock**: parsed from availability text into a boolean (1/0). Rows where this can't be determined are **dropped**, since stock status is a structural field.
- Scraping continues across categories until the dataset reaches at least 60 books spanning at least 3 categories.

## Database schema
Two tables with a primary/foreign key relationship:
```
categories(category_id INTEGER PRIMARY KEY, category_name TEXT UNIQUE)
books(book_id INTEGER PRIMARY KEY, title TEXT, price_gbp REAL, price_inr REAL,
      rating INTEGER, in_stock INTEGER, category_id INTEGER REFERENCES categories(category_id))
```

## Queries
Five SQL queries are run against the database, covering SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, BETWEEN, IN, and a JOIN across both tables. Full query text is in `main.py`; their printed output is captured in `output.txt`.

## pandas verification
Two query results are re-read via `pd.read_sql(...)`, and the JOIN query is separately reproduced using `pd.merge(...)` on the in-memory DataFrames (no SQL). Both outputs are printed side by side in `output.txt` and confirmed equivalent.

## Files in this folder
- `main.py` — full pipeline: scrape → clean → convert → load → query → verify
- `books.db` — generated SQLite database
- `output.txt` — captured run output (query results + pandas verification)
- `README.md` — this file