import re
import sqlite3
import statistics
import requests
import pandas as pd
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR = 105.50
MIN_BOOKS = 60
MIN_CATEGORIES = 3
DB_PATH = "books.db"

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 (CapstoneDataPipeline/1.0)"})


def get_soup(url):
    response = session.get(url, timeout=15)
    response.raise_for_status()
    return BeautifulSoup(response.text, "lxml")


def get_category_links():
    soup = get_soup(BASE_URL)
    nav = soup.select("div.side_categories ul li ul li a")
    categories = []
    for a in nav:
        name = a.get_text(strip=True)
        href = BASE_URL + a["href"]
        categories.append((name, href))
    return categories


def parse_listing_page(url):
    soup = get_soup(url)
    articles = soup.select("article.product_pod")
    rows = []
    for art in articles:
        title = art.h3.a["title"].strip()
        price_text = art.select_one("p.price_color").get_text(strip=True)
        rating_class = art.select_one("p.star-rating")["class"]
        rating_text = [c for c in rating_class if c != "star-rating"]
        rating_text = rating_text[0] if rating_text else None
        availability_text = art.select_one("p.instock.availability").get_text(strip=True)
        rows.append({
            "title": title,
            "price_text": price_text,
            "rating_text": rating_text,
            "availability_text": availability_text,
        })
    next_link = soup.select_one("li.next a")
    next_url = None
    if next_link:
        next_url = url.rsplit("/", 1)[0] + "/" + next_link["href"]
    return rows, next_url


def scrape_categories():
    categories = get_category_links()
    collected = []
    used_categories = 0
    for name, url in categories:
        if len(collected) >= MIN_BOOKS and used_categories >= MIN_CATEGORIES:
            break
        page_url = url
        category_had_rows = False
        while page_url:
            rows, next_url = parse_listing_page(page_url)
            for row in rows:
                row["category"] = name
                collected.append(row)
                category_had_rows = True
            page_url = next_url
        if category_had_rows:
            used_categories += 1
    return collected


def clean_price(price_text):
    match = re.search(r"[\d.]+", price_text)
    if not match:
        return None
    return float(match.group())


def clean_rating(rating_text):
    return RATING_MAP.get(rating_text)


def clean_availability(availability_text):
    text = availability_text.lower()
    if "in stock" in text:
        return 1
    if "out of stock" in text:
        return 0
    return None


def clean_rows(raw_rows):
    cleaned = []
    dropped = 0
    for row in raw_rows:
        price_gbp = clean_price(row["price_text"])
        rating = clean_rating(row["rating_text"])
        in_stock = clean_availability(row["availability_text"])
        if price_gbp is None or in_stock is None:
            dropped += 1
            continue
        cleaned.append({
            "title": row["title"],
            "price_gbp": price_gbp,
            "rating": rating,
            "in_stock": in_stock,
            "category": row["category"],
        })

    ratings_present = [r["rating"] for r in cleaned if r["rating"] is not None]
    median_rating = int(round(statistics.median(ratings_present))) if ratings_present else 3
    for row in cleaned:
        if row["rating"] is None:
            row["rating"] = median_rating

    for row in cleaned:
        row["price_inr"] = round(row["price_gbp"] * GBP_TO_INR, 2)

    return cleaned, dropped


def build_database(rows, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS books")
    cur.execute("DROP TABLE IF EXISTS categories")
    cur.execute("""
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            FOREIGN KEY (category_id) REFERENCES categories(category_id)
        )
    """)

    category_ids = {}
    for row in rows:
        name = row["category"]
        if name not in category_ids:
            cur.execute("INSERT INTO categories (category_name) VALUES (?)", (name,))
            category_ids[name] = cur.lastrowid

    for row in rows:
        cur.execute("""
            INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            row["title"],
            row["price_gbp"],
            row["price_inr"],
            row["rating"],
            row["in_stock"],
            category_ids[row["category"]],
        ))

    conn.commit()
    conn.close()


QUERIES = {
    "in_stock_under_20_gbp": """
        SELECT title, price_gbp, rating
        FROM books
        WHERE in_stock = 1 AND price_gbp < 20
        ORDER BY price_gbp ASC
        LIMIT 10
    """,
    "distinct_categories": """
        SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name ASC
    """,
    "price_between_10_and_30": """
        SELECT title, price_gbp
        FROM books
        WHERE price_gbp BETWEEN 10 AND 30
        ORDER BY price_gbp DESC
        LIMIT 15
    """,
    "rating_in_four_five": """
        SELECT title, rating, price_inr
        FROM books
        WHERE rating IN (4, 5)
        ORDER BY rating DESC, price_inr ASC
    """,
    "top_rated_per_category_join": """
        SELECT c.category_name, b.title, b.rating, b.price_gbp
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        ORDER BY c.category_name ASC, b.rating DESC, b.price_gbp ASC
        LIMIT 10
    """,
}


def run_all_queries(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    for name, sql in QUERIES.items():
        cur.execute(sql)
        rows = cur.fetchall()
        columns = [d[0] for d in cur.description]
        print(f"\n--- {name} ---")
        print(columns)
        for r in rows:
            print(r)
    conn.close()


def pandas_verification(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    df_in_stock = pd.read_sql(QUERIES["in_stock_under_20_gbp"], conn)
    df_join_sql = pd.read_sql(QUERIES["top_rated_per_category_join"], conn)

    books_df = pd.read_sql("SELECT * FROM books", conn)
    categories_df = pd.read_sql("SELECT * FROM categories", conn)
    conn.close()

    merged = pd.merge(books_df, categories_df, on="category_id", how="inner")
    merged = merged.sort_values(
        by=["category_name", "rating", "price_gbp"],
        ascending=[True, False, True]
    )
    df_join_merge = merged[["category_name", "title", "rating", "price_gbp"]].head(10).reset_index(drop=True)
    df_join_sql = df_join_sql.reset_index(drop=True)

    print("\n--- pd.read_sql: in_stock_under_20_gbp ---")
    print(df_in_stock)
    print("\n--- pd.read_sql: top_rated_per_category_join ---")
    print(df_join_sql)
    print("\n--- pd.merge equivalent of JOIN query ---")
    print(df_join_merge)

    equivalent = df_join_sql.values.tolist() == df_join_merge.values.tolist()
    print(f"\nread_sql vs merge outputs equivalent: {equivalent}")


def run():
    print("Step 1: Scraping books.toscrape.com ...")
    raw_rows = scrape_categories()
    print(f"Scraped rows: {len(raw_rows)}")

    print("\nStep 2: Cleaning and converting ...")
    cleaned_rows, dropped = clean_rows(raw_rows)
    print(f"Dropped rows: {dropped}")
    print(f"Loaded rows: {len(cleaned_rows)}")
    print(f"Categories used: {len(set(r['category'] for r in cleaned_rows))}")
    print(f"Fixed conversion rate used: 1 GBP = {GBP_TO_INR} INR")

    print("\nStep 3: Building SQLite database ...")
    build_database(cleaned_rows)
    print(f"Database written to {DB_PATH}")

    print("\nStep 4: Running SQL queries ...")
    run_all_queries()

    print("\nStep 5: pandas read_sql / merge verification ...")
    pandas_verification()


if __name__ == "__main__":
    run()