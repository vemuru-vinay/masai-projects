import pandas as pd
import seaborn as sns

RAW_CSV_PATH = "titanic.csv"

HIGH_MISSING_THRESHOLD = 30.0
LOW_MISSING_THRESHOLD = 5.0

DROP_COLUMN_ON_HIGH_MISSING = ["deck"]


def load_raw_and_save_csv(csv_path=RAW_CSV_PATH):
    df = sns.load_dataset("titanic")
    df.to_csv(csv_path, index=False)
    return df


def load_raw_from_csv(csv_path=RAW_CSV_PATH):
    return pd.read_csv(csv_path)


def compute_missing_report(df):
    missing_pct = df.isnull().mean() * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
    return missing_pct


def clean_data(df, verbose=True):
    df = df.copy()
    missing_pct = compute_missing_report(df)

    report = []
    for column, pct in missing_pct.items():
        if column in DROP_COLUMN_ON_HIGH_MISSING and pct > HIGH_MISSING_THRESHOLD:
            df = df.drop(columns=[column])
            report.append((column, pct, "dropped column (missing rate too high for reliable imputation)"))
            continue

        if pct > HIGH_MISSING_THRESHOLD:
            df[column] = df[column].fillna("Missing")
            report.append((column, pct, "encoded 'Missing' as its own category"))
            continue

        if pct < LOW_MISSING_THRESHOLD:
            df = df.dropna(subset=[column])
            report.append((column, pct, "dropped rows with missing values (below 5% threshold)"))
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)
            report.append((column, pct, f"imputed with median ({median_value:.2f})"))
        else:
            mode_value = df[column].mode()[0]
            df[column] = df[column].fillna(mode_value)
            report.append((column, pct, f"imputed with mode ({mode_value})"))

    if verbose:
        print("Missing-value handling report:")
        for column, pct, action in report:
            print(f"  {column}: {pct:.2f}% missing -> {action}")

    return df.reset_index(drop=True), report


if __name__ == "__main__":
    raw_df = load_raw_and_save_csv()
    cleaned_df, _ = clean_data(raw_df)
    print(cleaned_df.shape)
