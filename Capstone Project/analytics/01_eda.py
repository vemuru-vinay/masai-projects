import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from data_prep import load_raw_and_save_csv, clean_data

CHARTS_DIR = "charts"


def profile_dataset(df):
    print("=== df.info() ===")
    df.info()
    print("\n=== df.describe() ===")
    print(df.describe())
    print("\n=== df.shape ===")
    print(df.shape)


def iqr_outlier_count(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    return len(outliers), lower, upper


def univariate_analysis(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.histplot(df["age"], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Age Distribution")
    sns.boxplot(x=df["age"], ax=axes[0, 1])
    axes[0, 1].set_title("Age Boxplot")
    sns.histplot(df["fare"], kde=True, ax=axes[1, 0])
    axes[1, 0].set_title("Fare Distribution")
    sns.boxplot(x=df["fare"], ax=axes[1, 1])
    axes[1, 1].set_title("Fare Boxplot")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/univariate_age_fare.png")
    plt.close()

    age_outliers, age_lo, age_hi = iqr_outlier_count(df["age"])
    fare_outliers, fare_lo, fare_hi = iqr_outlier_count(df["fare"])

    print(f"\nAge IQR outliers: {age_outliers} (bounds: {age_lo:.2f} to {age_hi:.2f})")
    print(f"Fare IQR outliers: {fare_outliers} (bounds: {fare_lo:.2f} to {fare_hi:.2f})")

    fare_mean = df["fare"].mean()
    fare_median = df["fare"].median()
    fare_mode = df["fare"].mode()[0]
    print(f"\nFare mean: {fare_mean:.2f}, median: {fare_median:.2f}, mode: {fare_mode:.2f}")

    if fare_mean > fare_median > fare_mode:
        skew_conclusion = "right-skewed (mean > median > mode), driven by a small number of high-fare passengers"
    elif fare_mean < fare_median < fare_mode:
        skew_conclusion = "left-skewed (mean < median < mode)"
    else:
        skew_conclusion = "approximately symmetric (mean, median, and mode are close together)"
    print(f"Fare distribution: {skew_conclusion}")

    return age_outliers, fare_outliers, skew_conclusion


def bivariate_analysis(df):
    male_mask = df["sex"] == "male"
    female_mask = df["sex"] == "female"
    survival_by_sex = {
        "male": df.loc[male_mask, "survived"].mean(),
        "female": df.loc[female_mask, "survived"].mean(),
    }
    print("\nSurvival rate by sex:")
    for sex, rate in survival_by_sex.items():
        print(f"  {sex}: {rate:.3f}")

    survival_by_pclass = {}
    for pclass in sorted(df["pclass"].unique()):
        mask = df["pclass"] == pclass
        survival_by_pclass[pclass] = df.loc[mask, "survived"].mean()
    print("\nSurvival rate by pclass:")
    for pclass, rate in survival_by_pclass.items():
        print(f"  class {pclass}: {rate:.3f}")

    print("\nSurvival rate by sex and pclass:")
    survival_by_sex_pclass = {}
    for sex_val in ["male", "female"]:
        for pclass in sorted(df["pclass"].unique()):
            mask = (df["sex"] == sex_val) & (df["pclass"] == pclass)
            rate = df.loc[mask, "survived"].mean()
            survival_by_sex_pclass[(sex_val, pclass)] = rate
            print(f"  {sex_val}, class {pclass}: {rate:.3f}")

    return survival_by_sex, survival_by_pclass, survival_by_sex_pclass


def correlation_analysis(df):
    corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr_matrix = df[corr_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix (6 numeric columns)")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/correlation_heatmap.png")
    plt.close()

    corr_pairs = []
    for i in range(len(corr_cols)):
        for j in range(i + 1, len(corr_cols)):
            pair = (corr_cols[i], corr_cols[j])
            value = corr_matrix.iloc[i, j]
            corr_pairs.append((pair, value))
    corr_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\nTop 2 strongest correlations (by absolute value):")
    for pair, value in corr_pairs[:2]:
        print(f"  {pair[0]} vs {pair[1]}: {value:.3f}")

    return corr_matrix, corr_pairs[:2]


def multivariate_story(df):
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    sns.barplot(data=df, x="pclass", y="survived", hue="sex", ax=ax1)
    ax1.set_title("Survival Rate by Class and Sex")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/story_1_survival_by_class_sex.png")
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="survived", y="age", hue="sex", ax=ax2)
    ax2.set_title("Age Distribution by Survival and Sex")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/story_2_age_survival_sex.png")
    plt.close()

    fig3, ax3 = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x="age", y="fare", hue="survived", alpha=0.6, ax=ax3)
    ax3.set_title("Age vs Fare Colored by Survival")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/story_3_age_fare_survival.png")
    plt.close()

    fig4, ax4 = plt.subplots(figsize=(7, 5))
    sns.barplot(data=df, x="embarked", y="survived", hue="pclass", ax=ax4)
    ax4.set_title("Survival Rate by Embarkation Port and Class")
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/story_4_embarked_class_survival.png")
    plt.close()

    interpretations = [
        "Chart 1 (Survival Rate by Class and Sex): Women had a dramatically higher survival rate than men in every class, and this gap holds even in third class where overall conditions were worst. First-class women had the highest survival rate of any group, showing both sex and class independently mattered.",
        "Chart 2 (Age Distribution by Survival and Sex): Among men, survivors skew slightly younger than non-survivors, suggesting younger men were somewhat more able to reach lifeboats. Among women, age has far less impact on survival, since women were prioritized regardless of age.",
        "Chart 3 (Age vs Fare Colored by Survival): Passengers who paid higher fares (upper-right region) survived at a visibly higher rate than low-fare passengers clustered near the bottom, reinforcing that fare (a proxy for class and deck location) was protective. Age alone shows no clear separation, meaning fare/class carried more predictive signal than age by itself.",
        "Chart 4 (Survival Rate by Embarkation Port and Class): Passengers from Cherbourg had a higher survival rate than those from Southampton or Queenstown, largely because Cherbourg had a higher proportion of first-class passengers. This shows embarkation port's apparent effect on survival is mostly a class composition effect rather than a direct cause.",
    ]
    for text in interpretations:
        print(f"\n{text}")

    return interpretations


def standardization_check(df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[["age", "fare"]])
    scaled_df = pd.DataFrame(scaled, columns=["age_scaled", "fare_scaled"])

    print("\nBefore standardization:")
    print(df[["age", "fare"]].agg(["mean", "std"]))
    print("\nAfter standardization:")
    print(scaled_df.agg(["mean", "std"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["age"], kde=True, ax=axes[0], color="steelblue", label="age (raw)")
    sns.histplot(scaled_df["age_scaled"], kde=True, ax=axes[0], color="darkorange", label="age (scaled)")
    axes[0].legend()
    axes[0].set_title("Age: Before vs After Standardization")

    sns.histplot(df["fare"], kde=True, ax=axes[1], color="steelblue", label="fare (raw)")
    sns.histplot(scaled_df["fare_scaled"], kde=True, ax=axes[1], color="darkorange", label="fare (scaled)")
    axes[1].legend()
    axes[1].set_title("Fare: Before vs After Standardization")

    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/standardization_check.png")
    plt.close()


def run():
    import os
    os.makedirs(CHARTS_DIR, exist_ok=True)

    print("Step 1: Loading raw dataset and saving offline fallback CSV ...")
    raw_df = load_raw_and_save_csv()

    profile_dataset(raw_df)

    missing_pct = raw_df.isnull().mean() * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
    print("\n=== Missing value percentages ===")
    print(missing_pct)

    print("\nStep 2: Cleaning data ...")
    cleaned_df, _ = clean_data(raw_df)

    print("\nStep 3: Univariate analysis (age, fare) ...")
    univariate_analysis(cleaned_df)

    print("\nStep 4: Bivariate analysis (survival rates) ...")
    bivariate_analysis(cleaned_df)

    print("\nStep 5: Correlation analysis ...")
    correlation_analysis(cleaned_df)

    print("\nStep 6: Multivariate data story ...")
    multivariate_story(cleaned_df)

    print("\nStep 7: Standardization exploratory check ...")
    standardization_check(cleaned_df)

    print("\nEDA complete. Charts saved in ./charts/, offline fallback saved as titanic.csv")


if __name__ == "__main__":
    run()
