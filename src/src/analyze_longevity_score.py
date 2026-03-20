"""
RQ4 - Analyze Longevity Score Distribution and Trends

This script:
1. Loads longevity score data
2. Generates histogram
3. Generates boxplot
4. Generates trend plot (if date column exists)
5. Saves plots
"""

import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Step 1: Load dataset
# ------------------------------------------------------------
df = pd.read_csv("/workspaces/DAMO-699-5-Group-6-Capstone_project-Whoop/outputs/longevity_scores.csv")

# Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

score_col = "longevity_score_0_100"

if score_col not in df.columns:
    print(f"ERROR: '{score_col}' column not found.")
    print("Available columns:")
    print(df.columns.tolist())
    exit()

# ------------------------------------------------------------
# Step 2: Histogram
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.hist(df[score_col], bins=30)
plt.title("Distribution of Longevity Score")
plt.xlabel("Longevity Score (0-100)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("outputs/longevity_histogram.png")
plt.close()

# ------------------------------------------------------------
# Step 3: Boxplot
# ------------------------------------------------------------
plt.figure(figsize=(6, 5))
plt.boxplot(df[score_col])
plt.title("Boxplot of Longevity Score")
plt.ylabel("Longevity Score (0-100)")
plt.tight_layout()
plt.savefig("outputs/longevity_boxplot.png")
plt.close()

# ------------------------------------------------------------
# Step 4: Trend over time (if date column exists)
# ------------------------------------------------------------
possible_date_cols = ["date", "cycle_start_time", "sleep_start", "timestamp"]

date_col = None
for col in possible_date_cols:
    if col in df.columns:
        date_col = col
        break

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    trend_df = df.dropna(subset=[date_col]).copy()
    trend_df = trend_df.sort_values(date_col)

    trend_daily = trend_df.groupby(trend_df[date_col].dt.date)[score_col].mean()

    plt.figure(figsize=(10, 5))
    plt.plot(trend_daily.index, trend_daily.values)
    plt.title("Daily Average Longevity Score Trend")
    plt.xlabel("Date")
    plt.ylabel("Average Longevity Score (0-100)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/longevity_trend.png")
    plt.close()

    print("Trend plot saved as outputs/longevity_trend.png")
else:
    print("No date column found. Skipping trend analysis.")

# ------------------------------------------------------------
# Step 5: Summary statistics
# ------------------------------------------------------------
print("Longevity Score Summary:")
print(df[score_col].describe())
print("Plots saved successfully.")