import pandas as pd
import numpy as np

# -----------------------
# Load data
# -----------------------
DATA_PATH = "/Users/nickyl/Documents/GitHub/DAMO-699-5-Group-6-Capstone_project-Whoop/outputs/cleaned_whoop.csv"
df = pd.read_csv(DATA_PATH)

# -----------------------
# Task 32: Compute calories per minute (metabolic intensity)
# -----------------------
# Prefer workout-level calories if available; otherwise fall back to daily calories_burned
cal_col = "activity_calories" if "activity_calories" in df.columns else "calories_burned"
dur_col = "activity_duration_min"

# Ensure numeric
df[dur_col] = pd.to_numeric(df[dur_col], errors="coerce")
df[cal_col] = pd.to_numeric(df[cal_col], errors="coerce")

# Compute calories per minute safely (avoid divide-by-zero)
df["calories_per_minute"] = np.where(
    (df[dur_col].notna()) & (df[dur_col] > 0) & (df[cal_col].notna()),
    df[cal_col] / df[dur_col],
    np.nan
)

# Optional: keep only workout rows (recommended for "by activity" ranking)
# (If your dataset uses workout_completed to indicate a workout row)
if "workout_completed" in df.columns:
    workout_df = df[(df["workout_completed"] == 1)].copy()
else:
    workout_df = df.copy()

# Filter to valid intensity records
workout_df = workout_df[
    workout_df["activity_type"].notna()
    & workout_df["calories_per_minute"].notna()
    & (workout_df[dur_col] > 0)
].copy()

print("Compute calories per minute metric")
print(workout_df[["user_id", "date", "activity_type", dur_col, cal_col, "calories_per_minute"]].head(10))


# -----------------------
# Task 33: Rank activity types by metabolic intensity
# -----------------------
ranked = (
    workout_df.groupby("activity_type", dropna=False)
    .agg(
        workouts=("activity_type", "size"),
        avg_cal_per_min=("calories_per_minute", "mean"),
        median_cal_per_min=("calories_per_minute", "median"),
        std_cal_per_min=("calories_per_minute", "std"),
        avg_duration_min=(dur_col, "mean"),
        avg_total_calories=(cal_col, "mean"),
    )
    .reset_index()
)

# Sort by average calories/min (descending)
ranked = ranked.sort_values("avg_cal_per_min", ascending=False)

# Optional: remove tiny-sample activities (set threshold as you like)
MIN_WORKOUTS = 30
ranked_filtered = ranked[ranked["workouts"] >= MIN_WORKOUTS].copy()

print("Rank activity types by metabolic intensity")
print("\nTop 10 activities by avg calories/min (min workouts =", MIN_WORKOUTS, "):")
print(ranked_filtered[["activity_type", "workouts", "avg_cal_per_min", "median_cal_per_min", "avg_duration_min"]].head(10))
