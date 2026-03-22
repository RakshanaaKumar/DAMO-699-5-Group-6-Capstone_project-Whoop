import pandas as pd

# Load data
df = pd.read_csv("/workspaces/DAMO-699-5-Group-6-Capstone_project-Whoop/outputs/cleaned_whoop.csv")

# Convert sleep hours to minutes for validation
df["sleep_minutes"] = df["sleep_hours"] * 60

# ---- Physiological range checks ----
invalid = pd.Series(False, index=df.index)

invalid |= (df["day_strain"] < 0) | (df["day_strain"] > 21)
invalid |= (df["recovery_score"] < 0) | (df["recovery_score"] > 100)
invalid |= (df["hrv"] < 5) | (df["hrv"] > 300)
invalid |= (df["resting_heart_rate"] < 30) | (df["resting_heart_rate"] > 120)
invalid |= (df["sleep_minutes"] < 0) | (df["sleep_minutes"] > 960)
invalid |= (df["max_heart_rate"] < 60) | (df["max_heart_rate"] > 220)
invalid |= (df["avg_heart_rate"] < 40) | (df["avg_heart_rate"] > 180)

# Sleep stages consistency
invalid |= (
    (df["light_sleep_hours"] +
     df["rem_sleep_hours"] +
     df["deep_sleep_hours"]) > df["sleep_hours"] + 0.1
)

# ---- Missing value check ----
critical_cols = [
    "day_strain",
    "recovery_score",
    "hrv",
    "resting_heart_rate",
    "sleep_hours"
]

missing_critical = df[critical_cols].isna().any(axis=1)

# ---- Final clean dataset ----
model_ready = df[~invalid & ~missing_critical]

# ---- Print Summary ----
print("Total rows:", len(df))
print("Invalid physiological rows:", invalid.sum())
print("Rows with missing critical values:", missing_critical.sum())
print("Model-ready rows:", len(model_ready))

# ---- Save only clean dataset ----
model_ready.to_csv("validated_for_modeling.csv", index=False)

print("\nSaved:")
print("validated_for_modeling.csv")
