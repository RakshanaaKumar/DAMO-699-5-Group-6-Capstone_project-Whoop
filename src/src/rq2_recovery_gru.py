import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# =========================================================
# CONFIG
# =========================================================
DATA_PATH = "/workspaces/DAMO-699-5-Group-6-Capstone_project-Whoop/outputs/cleaned_whoop.csv"
OUTPUT_DIR = "/workspaces/DAMO-699-5-Group-6-Capstone_project-Whoop/outputs"

WINDOW_SIZE = 7
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 20
BATCH_SIZE = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def find_column(df, possible_names, required=True):
    lower_map = {col.lower().strip(): col for col in df.columns}
    for name in possible_names:
        key = name.lower().strip()
        if key in lower_map:
            return lower_map[key]
    if required:
        raise ValueError(
            f"Could not find any of these columns: {possible_names}\n"
            f"Available columns: {list(df.columns)}"
        )
    return None


def create_sequences(features, target, window_size=7):
    X_seq, y_seq = [], []
    for i in range(window_size, len(features)):
        X_seq.append(features[i - window_size:i])
        y_seq.append(target[i])
    return np.array(X_seq), np.array(y_seq)


def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "Model": name,
        "MAE": round(float(mae), 3),
        "RMSE": round(float(rmse), 3),
        "R2": round(float(r2), 3)
    }


# =========================================================
# LOAD DATA
# =========================================================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")
print("Shape:", df.shape)
print("Columns:", list(df.columns))


# =========================================================
# SORT BY DATE IF AVAILABLE
# =========================================================
date_col = find_column(
    df,
    ["date", "Date", "datetime", "Datetime", "timestamp", "Timestamp"],
    required=False
)

if date_col is not None:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(by=date_col).reset_index(drop=True)


# =========================================================
# TARGET: NEXT-DAY RECOVERY
# =========================================================
# Prefer an existing next_day_recovery column if present.
# Otherwise create it by shifting recovery_score by -1.
existing_next_day_target = find_column(
    df,
    ["next_day_recovery", "Next Day Recovery", "next day recovery"],
    required=False
)

if existing_next_day_target is not None:
    target_col = existing_next_day_target
else:
    recovery_col = find_column(
        df,
        ["recovery_score", "Recovery Score", "recovery", "Recovery"]
    )
    target_col = "next_day_recovery_created"
    df[target_col] = pd.to_numeric(df[recovery_col], errors="coerce").shift(-1)

print("\nTarget column used:", target_col)


# =========================================================
# FEATURE SELECTION (NO LEAKAGE)
# =========================================================
# Use only current-day biometric features.
# Exclude any column containing next_day or recovery target-like leakage.
candidate_features = [
    "hrv",
    "sleep_efficiency",
    "day_strain",
    "deep_sleep_hours",
    "rem_sleep_hours",
    "light_sleep_hours",
    "resting_heart_rate",
    "calories_burned",
    "sleep_performance",
    "respiratory_rate",
    "skin_temp_deviation",
    "avg_heart_rate",
    "max_heart_rate",
    "activity_duration",
    "activity_strain",
    "calories_per_minute",
    "wake_ups",
    "time_to_fall_asleep_min"
]

selected_features = []
lower_cols = {c.lower().strip(): c for c in df.columns}

for feat in candidate_features:
    if feat.lower() in lower_cols:
        selected_features.append(lower_cols[feat.lower()])

# If some names differ, search by partial matching
if len(selected_features) < 5:
    fallback_features = []
    for col in df.columns:
        col_low = col.lower().strip()

        # Exclude leakage columns
        if "next_day" in col_low:
            continue
        if col_low == target_col.lower():
            continue
        if col_low in ["recovery_score", "recovery", "score"]:
            continue
        if "recovery" in col_low:
            continue

        # Include only numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            fallback_features.append(col)

    selected_features = fallback_features[:12]

# Remove duplicates
selected_features = list(dict.fromkeys(selected_features))

print("Selected features before cleaning:", selected_features)


# =========================================================
# BUILD MODEL DATAFRAME
# =========================================================
needed_cols = selected_features + [target_col]
model_df = df[needed_cols].copy()

for col in model_df.columns:
    model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

# Remove rows with missing target
model_df = model_df.dropna(subset=[target_col]).reset_index(drop=True)

# Fill only feature-side gaps
feature_only_cols = [c for c in model_df.columns if c != target_col]
model_df[feature_only_cols] = model_df[feature_only_cols].ffill().bfill()

# Add safe lag features ONLY for predictors, not target
for col in selected_features:
    model_df[f"{col}_lag1"] = model_df[col].shift(1)

# Final cleanup
model_df = model_df.ffill().bfill()
model_df = model_df.dropna().reset_index(drop=True)

if len(model_df) < WINDOW_SIZE + 30:
    raise ValueError(f"Not enough rows after preprocessing. Rows available: {len(model_df)}")

feature_cols = [c for c in model_df.columns if c != target_col]

# Final leakage check
safe_feature_cols = []
for col in feature_cols:
    cl = col.lower()
    if "next_day" in cl:
        continue
    if cl == target_col.lower():
        continue
    if "recovery_score" in cl:
        continue
    safe_feature_cols.append(col)

feature_cols = safe_feature_cols

print("\nFinal feature count:", len(feature_cols))
print("Final features used:", feature_cols)


# =========================================================
# PREPARE X AND y
# =========================================================
X = model_df[feature_cols].values
y = model_df[target_col].values.reshape(-1, 1)

# Scale features
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)

# Scale target for GRU only
target_scaler = MinMaxScaler()
y_scaled = target_scaler.fit_transform(y).flatten()

# Build sequences
X_seq, y_seq = create_sequences(X_scaled, y_scaled, WINDOW_SIZE)

# Align baseline data
X_baseline = X_scaled[WINDOW_SIZE:]
y_baseline_original = y.flatten()[WINDOW_SIZE:]

# Time-based split
split_idx = int(len(X_seq) * (1 - TEST_SIZE))

X_train_base, X_test_base = X_baseline[:split_idx], X_baseline[split_idx:]
y_train_base, y_test_base = y_baseline_original[:split_idx], y_baseline_original[split_idx:]

X_train_seq, X_test_seq = X_seq[:split_idx], X_seq[split_idx:]
y_train_seq, y_test_seq = y_seq[:split_idx], y_seq[split_idx:]

y_test_seq_original = target_scaler.inverse_transform(
    y_test_seq.reshape(-1, 1)
).flatten()

print("\nBaseline train shape:", X_train_base.shape)
print("Baseline test shape:", X_test_base.shape)
print("GRU train shape:", X_train_seq.shape)
print("GRU test shape:", X_test_seq.shape)


# =========================================================
# BASELINE MODELS
# =========================================================
print("\nTraining baseline models...")

lr_model = LinearRegression()
lr_model.fit(X_train_base, y_train_base)
lr_pred = lr_model.predict(X_test_base)

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_model.fit(X_train_base, y_train_base)
rf_pred = rf_model.predict(X_test_base)

print("Baseline models completed.")


# =========================================================
# GRU MODEL
# =========================================================
print("\nTraining GRU model...")

gru_model = Sequential([
    GRU(32, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1)
])

gru_model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = gru_model.fit(
    X_train_seq,
    y_train_seq,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)

gru_pred_scaled = gru_model.predict(X_test_seq).flatten()
gru_pred = target_scaler.inverse_transform(
    gru_pred_scaled.reshape(-1, 1)
).flatten()

print("GRU model completed.")


# =========================================================
# EVALUATION
# =========================================================
results = []
results.append(evaluate_model("Linear Regression", y_test_base, lr_pred))
results.append(evaluate_model("Random Forest", y_test_base, rf_pred))
results.append(evaluate_model("GRU (7-day)", y_test_seq_original, gru_pred))

results_df = pd.DataFrame(results)

print("\nModel Comparison:")
print(results_df)

results_csv_path = os.path.join(OUTPUT_DIR, "rq2_model_comparison_clean.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"\nSaved model comparison CSV to: {results_csv_path}")


# =========================================================
# TIME-SERIES GRAPH
# =========================================================
plot_df = pd.DataFrame({
    "Actual Recovery Score": y_test_seq_original,
    "Predicted Recovery Score (GRU)": gru_pred
}).reset_index(drop=True)

plot_df = plot_df.tail(min(120, len(plot_df))).copy()
plot_df["Time Step"] = range(1, len(plot_df) + 1)

plt.figure(figsize=(12, 6))
plt.plot(
    plot_df["Time Step"],
    plot_df["Actual Recovery Score"],
    label="Actual Recovery Score",
    linewidth=2
)
plt.plot(
    plot_df["Time Step"],
    plot_df["Predicted Recovery Score (GRU)"],
    label="Predicted Recovery Score (GRU)",
    linewidth=2
)
plt.xlabel("Time Step")
plt.ylabel("Next-Day Recovery Score")
plt.title("Time-Series: Actual vs Predicted Next-Day Recovery Score")
plt.legend()
plt.tight_layout()

time_series_path = os.path.join(OUTPUT_DIR, "gru_time_series_actual_vs_predicted_clean.png")
plt.savefig(time_series_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved time-series graph to: {time_series_path}")


# =========================================================
# TRAINING HISTORY GRAPH
# =========================================================
plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Training Loss", linewidth=2)
plt.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GRU Training History")
plt.legend()
plt.tight_layout()

history_path = os.path.join(OUTPUT_DIR, "gru_training_history_clean.png")
plt.savefig(history_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved training history graph to: {history_path}")


# =========================================================
# METRIC CHARTS
# =========================================================
metrics_plot_df = results_df.set_index("Model")

for metric in ["MAE", "RMSE", "R2"]:
    plt.figure(figsize=(8, 5))
    plt.bar(metrics_plot_df.index, metrics_plot_df[metric])
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.title(f"Model Comparison - {metric}")
    plt.xticks(rotation=15)
    plt.tight_layout()

    metric_path = os.path.join(OUTPUT_DIR, f"model_comparison_{metric.lower()}_clean.png")
    plt.savefig(metric_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved {metric} comparison chart to: {metric_path}")

print("\nAll tasks completed successfully.")