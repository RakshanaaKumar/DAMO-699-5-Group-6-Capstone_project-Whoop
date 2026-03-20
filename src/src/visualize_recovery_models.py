"""
Task #40: Recovery Prediction Visualizations
RQ2 - Model Performance Comparison

This script generates:
1. Model comparison charts (MAE, RMSE, R²)
2. GRU Actual vs Predicted plot
3. Scatter plot for prediction evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------
# Step 1: Model Evaluation Results (from experiment)
# ------------------------------------------------------------
results_df = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "GRU"],
    "MAE": [9.345, 10.534, 17.589],
    "RMSE": [11.701, 13.119, 20.886],
    "R2": [0.561, 0.448, 0.002]
})

# ------------------------------------------------------------
# Step 2: Save MAE Comparison Chart
# ------------------------------------------------------------
plt.figure()
plt.bar(results_df["Model"], results_df["MAE"])
plt.title("Model Comparison - MAE")
plt.xlabel("Model")
plt.ylabel("MAE")
plt.tight_layout()
plt.savefig("mae_comparison.png")
plt.close()

# ------------------------------------------------------------
# Step 3: Save RMSE Comparison Chart
# ------------------------------------------------------------
plt.figure()
plt.bar(results_df["Model"], results_df["RMSE"])
plt.title("Model Comparison - RMSE")
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.tight_layout()
plt.savefig("rmse_comparison.png")
plt.close()

# ------------------------------------------------------------
# Step 4: Save R² Comparison Chart
# ------------------------------------------------------------
plt.figure()
plt.bar(results_df["Model"], results_df["R2"])
plt.title("Model Comparison - R2 Score")
plt.xlabel("Model")
plt.ylabel("R2")
plt.tight_layout()
plt.savefig("r2_comparison.png")
plt.close()

# ------------------------------------------------------------
# Step 5: Load Actual vs Predicted Data
# (Replace with your saved arrays or file)
# ------------------------------------------------------------
# Example: load from CSV (recommended for GitHub)
try:
    df = pd.read_csv("/workspaces/DAMO-699-5-Group-6-Capstone_project-Whoop/outputs/gru_predictions.csv")
    actual = df["Actual"].values
    predicted = df["Predicted"].values
except:
    print("ERROR: Please provide 'gru_predictions.csv' with Actual and Predicted columns.")
    exit()

# ------------------------------------------------------------
# Step 6: Plot Actual vs Predicted (Time Series)
# ------------------------------------------------------------
plt.figure()
plt.plot(actual[-100:], label="Actual")
plt.plot(predicted[-100:], label="Predicted (GRU)")
plt.title("Actual vs Predicted Recovery (GRU)")
plt.xlabel("Time Step")
plt.ylabel("Recovery Score")
plt.legend()
plt.tight_layout()
plt.savefig("gru_actual_vs_predicted.png")
plt.close()

# ------------------------------------------------------------
# Step 7: Scatter Plot
# ------------------------------------------------------------
plt.figure()
plt.scatter(actual, predicted, alpha=0.5)

# Ideal line
min_val = min(actual.min(), predicted.min())
max_val = max(actual.max(), predicted.max())
plt.plot([min_val, max_val], [min_val, max_val])

plt.title("Actual vs Predicted (GRU)")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.tight_layout()
plt.savefig("gru_scatter.png")
plt.close()

# ------------------------------------------------------------
# Step 8: Print Metrics
# ------------------------------------------------------------
mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
r2 = r2_score(actual, predicted)

print("GRU Model Performance:")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R2   : {r2:.3f}")