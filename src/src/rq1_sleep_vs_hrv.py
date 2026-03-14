import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# ---------------------------------------------------
# Helper: detect columns safely
# ---------------------------------------------------
def find_col(df, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


# ---------------------------------------------------
# Task #28: Prepare dataset for sleep vs HRV analysis
# ---------------------------------------------------
def prepare_rq1_dataset(df):
    """
    Prepares dataset for RQ1:
    Deep Sleep / REM Sleep vs next-day HRV

    Expected logic:
    - detect required columns
    - convert date
    - sort by user/date if available
    - create next_day_hrv using shift(-1)
    - return filtered dataframe
    """

    date_col = find_col(df, ["date", "day", "datetime", "timestamp"])
    user_col = find_col(df, ["user_id", "userid", "user", "athlete_id", "id"])
    hrv_col = find_col(df, ["hrv", "heart_rate_variability", "hrv_ms"])
    deep_col = find_col(df, ["deep_sleep", "deep_sleep_hours", "deep_sleep_minutes", "sws"])
    rem_col = find_col(df, ["rem_sleep", "rem_sleep_hours", "rem_sleep_minutes", "rem"])

    if date_col is None:
        raise ValueError("No date column found.")
    if hrv_col is None:
        raise ValueError("No HRV column found.")
    if deep_col is None:
        raise ValueError("No Deep Sleep column found.")
    if rem_col is None:
        raise ValueError("No REM Sleep column found.")

    out = df.copy()

    # date conversion
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])

    # sort properly
    sort_cols = [date_col]
    if user_col is not None:
        sort_cols = [user_col, date_col]
    out = out.sort_values(sort_cols).reset_index(drop=True)

    # create next_day_hrv without leakage across users
    if user_col is not None:
        out["next_day_hrv"] = out.groupby(user_col)[hrv_col].shift(-1)
    else:
        out["next_day_hrv"] = out[hrv_col].shift(-1)

    # keep only needed columns
    keep_cols = [date_col, deep_col, rem_col, hrv_col, "next_day_hrv"]
    if user_col is not None:
        keep_cols.insert(1, user_col)

    out = out[keep_cols].dropna(subset=[deep_col, rem_col, "next_day_hrv"]).copy()

    return out, {
        "date_col": date_col,
        "user_col": user_col,
        "hrv_col": hrv_col,
        "deep_col": deep_col,
        "rem_col": rem_col
    }


# ---------------------------------------------------
# Task #29: Generate Deep and REM sleep vs HRV plots
# ---------------------------------------------------
def generate_sleep_hrv_plots(df_rq1, col_map, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    deep_col = col_map["deep_col"]
    rem_col = col_map["rem_col"]

    # Deep sleep vs next-day HRV
    plt.figure(figsize=(7, 5))
    plt.scatter(df_rq1[deep_col], df_rq1["next_day_hrv"], alpha=0.5)
    plt.xlabel(deep_col)
    plt.ylabel("next_day_hrv")
    plt.title("Deep Sleep vs Next-day HRV")
    plt.tight_layout()
    deep_plot_path = os.path.join(output_dir, "rq1_deep_sleep_vs_next_day_hrv.png")
    plt.savefig(deep_plot_path, dpi=300)
    plt.close()

    # REM sleep vs next-day HRV
    plt.figure(figsize=(7, 5))
    plt.scatter(df_rq1[rem_col], df_rq1["next_day_hrv"], alpha=0.5)
    plt.xlabel(rem_col)
    plt.ylabel("next_day_hrv")
    plt.title("REM Sleep vs Next-day HRV")
    plt.tight_layout()
    rem_plot_path = os.path.join(output_dir, "rq1_rem_sleep_vs_next_day_hrv.png")
    plt.savefig(rem_plot_path, dpi=300)
    plt.close()

    return deep_plot_path, rem_plot_path


# ---------------------------------------------------
# Task #30: Perform Pearson correlation tests
# ---------------------------------------------------
def run_pearson_tests(df_rq1, col_map):
    deep_col = col_map["deep_col"]
    rem_col = col_map["rem_col"]

    r_deep, p_deep = pearsonr(df_rq1[deep_col], df_rq1["next_day_hrv"])
    r_rem, p_rem = pearsonr(df_rq1[rem_col], df_rq1["next_day_hrv"])

    results = pd.DataFrame([
        {
            "sleep_stage": "Deep Sleep",
            "correlation_r": r_deep,
            "p_value": p_deep
        },
        {
            "sleep_stage": "REM Sleep",
            "correlation_r": r_rem,
            "p_value": p_rem
        }
    ])

    return results


# ---------------------------------------------------
# Task #31: Interpret sleep architecture vs HRV results
# ---------------------------------------------------
def interpret_rq1_results(results_df):
    deep_r = results_df.loc[results_df["sleep_stage"] == "Deep Sleep", "correlation_r"].values[0]
    deep_p = results_df.loc[results_df["sleep_stage"] == "Deep Sleep", "p_value"].values[0]

    rem_r = results_df.loc[results_df["sleep_stage"] == "REM Sleep", "correlation_r"].values[0]
    rem_p = results_df.loc[results_df["sleep_stage"] == "REM Sleep", "p_value"].values[0]

    interpretation_lines = []

    interpretation_lines.append("RQ1 Interpretation: Sleep Architecture vs Next-day HRV")
    interpretation_lines.append(f"Deep Sleep correlation: r = {deep_r:.6f}, p = {deep_p:.6f}")
    interpretation_lines.append(f"REM Sleep correlation: r = {rem_r:.6f}, p = {rem_p:.6f}")

    # Compare magnitude
    if abs(deep_r) > abs(rem_r):
        interpretation_lines.append("Deep Sleep shows a stronger relationship with next-day HRV than REM Sleep.")
    elif abs(rem_r) > abs(deep_r):
        interpretation_lines.append("REM Sleep shows a stronger relationship with next-day HRV than Deep Sleep.")
    else:
        interpretation_lines.append("Deep Sleep and REM Sleep show similar relationship strength with next-day HRV.")

    # Significance
    if deep_p < 0.05:
        interpretation_lines.append("The Deep Sleep relationship is statistically significant.")
    else:
        interpretation_lines.append("The Deep Sleep relationship is not statistically significant.")

    if rem_p < 0.05:
        interpretation_lines.append("The REM Sleep relationship is statistically significant.")
    else:
        interpretation_lines.append("The REM Sleep relationship is not statistically significant.")

    # Final conclusion
    if deep_p >= 0.05 and rem_p >= 0.05:
        interpretation_lines.append(
            "Overall conclusion: There is no strong statistical evidence that sleep stages independently predict next-day HRV in this dataset."
        )
    else:
        interpretation_lines.append(
            "Overall conclusion: At least one sleep stage shows a statistically meaningful relationship with next-day HRV."
        )

    return "\n".join(interpretation_lines)


# ---------------------------------------------------
# Save outputs
# ---------------------------------------------------
def save_outputs(results_df, interpretation_text, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "rq1_sleep_vs_hrv_results.csv")
    interpretation_path = os.path.join(output_dir, "rq1_sleep_vs_hrv_interpretation.txt")

    results_df.to_csv(results_path, index=False)

    with open(interpretation_path, "w", encoding="utf-8") as f:
        f.write(interpretation_text)

    return results_path, interpretation_path


# ---------------------------------------------------
# Main runner
# ---------------------------------------------------
def main():
    INPUT_FILE = "outputs/cleaned_whoop.csv"   # change if needed
    OUTPUT_DIR = "outputs"

    print(f"Loading dataset from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    print("Task #28: Preparing dataset for sleep vs HRV analysis...")
    df_rq1, col_map = prepare_rq1_dataset(df)
    print("Prepared rows:", len(df_rq1))

    print("Task #29: Generating Deep Sleep and REM Sleep plots...")
    deep_plot, rem_plot = generate_sleep_hrv_plots(df_rq1, col_map, output_dir=OUTPUT_DIR)
    print("Saved:", deep_plot)
    print("Saved:", rem_plot)

    print("Task #30: Running Pearson correlation tests...")
    results_df = run_pearson_tests(df_rq1, col_map)
    print(results_df)

    print("Task #31: Interpreting results...")
    interpretation_text = interpret_rq1_results(results_df)
    print("\n" + interpretation_text)

    results_path, interpretation_path = save_outputs(results_df, interpretation_text, output_dir=OUTPUT_DIR)
    print("Saved:", results_path)
    print("Saved:", interpretation_path)


if __name__ == "__main__":
    main()