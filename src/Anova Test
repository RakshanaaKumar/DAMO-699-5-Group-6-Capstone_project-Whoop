import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("/cleaned_whoop.csv")

# Remove missing values
df = df.dropna(subset=["activity_type", "calories_per_minute"])

# Group Summary Table
grouped = df.groupby("activity_type")["calories_per_minute"]
summary = grouped.agg(["count", "mean", "std"])

print("\nGroup sample sizes + means:")
print(summary)

# ANOVA Test
groups = [group["calories_per_minute"].values
          for name, group in df.groupby("activity_type")]

f_stat, p_value = stats.f_oneway(*groups)

print("\nANOVA Results:")
print("F-statistic =", f_stat)
print("P-value =", p_value)

#Interpretation
if p_value < 0.05:
    print("\nConclusion: Reject Null Hypothesis.")
    print("Activity intensity differs significantly across activity types.")
else:
    print("\nConclusion: Fail to Reject Null Hypothesis.")
    print("No significant difference in activity intensity across groups.")
#summary table
summary = df.groupby("activity_type")["calories_per_minute"] \
            .agg(["count", "mean", "std"])

# Order activity types by mean (highest first)
order = summary.sort_values("mean", ascending=False).index.tolist()

# Create boxplot
plt.figure(figsize=(12,6))
sns.boxplot(x='activity_type',
            y='calories_per_minute',
            data=df,
            order=order)

plt.title("Calories per Minute Across Activity Types")
plt.xlabel("Activity Type")
plt.ylabel("Calories per Minute")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
