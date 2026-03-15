# 🧠 Sleep Architecture & Recovery Prediction Using WHOOP Data
### DAMO-699 Capstone Project – Wearable Health Analytics

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green)
![Deep Learning](https://img.shields.io/badge/Deep-Learning-orange)
![Time Series](https://img.shields.io/badge/Time-Series-red)
![Status](https://img.shields.io/badge/Project-Active-success)

---

# Project Overview

Wearable health devices generate continuous physiological data such as sleep stages, recovery metrics, and activity intensity. Understanding relationships between these variables is essential for improving health monitoring and recovery optimization.

This capstone project analyzes wearable fitness data collected from WHOOP devices to explore how sleep architecture and activity intensity influence physiological recovery.

The project combines **statistical analysis, predictive modeling, and deep learning techniques** to uncover patterns in physiological signals and predict next-day recovery indicators such as Heart Rate Variability (HRV).

---

# Research Questions

**RQ1**  
How do Deep Sleep and REM Sleep durations correlate with next-day Heart Rate Variability (HRV)?

**RQ2**  
Can deep learning models predict next-day HRV using historical physiological signals?

**RQ3**  
Do different activity types exhibit significantly different metabolic intensities?

**RQ4**  
Which predictive modeling approach provides the most accurate recovery predictions?

---

# Dataset

The dataset used in this project originates from **WHOOP wearable fitness trackers**.

The dataset contains daily physiological and activity measurements including:

• user_id  
• date  
• heart rate variability (HRV)  
• recovery score  
• deep sleep duration  
• REM sleep duration  
• activity duration  
• calories burned  
• activity type  

This dataset enables analysis of relationships between **sleep architecture, activity intensity, and recovery performance**.

---

# Project Pipeline

## Project Pipeline

```mermaid
flowchart TD
A[Raw WHOOP Data] --> B[Data Cleaning]
B --> C[Feature Engineering]
C --> D[Exploratory Data Analysis]
D --> E[Statistical Analysis]
E --> F[Predictive Modeling]
F --> G[Deep Learning Model]
G --> H[Model Evaluation]
H --> I[Insights and Visualization]
```

## Machine Learning Pipeline

```mermaid
flowchart LR
A[Dataset] --> B[Preprocessing]
B --> C[Feature Engineering]
C --> D[Train Test Split]
D --> E[Baseline Model]
D --> F[Linear Regression]
D --> G[GRU Deep Learning Model]
E --> H[Model Evaluation]
F --> H
G --> H
```

## Deep Learning Model Workflow

```mermaid
flowchart TD
A[Input Time Series Data] --> B[Normalization]
B --> C[Sequence Creation]
C --> D[GRU Model]
D --> E[Training]
E --> F[Validation]
F --> G[Prediction]
G --> H[Performance Evaluation]
```

Analytical Methods Used
Statistical Analysis

• Pearson Correlation Analysis
• One-Way ANOVA
• Descriptive Statistics


These methods help determine whether relationships exist between sleep stages and physiological recovery metrics.

Predictive Modeling

Baseline Mean Model
Linear Regression Model

Baseline models provide a reference point for evaluating model performance.

Deep Learning

A GRU neural network is used to capture temporal patterns within physiological time-series data.

The model predicts next-day HRV based on previous physiological signals and sleep metrics.

Visualization

The project generates multiple visual outputs including:

• Deep Sleep vs HRV scatter plots
• REM Sleep vs HRV scatter plots
• Activity metabolic intensity comparisons
• Model performance comparison charts

These visualizations help interpret statistical results and identify physiological patterns.

DAMO-699-Capstone-project-Whoop

notebooks/
    whoop_analysis.ipynb

src/
    data_prep.py
    rq1_sleep_vs_hrv.py
    rq2_recovery_prediction.py
    rq3_metabolic_intensity.py

outputs/
    cleaned_whoop.csv
    analysis_results.csv
    figures/

README.md
requirements.txt
.gitignore

Technologies Used

Python
Pandas
NumPy
Matplotlib
SciPy
Scikit-Learn
TensorFlow / Keras

These technologies enable efficient data processing, modeling, and visualization.

Key Insights

Initial analysis indicates that individual sleep stages show weak direct correlations with next-day HRV.

However, activity intensity shows statistically significant differences across activity types.

Deep learning models capture temporal physiological patterns and demonstrate improved prediction capability compared to baseline models.

These findings highlight the complexity of physiological recovery processes and the importance of analyzing multiple interacting factors.

Future Work

Potential improvements include:

• Expanding the dataset to include more users and longer time periods
• Implementing advanced time-series models such as LSTM or Transformers
• Building interactive dashboards using Power BI or Streamlit
• Incorporating additional physiological metrics such as resting heart rate and strain score
