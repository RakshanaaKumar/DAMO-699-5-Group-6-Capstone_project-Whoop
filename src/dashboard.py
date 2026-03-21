
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="WHOOP Analytics Dashboard",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .section-title {
        font-size: 1.3rem; font-weight: 700; color: #a78bfa;
        border-bottom: 1px solid #374151; padding-bottom: .3rem; margin-bottom: 1rem;
    }
    .insight-box {
        background: linear-gradient(135deg, rgba(167,139,250,0.12), rgba(52,211,153,0.08));
        border: 1px solid rgba(167,139,250,0.3);
        border-radius: 10px; padding: 14px 18px; margin-bottom: 12px;
        font-size: 0.92rem; line-height: 1.6;
    }
    .metric-badge {
        display: inline-block; background: rgba(167,139,250,0.2);
        color: #c4b5fd; border-radius: 6px; padding: 2px 10px;
        font-family: 'Space Mono', monospace; font-size: 0.82rem; font-weight: 700;
    }
    .pass-badge  { background: rgba(52,211,153,0.2);  color: #6ee7b7; }
    .fail-badge  { background: rgba(248,113,113,0.2); color: #fca5a5; }
    .warn-badge  { background: rgba(251,191,36,0.2);  color: #fcd34d; }
    .stTabs [data-baseweb="tab"] { font-family: 'DM Sans', sans-serif; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


@st.cache_data
def compute_longevity_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Derive a composite longevity score (0-100) from biometric features."""
    cols = {
        "hrv":                  ("high", 20),
        "sleep_efficiency":     ("high", 20),
        "resting_heart_rate":   ("low",  15),
        "recovery_score":       ("high", 25),
        "day_strain":           ("high", 10),
        "sleep_hours":          ("high", 10),
    }
    out = df.copy()
    score = pd.Series(0.0, index=out.index)
    for col, (direction, weight) in cols.items():
        if col not in out.columns:
            continue
        s = out[col].copy()
        s_min, s_max = s.min(), s.max()
        if s_max == s_min:
            norm = pd.Series(0.5, index=s.index)
        else:
            norm = (s - s_min) / (s_max - s_min)
        if direction == "low":
            norm = 1 - norm
        score += norm * weight
    out["longevity_score_0_100"] = score.clip(0, 100)
    return out


@st.cache_data
def compute_model_comparison() -> pd.DataFrame:
    """Static model results from rq2 / visualize_recovery_models."""
    return pd.DataFrame({
        "Model":  ["Linear Regression", "Random Forest", "GRU (7-day)"],
        "MAE":    [9.345, 10.534, 17.589],
        "RMSE":   [11.701, 13.119, 20.886],
        "R2":     [0.561,  0.448,  0.002],
    })


@st.cache_data
def run_anova(df: pd.DataFrame):
    """Reproduce Anova_Test.py logic on the loaded dataframe."""
    needed = {"activity_type", "calories_per_minute"}
    if not needed.issubset(df.columns):
        return None, None, None
    sub = df.dropna(subset=["activity_type", "calories_per_minute"])
    groups = [g["calories_per_minute"].values for _, g in sub.groupby("activity_type")]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return None, None, None
    f_stat, p_value = stats.f_oneway(*groups)
    summary = sub.groupby("activity_type")["calories_per_minute"].agg(["count", "mean", "std"])
    return f_stat, p_value, summary


@st.cache_data
def compute_calories_per_minute(df: pd.DataFrame) -> pd.DataFrame:
    """UserStory5 logic: compute and rank activity metabolic intensity."""
    out = df.copy()
    cal_col = "activity_calories" if "activity_calories" in out.columns else "calories_burned"
    dur_col = "activity_duration_min" if "activity_duration_min" in out.columns else None
    if dur_col is None:
        return pd.DataFrame()
    out[dur_col] = pd.to_numeric(out[dur_col], errors="coerce")
    out[cal_col] = pd.to_numeric(out[cal_col], errors="coerce")
    out["calories_per_minute"] = np.where(
        (out[dur_col].notna()) & (out[dur_col] > 0) & (out[cal_col].notna()),
        out[cal_col] / out[dur_col],
        np.nan
    )
    return out


@st.cache_data
def run_pearson_rq1(df: pd.DataFrame) -> pd.DataFrame:
    """rq1_sleep_vs_hrv: Pearson correlation of sleep stages vs next-day HRV."""
    needed = {"hrv", "deep_sleep_hours", "rem_sleep_hours"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    out = df.sort_values("date").copy()
    if "user_id" in out.columns:
        out["next_day_hrv"] = out.groupby("user_id")["hrv"].shift(-1)
    else:
        out["next_day_hrv"] = out["hrv"].shift(-1)
    sub = out.dropna(subset=["deep_sleep_hours", "rem_sleep_hours", "next_day_hrv"])
    if len(sub) < 10:
        return pd.DataFrame()
    r_deep, p_deep = stats.pearsonr(sub["deep_sleep_hours"], sub["next_day_hrv"])
    r_rem,  p_rem  = stats.pearsonr(sub["rem_sleep_hours"],  sub["next_day_hrv"])
    return pd.DataFrame([
        {"sleep_stage": "Deep Sleep", "correlation_r": round(r_deep, 4), "p_value": round(p_deep, 6)},
        {"sleep_stage": "REM Sleep",  "correlation_r": round(r_rem,  4), "p_value": round(p_rem,  6)},
    ])


@st.cache_data
def run_data_validation(df: pd.DataFrame) -> dict:
    """data_validation_py logic: physiological range + missing checks."""
    if "sleep_hours" not in df.columns:
        return {}
    d = df.copy()
    d["sleep_minutes"] = d["sleep_hours"] * 60
    invalid = pd.Series(False, index=d.index)
    checks = {}
    if "day_strain" in d.columns:
        checks["day_strain out of [0,21]"] = (d["day_strain"] < 0) | (d["day_strain"] > 21)
    if "recovery_score" in d.columns:
        checks["recovery_score out of [0,100]"] = (d["recovery_score"] < 0) | (d["recovery_score"] > 100)
    if "hrv" in d.columns:
        checks["HRV out of [5,300]"] = (d["hrv"] < 5) | (d["hrv"] > 300)
    if "resting_heart_rate" in d.columns:
        checks["RHR out of [30,120]"] = (d["resting_heart_rate"] < 30) | (d["resting_heart_rate"] > 120)
    checks["Sleep out of [0,960] min"] = (d["sleep_minutes"] < 0) | (d["sleep_minutes"] > 960)
    if "max_heart_rate" in d.columns:
        checks["Max HR out of [60,220]"] = (d["max_heart_rate"] < 60) | (d["max_heart_rate"] > 220)
    if "avg_heart_rate" in d.columns:
        checks["Avg HR out of [40,180]"] = (d["avg_heart_rate"] < 40) | (d["avg_heart_rate"] > 180)
    for mask in checks.values():
        invalid |= mask
    critical_cols = [c for c in ["day_strain","recovery_score","hrv","resting_heart_rate","sleep_hours"] if c in d.columns]
    missing_critical = d[critical_cols].isna().any(axis=1)
    return {
        "total":            len(d),
        "invalid_physio":   int(invalid.sum()),
        "missing_critical": int(missing_critical.sum()),
        "model_ready":      int((~invalid & ~missing_critical).sum()),
        "checks":           {k: int(v.sum()) for k, v in checks.items()},
    }


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("🔧 Filters")

data_path = st.sidebar.text_input(
    "CSV path",
    value="/Users/nickyl/Documents/GitHub/DAMO-699-5-Group-6-Capstone_project-Whoop/outputs/cleaned_whoop.csv",
    help="Absolute path to cleaned_whoop.csv"
)

try:
    df_raw = load_data(data_path)
    data_loaded = True
except Exception as e:
    st.sidebar.error(f"Could not load data: {e}")
    data_loaded = False
    df_raw = pd.DataFrame()

if data_loaded:
    users = ["All"] + sorted(df_raw["user_id"].unique().tolist()) if "user_id" in df_raw.columns else ["All"]
    selected_user = st.sidebar.selectbox("User", users)

    date_min, date_max = df_raw["date"].min(), df_raw["date"].max()
    start_date, end_date = st.sidebar.date_input(
        "Date range", value=[date_min, date_max], min_value=date_min, max_value=date_max
    )

    # When a specific user is selected, freeze Fitness Level & Activity Type
    # (a single user has exactly one fitness level and filtering by activity would
    #  drop most of their data — show an informational note instead)
    user_selected = selected_user != "All"

    if user_selected:
        #st.sidebar.info(
          #  "🔒 **Fitness Level** and **Activity Type** filters are disabled when a specific user is selected."
        #)
        sel_fitness  = "All"
        sel_activity = "All"
    else:
        fitness_levels = ["All"] + sorted(df_raw["fitness_level"].dropna().unique().tolist()) if "fitness_level" in df_raw.columns else ["All"]
        sel_fitness = st.sidebar.selectbox("Fitness Level", fitness_levels)

        activities = ["All"] + sorted(df_raw["activity_type"].dropna().unique().tolist()) if "activity_type" in df_raw.columns else ["All"]
        sel_activity = st.sidebar.selectbox("Activity Type", activities)

    # Apply filters
    fdf = df_raw.copy()
    if selected_user != "All" and "user_id" in fdf.columns:
        fdf = fdf[fdf["user_id"] == selected_user]
    fdf = fdf[(fdf["date"] >= pd.Timestamp(start_date)) & (fdf["date"] <= pd.Timestamp(end_date))]
    if sel_fitness != "All" and "fitness_level" in fdf.columns:
        fdf = fdf[fdf["fitness_level"] == sel_fitness]
    if sel_activity != "All" and "activity_type" in fdf.columns:
        fdf = fdf[fdf["activity_type"] == sel_activity]

    workout_df = fdf[fdf["workout_completed"] == 1] if "workout_completed" in fdf.columns else fdf.copy()
    dow_order  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    # Pre-compute derived datasets
    fdf_with_cpm    = compute_calories_per_minute(fdf)
    fdf_with_long   = compute_longevity_scores(fdf)
    pearson_results = run_pearson_rq1(fdf)
    model_results   = compute_model_comparison()
    validation_info = run_data_validation(fdf)
    f_stat, p_value, anova_summary = run_anova(fdf_with_cpm if not fdf_with_cpm.empty else fdf)
else:
    fdf = workout_df = pd.DataFrame()
    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title(" WHOOP Analytics Dashboard")
if data_loaded:
    st.caption(f"Showing **{len(fdf):,}** records · {start_date} → {end_date}")
else:
    st.warning("⚠️ Please provide a valid path to `cleaned_whoop.csv` in the sidebar.")
    st.stop()

# ─────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Avg Recovery",  f"{fdf['recovery_score'].mean():.1f}"   if "recovery_score"  in fdf.columns else "—")
k2.metric("Avg Sleep",     f"{fdf['sleep_hours'].mean():.1f} h"    if "sleep_hours"     in fdf.columns else "—")
k3.metric("Avg Calories",  f"{fdf['calories_burned'].mean():,.0f}" if "calories_burned" in fdf.columns else "—")
k4.metric("Avg HRV",       f"{fdf['hrv'].mean():.1f} ms"           if "hrv"             in fdf.columns else "—")
k5.metric("Workouts",      f"{len(workout_df):,}")
st.markdown("---")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
(tab_sleep, tab_cal, tab_cardio, tab_act, tab_perf,
 tab_rq1, tab_rq2, tab_long, tab_valid) = st.tabs([
    "😴 Sleep",
    "🔥 Calories",
    "❤️ Cardiovascular",
    "🏃 Activities",
    "📈 Performance",
    "🔬 RQ1 – Sleep vs HRV",
    "🤖 RQ2 – Recovery Models",
    "🌿 Longevity Trends",
    "✅ Data Validation",
])

# ══════════════════════════════════════════════
# TAB: SLEEP
# ══════════════════════════════════════════════
with tab_sleep:
    st.markdown('<p class="section-title">Sleep Behaviour Analysis</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    if all(c in fdf.columns for c in ["light_sleep_hours","rem_sleep_hours","deep_sleep_hours"]):
        sleep_time = (fdf.set_index("date")[["light_sleep_hours","rem_sleep_hours","deep_sleep_hours"]]
                      .resample("W").mean().reset_index())
        fig = px.area(sleep_time, x="date",
                      y=["light_sleep_hours","rem_sleep_hours","deep_sleep_hours"],
                      title="Weekly Avg Sleep Stages",
                      color_discrete_sequence=["#60a5fa","#a78bfa","#34d399"])
        fig.update_layout(template="plotly_dark", hovermode="x unified")
        c1.plotly_chart(fig, use_container_width=True)

    if "sleep_efficiency" in fdf.columns:
        fig2 = px.histogram(fdf, x="sleep_efficiency", nbins=30,
                            color="fitness_level" if "fitness_level" in fdf.columns else None,
                            title="Sleep Efficiency by Fitness Level",
                            barmode="overlay", opacity=0.75,
                            color_discrete_sequence=px.colors.qualitative.Vivid)
        fig2.update_layout(template="plotly_dark")
        c2.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    if "day_of_week" in fdf.columns and "sleep_hours" in fdf.columns:
        sleep_dow = fdf.groupby("day_of_week")["sleep_hours"].mean().reindex(dow_order).reset_index()
        fig3 = px.bar(sleep_dow, x="day_of_week", y="sleep_hours",
                      title="Avg Sleep Hours by Day of Week",
                      color="sleep_hours", color_continuous_scale="Purples")
        fig3.update_layout(template="plotly_dark", coloraxis_showscale=False)
        c3.plotly_chart(fig3, use_container_width=True)

    if all(c in fdf.columns for c in ["time_to_fall_asleep_min","wake_ups","recovery_score"]):
        fig4 = px.scatter(fdf.sample(min(3000, len(fdf))),
                          x="time_to_fall_asleep_min", y="wake_ups",
                          color="recovery_score", title="Time to Fall Asleep vs Wake-ups",
                          color_continuous_scale="RdYlGn", opacity=0.6)
        fig4.update_layout(template="plotly_dark")
        c4.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════
# TAB: CALORIES
# ══════════════════════════════════════════════
with tab_cal:
    st.markdown('<p class="section-title">Calorie Burn Analysis</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    if "calories_burned" in fdf.columns:
        cal_trend = fdf.set_index("date")["calories_burned"].resample("W").mean().reset_index()
        fig = px.line(cal_trend, x="date", y="calories_burned",
                      title="Weekly Avg Calories Burned",
                      color_discrete_sequence=["#f97316"])
        fig.update_traces(fill="tozeroy", fillcolor="rgba(249,115,22,0.15)")
        fig.update_layout(template="plotly_dark")
        c1.plotly_chart(fig, use_container_width=True)

    if "activity_calories" in workout_df.columns and "activity_type" in workout_df.columns:
        act_cal = (workout_df.groupby("activity_type")["activity_calories"]
                   .mean().sort_values(ascending=False).reset_index())
        fig2 = px.bar(act_cal, x="activity_calories", y="activity_type", orientation="h",
                      title="Avg Activity Calories by Sport",
                      color="activity_calories", color_continuous_scale="YlOrRd")
        fig2.update_layout(template="plotly_dark", coloraxis_showscale=False)
        c2.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    if all(c in fdf.columns for c in ["day_strain","calories_burned"]):
        fig3 = px.scatter(fdf.sample(min(3000, len(fdf))),
                          x="day_strain", y="calories_burned",
                          color="fitness_level" if "fitness_level" in fdf.columns else None,
                          trendline="ols",
                          title="Day Strain vs Calories Burned", opacity=0.55,
                          color_discrete_sequence=px.colors.qualitative.Pastel)
        fig3.update_layout(template="plotly_dark")
        c3.plotly_chart(fig3, use_container_width=True)

    if all(c in fdf.columns for c in ["fitness_level","calories_burned","gender"]):
        fig4 = px.box(fdf, x="fitness_level", y="calories_burned", color="gender",
                      title="Calories: Fitness Level × Gender",
                      color_discrete_sequence=["#818cf8","#f472b6"])
        fig4.update_layout(template="plotly_dark")
        c4.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════
# TAB: CARDIOVASCULAR
# ══════════════════════════════════════════════
with tab_cardio:
    st.markdown('<p class="section-title">Cardiovascular Health & Projections</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    if "hrv" in fdf.columns:
        hrv_trend = fdf.set_index("date")["hrv"].resample("W").mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hrv_trend["date"], y=hrv_trend["hrv"],
                                 name="Weekly HRV", line=dict(color="#34d399")))
        if "hrv_baseline" in fdf.columns:
            baseline = fdf["hrv_baseline"].mean()
            fig.add_hline(y=baseline, line_dash="dot", line_color="#f87171",
                          annotation_text=f"Baseline {baseline:.0f}")
        fig.update_layout(title="HRV Trend vs Baseline", template="plotly_dark", hovermode="x unified")
        c1.plotly_chart(fig, use_container_width=True)

    if "resting_heart_rate" in fdf.columns:
        rhr_trend = fdf.set_index("date")["resting_heart_rate"].resample("W").mean().reset_index()
        fig2 = px.line(rhr_trend, x="date", y="resting_heart_rate",
                       title="Weekly Avg Resting Heart Rate",
                       color_discrete_sequence=["#f87171"])
        fig2.update_layout(template="plotly_dark")
        c2.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    hr_zone_cols = [c for c in ["hr_zone_1_min","hr_zone_2_min","hr_zone_3_min","hr_zone_4_min","hr_zone_5_min"] if c in workout_df.columns]
    if hr_zone_cols:
        zones = workout_df[hr_zone_cols].mean()
        zone_labels = ["Zone 1\n(<50%)","Zone 2\n(50-60%)","Zone 3\n(60-70%)","Zone 4\n(70-80%)","Zone 5\n(>80%)"][:len(hr_zone_cols)]
        fig3 = px.bar(x=zone_labels, y=zones.values,
                      title="Avg Minutes per HR Zone (Workouts)",
                      color=zones.values, color_continuous_scale="RdYlGn_r",
                      labels={"x":"Zone","y":"Minutes"})
        fig3.update_layout(template="plotly_dark", coloraxis_showscale=False)
        c3.plotly_chart(fig3, use_container_width=True)

    if all(c in fdf.columns for c in ["resting_heart_rate","hrv","recovery_score"]):
        fig4 = px.scatter(fdf.sample(min(3000, len(fdf))),
                          x="resting_heart_rate", y="hrv",
                          color="recovery_score", title="HRV vs Resting HR",
                          color_continuous_scale="RdYlGn", opacity=0.5)
        fig4.update_layout(template="plotly_dark")
        c4.plotly_chart(fig4, use_container_width=True)

    if all(c in fdf.columns for c in ["respiratory_rate","fitness_level","gender"]):
        fig5 = px.violin(fdf, y="respiratory_rate", x="fitness_level", color="gender",
                         box=True, title="Respiratory Rate by Fitness Level & Gender",
                         color_discrete_sequence=["#818cf8","#f472b6"])
        fig5.update_layout(template="plotly_dark")
        st.plotly_chart(fig5, use_container_width=True)

# ══════════════════════════════════════════════
# TAB: ACTIVITIES
# ══════════════════════════════════════════════
with tab_act:
    st.markdown('<p class="section-title">Activity Breakdown</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    if "activity_type" in workout_df.columns:
        act_counts = workout_df["activity_type"].value_counts().reset_index()
        act_counts.columns = ["activity_type","count"]
        fig = px.pie(act_counts, names="activity_type", values="count",
                     title="Activity Type Distribution", hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.update_layout(template="plotly_dark")
        c1.plotly_chart(fig, use_container_width=True)

    if all(c in workout_df.columns for c in ["activity_type","activity_duration_min"]):
        fig2 = px.box(workout_df, x="activity_type", y="activity_duration_min",
                      color="fitness_level" if "fitness_level" in workout_df.columns else None,
                      title="Workout Duration by Activity & Fitness Level",
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        fig2.update_layout(template="plotly_dark", xaxis_tickangle=-30)
        c2.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    if all(c in workout_df.columns for c in ["day_of_week","activity_type","activity_strain"]):
        heat_data = (workout_df.groupby(["day_of_week","activity_type"])["activity_strain"]
                     .mean().unstack(fill_value=0).reindex(dow_order))
        fig3 = px.imshow(heat_data, title="Avg Strain: Day × Activity",
                         color_continuous_scale="Plasma", aspect="auto")
        fig3.update_layout(template="plotly_dark")
        c3.plotly_chart(fig3, use_container_width=True)

    if "day_of_week" in workout_df.columns:
        freq_day = workout_df["day_of_week"].value_counts().reindex(dow_order).reset_index()
        freq_day.columns = ["day_of_week","count"]
        fig4 = px.bar(freq_day, x="day_of_week", y="count",
                      title="Workout Frequency by Day",
                      color="count", color_continuous_scale="Blues")
        fig4.update_layout(template="plotly_dark", coloraxis_showscale=False)
        c4.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════
# TAB: PERFORMANCE
# ══════════════════════════════════════════════
with tab_perf:
    st.markdown('<p class="section-title">Overall Performance Overview</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    if "recovery_score" in fdf.columns:
        rec_trend = fdf.set_index("date")["recovery_score"].resample("W").mean().reset_index()
        fig = px.line(rec_trend, x="date", y="recovery_score",
                      title="Weekly Avg Recovery Score",
                      color_discrete_sequence=["#a78bfa"])
        fig.add_hrect(y0=67, y1=100, fillcolor="green",  opacity=0.07, line_width=0, annotation_text="Green")
        fig.add_hrect(y0=34, y1=67,  fillcolor="yellow", opacity=0.07, line_width=0, annotation_text="Yellow")
        fig.add_hrect(y0=0,  y1=34,  fillcolor="red",    opacity=0.07, line_width=0, annotation_text="Red")
        fig.update_layout(template="plotly_dark")
        c1.plotly_chart(fig, use_container_width=True)

    if all(c in fdf.columns for c in ["fitness_level","recovery_score","gender"]):
        fig2 = px.box(fdf, x="fitness_level", y="recovery_score", color="gender",
                      title="Recovery Score by Fitness Level & Gender",
                      color_discrete_sequence=["#818cf8","#f472b6"])
        fig2.update_layout(template="plotly_dark")
        c2.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    if all(c in fdf.columns for c in ["recovery_score","day_strain","sleep_hours"]):
        fig3 = px.scatter(fdf.sample(min(3000, len(fdf))),
                          x="recovery_score", y="day_strain",
                          color="sleep_hours", title="Recovery vs Day Strain (coloured by Sleep)",
                          color_continuous_scale="Viridis", opacity=0.55)
        fig3.update_layout(template="plotly_dark")
        c3.plotly_chart(fig3, use_container_width=True)

    corr_candidates = ["recovery_score","day_strain","sleep_hours","sleep_efficiency",
                       "hrv","resting_heart_rate","calories_burned","activity_strain"]
    corr_cols = [c for c in corr_candidates if c in fdf.columns]
    if len(corr_cols) >= 3:
        corr = fdf[corr_cols].corr()
        fig4 = px.imshow(corr, title="Feature Correlation Matrix",
                         color_continuous_scale="RdBu", zmin=-1, zmax=1, text_auto=".2f")
        fig4.update_layout(template="plotly_dark")
        c4.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════
# TAB: RQ1 – Sleep vs HRV
# ══════════════════════════════════════════════
with tab_rq1:
    st.markdown('<p class="section-title">RQ1 · Sleep Architecture vs Next-Day HRV</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    Investigates whether <b>Deep Sleep</b> and <b>REM Sleep</b> duration independently predict next-day
    Heart Rate Variability (HRV). Next-day HRV is computed via a <code>shift(-1)</code> per user,
    preventing data leakage across users. Pearson correlation is used for significance testing.
    </div>
    """, unsafe_allow_html=True)

    if not pearson_results.empty:
        fig_r = px.bar(
            pearson_results, x="sleep_stage", y="correlation_r",
            color="sleep_stage", title="Pearson r: Sleep Stage vs Next-Day HRV",
            color_discrete_sequence=["#34d399","#a78bfa"],
            text=pearson_results["correlation_r"].apply(lambda x: f"{x:.4f}")
        )
        fig_r.update_traces(textposition="outside")
        fig_r.update_layout(template="plotly_dark", showlegend=False, yaxis_title="Pearson r")
        st.plotly_chart(fig_r, use_container_width=True)

        fig_p = px.bar(
            pearson_results, x="sleep_stage", y="p_value",
            color="sleep_stage", title="P-Values (dashed line = α=0.05 significance threshold)",
            color_discrete_sequence=["#34d399","#a78bfa"],
            text=pearson_results["p_value"].apply(lambda x: f"{x:.4f}")
        )
        fig_p.update_traces(textposition="outside")
        fig_p.add_hline(y=0.05, line_dash="dash", line_color="#f87171", annotation_text="α = 0.05")
        fig_p.update_layout(template="plotly_dark", showlegend=False, yaxis_title="p-value")
        st.plotly_chart(fig_p, use_container_width=True)

        # Scatter plots
        if all(c in fdf.columns for c in ["deep_sleep_hours","rem_sleep_hours","hrv"]):
            out = fdf.sort_values("date").copy()
            if "user_id" in out.columns:
                out["next_day_hrv"] = out.groupby("user_id")["hrv"].shift(-1)
            else:
                out["next_day_hrv"] = out["hrv"].shift(-1)
            out = out.dropna(subset=["deep_sleep_hours","rem_sleep_hours","next_day_hrv"])

            c1, c2 = st.columns(2)
            fig_ds = px.scatter(out.sample(min(3000, len(out))),
                                x="deep_sleep_hours", y="next_day_hrv",
                                trendline="ols", opacity=0.5,
                                title="Deep Sleep vs Next-Day HRV",
                                color_discrete_sequence=["#34d399"])
            fig_ds.update_layout(template="plotly_dark")
            c1.plotly_chart(fig_ds, use_container_width=True)

            fig_rem = px.scatter(out.sample(min(3000, len(out))),
                                 x="rem_sleep_hours", y="next_day_hrv",
                                 trendline="ols", opacity=0.5,
                                 title="REM Sleep vs Next-Day HRV",
                                 color_discrete_sequence=["#a78bfa"])
            fig_rem.update_layout(template="plotly_dark")
            c2.plotly_chart(fig_rem, use_container_width=True)

        st.markdown("#### 📋 Statistical Interpretation")
        for _, row in pearson_results.iterrows():
            sig    = "✅ Statistically significant" if row["p_value"] < 0.05 else "⚠️ Not statistically significant"
            direct = "positive" if row["correlation_r"] > 0 else "negative"
            st.markdown(f"""
            <div class="insight-box">
            <b>{row['sleep_stage']}</b> &nbsp;
            <span class="metric-badge">r = {row['correlation_r']:.4f}</span> &nbsp;
            <span class="metric-badge">p = {row['p_value']:.4f}</span><br>
            {sig} ({direct} correlation with next-day HRV).
            </div>
            """, unsafe_allow_html=True)

        deeper = pearson_results.loc[pearson_results["correlation_r"].abs().idxmax(), "sleep_stage"]
        st.markdown(f'<div class="insight-box">🏆 <b>{deeper}</b> shows the stronger relationship with next-day HRV.</div>', unsafe_allow_html=True)
        st.dataframe(pearson_results, use_container_width=True)
    else:
        st.info("Required columns (hrv, deep_sleep_hours, rem_sleep_hours) not found in dataset.")

# ══════════════════════════════════════════════
# TAB: RQ2 – Recovery Models
# ══════════════════════════════════════════════
with tab_rq2:
    st.markdown('<p class="section-title">RQ2 · Next-Day Recovery Prediction – Model Comparison</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    Three models were trained to predict <b>next-day recovery score</b> from current-day biometrics
    (HRV, sleep efficiency, day strain, sleep stages, RHR, etc.) using a 7-day sliding window for the GRU.
    All models use a strict time-based train/test split (80/20) to prevent leakage.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Model Performance Summary")
    st.dataframe(
        model_results.style
            .highlight_min(subset=["MAE","RMSE"], color="#14532d")
            .highlight_max(subset=["R2"],         color="#14532d"),
        use_container_width=True
    )

    colors = ["#60a5fa","#34d399","#f97316"]
    c1, c2, c3 = st.columns(3)

    fig_mae = px.bar(model_results, x="Model", y="MAE", title="MAE ↓ lower = better",
                     color="Model", color_discrete_sequence=colors,
                     text=model_results["MAE"].apply(lambda x: f"{x:.3f}"))
    fig_mae.update_traces(textposition="outside")
    fig_mae.update_layout(template="plotly_dark", showlegend=False)
    c1.plotly_chart(fig_mae, use_container_width=True)

    fig_rmse = px.bar(model_results, x="Model", y="RMSE", title="RMSE ↓ lower = better",
                      color="Model", color_discrete_sequence=colors,
                      text=model_results["RMSE"].apply(lambda x: f"{x:.3f}"))
    fig_rmse.update_traces(textposition="outside")
    fig_rmse.update_layout(template="plotly_dark", showlegend=False)
    c2.plotly_chart(fig_rmse, use_container_width=True)

    fig_r2 = px.bar(model_results, x="Model", y="R2", title="R² ↑ higher = better",
                    color="Model", color_discrete_sequence=colors,
                    text=model_results["R2"].apply(lambda x: f"{x:.3f}"))
    fig_r2.update_traces(textposition="outside")
    fig_r2.update_layout(template="plotly_dark", showlegend=False)
    c3.plotly_chart(fig_r2, use_container_width=True)

    # ── Chart 1: Grouped metric comparison (MAE & RMSE side-by-side per model)
    st.subheader("📊 Error Metric Comparison (MAE vs RMSE per Model)")
    mr_melt = model_results.melt(id_vars="Model", value_vars=["MAE","RMSE"],
                                  var_name="Metric", value_name="Value")
    fig_grouped = px.bar(
        mr_melt, x="Model", y="Value", color="Metric", barmode="group",
        title="MAE vs RMSE per Model (lower = better)",
        color_discrete_map={"MAE": "#60a5fa", "RMSE": "#f97316"},
        text=mr_melt["Value"].apply(lambda x: f"{x:.2f}")
    )
    fig_grouped.update_traces(textposition="outside")
    fig_grouped.update_layout(template="plotly_dark", yaxis_title="Error Value",
                               legend_title="Metric")
    st.plotly_chart(fig_grouped, use_container_width=True)

    # ── Chart 2: R² vs MAE bubble chart — size = RMSE (larger bubble = worse fit)
    st.subheader("🔵 R² vs MAE Bubble Chart (bubble size = RMSE)")
    fig_bubble = px.scatter(
        model_results, x="MAE", y="R2", size="RMSE", color="Model",
        color_discrete_sequence=colors,
        text="Model",
        title="Model Efficiency — R² vs MAE (bubble size = RMSE; top-left = best)",
        size_max=60,
        labels={"MAE": "MAE (lower = better)", "R2": "R² (higher = better)"}
    )
    fig_bubble.update_traces(textposition="top center")
    # Add quadrant lines at mean MAE and mean R2
    mean_mae = model_results["MAE"].mean()
    mean_r2  = model_results["R2"].mean()
    fig_bubble.add_vline(x=mean_mae, line_dash="dot", line_color="#6b7280",
                         annotation_text="Avg MAE", annotation_position="top right")
    fig_bubble.add_hline(y=mean_r2, line_dash="dot", line_color="#6b7280",
                         annotation_text="Avg R²", annotation_position="bottom right")
    fig_bubble.update_layout(template="plotly_dark", showlegend=True)
    st.plotly_chart(fig_bubble, use_container_width=True)

    # ── Improved GRU Actual vs Predicted chart
    st.subheader("📈 GRU – Actual vs Predicted Recovery Score")
    st.caption("Simulated using model RMSE statistics. Swap in real `gru_predictions.csv` for production results.")
    if "recovery_score" in fdf.columns and len(fdf) > 50:
        sample = fdf[["date","recovery_score"]].dropna().tail(120).copy().reset_index(drop=True)
        np.random.seed(42)
        gru_rmse = model_results[model_results["Model"] == "GRU (7-day)"]["RMSE"].values[0]

        # Simulate prediction with a slight lag-bias to mimic real GRU underfitting
        noise       = np.random.normal(0, gru_rmse * 0.45, len(sample))
        trend_bias  = np.linspace(-3, 3, len(sample))   # mild drift typical of GRU
        sample["gru_predicted"] = (sample["recovery_score"] + noise + trend_bias).clip(0, 100)
        sample["residual"]      = sample["recovery_score"] - sample["gru_predicted"]
        sample["abs_error"]     = sample["residual"].abs()
        sample["rolling_mae"]   = sample["abs_error"].rolling(14, min_periods=1).mean()

        # Upper/lower confidence band (±1 RMSE around prediction)
        sample["upper"] = (sample["gru_predicted"] + gru_rmse).clip(0, 100)
        sample["lower"] = (sample["gru_predicted"] - gru_rmse).clip(0, 100)

        # Row 1: time-series with confidence band
        fig_ts = go.Figure()
        # Shaded error band
        fig_ts.add_trace(go.Scatter(
            x=pd.concat([sample["date"], sample["date"].iloc[::-1]]),
            y=pd.concat([sample["upper"], sample["lower"].iloc[::-1]]),
            fill="toself", fillcolor="rgba(249,115,22,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="±1 RMSE band", hoverinfo="skip"
        ))
        # Predicted line
        fig_ts.add_trace(go.Scatter(
            x=sample["date"], y=sample["gru_predicted"],
            name="GRU Predicted", mode="lines",
            line=dict(color="#f97316", width=2, dash="dash")
        ))
        # Actual line
        fig_ts.add_trace(go.Scatter(
            x=sample["date"], y=sample["recovery_score"],
            name="Actual Recovery", mode="lines",
            line=dict(color="#a78bfa", width=2.5)
        ))
        fig_ts.update_layout(
            template="plotly_dark",
            title="Actual vs Predicted Recovery Score — GRU (with ±1 RMSE confidence band)",
            hovermode="x unified", xaxis_title="Date", yaxis_title="Recovery Score (0–100)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    best_model = model_results.sort_values("R2", ascending=False).iloc[0]
    st.markdown(f"""
    <div class="insight-box">
    🏆 <b>Best model by R²:</b> <span class="metric-badge">{best_model['Model']}</span>
    — R² = <span class="metric-badge">{best_model['R2']}</span>,
    MAE = <span class="metric-badge">{best_model['MAE']}</span>,
    RMSE = <span class="metric-badge">{best_model['RMSE']}</span>.<br><br>
    The GRU model achieves near-zero R² (0.002), suggesting 7-day temporal sequences alone do not strongly
    predict next-day recovery. Linear Regression dominates — biometric signals (HRV, sleep efficiency,
    strain) carry mostly linear predictive power for recovery.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB: LONGEVITY TRENDS
# ══════════════════════════════════════════════
with tab_long:
    st.markdown('<p class="section-title">🌿 Longevity Score Analysis (RQ4)</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    The <b>Longevity Score (0–100)</b> is a weighted composite of HRV, sleep efficiency,
    resting heart rate, recovery score, day strain, and sleep hours.
    Higher scores reflect better long-term cardiovascular and recovery health markers.
    </div>
    """, unsafe_allow_html=True)

    ls_df = fdf_with_long.copy()
    score_col = "longevity_score_0_100"

    if score_col in ls_df.columns:
        c1, c2 = st.columns(2)

        fig_hist = px.histogram(ls_df, x=score_col, nbins=40,
                                title="Distribution of Longevity Score",
                                color_discrete_sequence=["#34d399"], opacity=0.85)
        fig_hist.update_layout(template="plotly_dark",
                               xaxis_title="Longevity Score (0–100)", yaxis_title="Frequency")
        c1.plotly_chart(fig_hist, use_container_width=True)

        if "fitness_level" in ls_df.columns:
            fig_box = px.box(ls_df, x="fitness_level", y=score_col, color="fitness_level",
                             title="Longevity Score by Fitness Level",
                             color_discrete_sequence=px.colors.qualitative.Vivid)
        else:
            fig_box = px.box(ls_df, y=score_col, title="Longevity Score Distribution",
                             color_discrete_sequence=["#a78bfa"])
        fig_box.update_layout(template="plotly_dark")
        c2.plotly_chart(fig_box, use_container_width=True)

        # Trend
        st.subheader("📅 Daily Avg Longevity Score Trend")
        trend_daily = ls_df.groupby(ls_df["date"].dt.date)[score_col].mean().reset_index()
        trend_daily.columns = ["date", score_col]
        trend_daily["date"] = pd.to_datetime(trend_daily["date"])
        trend_daily["rolling_30d"] = trend_daily[score_col].rolling(30, min_periods=1).mean()

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=trend_daily["date"], y=trend_daily[score_col],
                                       name="Daily Score", mode="lines",
                                       line=dict(color="#34d399", width=1), opacity=0.5))
        fig_trend.add_trace(go.Scatter(x=trend_daily["date"], y=trend_daily["rolling_30d"],
                                       name="30-Day Rolling Avg", mode="lines",
                                       line=dict(color="#f97316", width=2.5)))
        fig_trend.update_layout(template="plotly_dark",
                                title="Daily Average Longevity Score Trend",
                                xaxis_title="Date", yaxis_title="Longevity Score (0–100)",
                                hovermode="x unified")
        st.plotly_chart(fig_trend, use_container_width=True)

        # Component weights
        c1, c2 = st.columns(2)
        comp_df = pd.DataFrame({
            "Component": ["Recovery Score", "HRV", "Sleep Efficiency",
                          "Resting HR (inv)", "Day Strain", "Sleep Hours"],
            "Weight": [25, 20, 20, 15, 10, 10]
        })
        fig_comp = px.pie(comp_df, names="Component", values="Weight",
                          title="Longevity Score Component Weights", hole=0.45,
                          color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_comp.update_layout(template="plotly_dark")
        c1.plotly_chart(fig_comp, use_container_width=True)

        # Avg score by fitness level
        if "fitness_level" in ls_df.columns:
            avg_by_fitness = ls_df.groupby("fitness_level")[score_col].mean().sort_values(ascending=False).reset_index()
            fig_fl = px.bar(avg_by_fitness, x="fitness_level", y=score_col,
                            title="Avg Longevity Score by Fitness Level",
                            color=score_col, color_continuous_scale="Viridis",
                            text=avg_by_fitness[score_col].apply(lambda x: f"{x:.1f}"))
            fig_fl.update_traces(textposition="outside")
            fig_fl.update_layout(template="plotly_dark", coloraxis_showscale=False)
            c2.plotly_chart(fig_fl, use_container_width=True)

        st.subheader("📊 Summary Statistics")
        st.dataframe(pd.DataFrame(ls_df[score_col].describe().rename("Longevity Score")).T, use_container_width=True)
    else:
        st.warning("Could not compute longevity score — check required columns exist.")

# ══════════════════════════════════════════════
# TAB: DATA VALIDATION
# ══════════════════════════════════════════════
with tab_valid:
    st.markdown('<p class="section-title">✅ Data Quality & Validation Report</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    Data quality summary with row-level counts for physiological violations and missing critical fields.
    One-way ANOVA tests metabolic intensity differences across activity types.
    </div>
    """, unsafe_allow_html=True)

    if validation_info:
        v = validation_info
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Total Rows",          f"{v['total']:,}")
        v2.metric("Invalid Physio Rows", f"{v['invalid_physio']:,}",
                  delta=f"-{v['invalid_physio']/max(v['total'],1)*100:.1f}%", delta_color="inverse")
        v3.metric("Missing Critical",    f"{v['missing_critical']:,}",
                  delta=f"-{v['missing_critical']/max(v['total'],1)*100:.1f}%", delta_color="inverse")
        v4.metric("Model-Ready Rows",    f"{v['model_ready']:,}",
                  delta=f"{v['model_ready']/max(v['total'],1)*100:.1f}% usable")

        st.markdown("---")

    # ANOVA section
    st.markdown("---")
    st.subheader("📊 ANOVA – Metabolic Intensity Across Activity Types")
    st.markdown("""
    <div class="insight-box">
    One-way ANOVA tests whether <b>calories per minute</b> differs significantly across activity types
    (H₀: all group means are equal). Implements logic from <code>Anova_Test.py</code> and <code>UserStory5.py</code>.
    </div>
    """, unsafe_allow_html=True)

    if f_stat is not None:
        sig = p_value < 0.05
        st.markdown(f"""
        <div class="insight-box">
        F-statistic = <span class="metric-badge">{f_stat:.4f}</span> &nbsp;
        P-value = <span class="metric-badge">{p_value:.4e}</span><br>
        <b>{'✅ Reject H₀ — activity intensity differs significantly across activity types.' if sig else '⚠️ Fail to reject H₀ — no significant difference detected.'}</b>
        </div>
        """, unsafe_allow_html=True)

        cpm_df = fdf_with_cpm if not fdf_with_cpm.empty else fdf
       # if anova_summary is not None and "calories_per_minute" in cpm_df.columns and "activity_type" in cpm_df.columns:
           # order = anova_summary.sort_values("mean", ascending=False).index.tolist()
            #fig_box_anova = px.box(
             #   cpm_df.dropna(subset=["activity_type","calories_per_minute"]),
              #  x="activity_type", y="calories_per_minute",
               # category_orders={"activity_type": order},
                #title="Calories per Minute Across Activity Types",
                #color="activity_type",
               # color_discrete_sequence=px.colors.qualitative.Vivid
            #)
            #fig_box_anova.update_layout(template="plotly_dark", xaxis_tickangle=-35, showlegend=False)
            #st.plotly_chart(fig_box_anova, use_container_width=True)

        st.subheader("Group Summary Table (sorted by mean calories/min)")
        st.dataframe(
            anova_summary.sort_values("mean", ascending=False)
                            .style.format({"mean": "{:.2f}", "std": "{:.2f}"}),
            use_container_width=True
        )
    else:
        st.info("activity_type and calories_per_minute columns required for ANOVA.")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#6b7280; font-size:0.85rem;'>"
    " University of Niagara Falls . DAMO 699 – Capstone Project . Instructor: Touraj BaniRostam "
    "</p>",
    unsafe_allow_html=True
)