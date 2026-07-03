"""
job_analyse_v5.py  —  Cambridge University Jobs Dashboard
Visual improvements over v4:
  • KPI summary cards row at the top
  • Consistent gradient colouring (high=HIGHLIGHT, low=ACCENT) across all bars
  • Department full names on y-axis (truncated), ID moved to hover
  • Closing-date bars coloured by urgency (<7d HIGHLIGHT, 7-14d GOLD, 14+d BLUE)
  • Scatter replaced with bubble chart (size = Days_Open)
  • Status badges with coloured background in table panel
  • Cleaner typography hierarchy (title · subtitle · body)
  • Reduced grid noise (y-axis only, x-axis gridlines removed on bars)
  • Tighter wspace/hspace
  • Plotly hover templates enriched (dept salary on bar-2, title on scatter)
  • Interactive buttons moved directly inside the job table header
  • FIXED: Converted layout to CSS Grid to perfectly match Plotly subplots layout without shifting.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap
from plotly.subplots import make_subplots

warnings.simplefilter(action="ignore", category=FutureWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Design tokens
# ─────────────────────────────────────────────────────────────────────────────

class C:
    PRIMARY    = "#1a1a2e"
    SURFACE    = "#16213e"
    SURFACE2   = "#1e2a45"          # slightly lighter surface for card backgrounds
    ACCENT     = "#0f3460"
    HIGHLIGHT  = "#e94560"
    GOLD       = "#f5a623"
    TEXT_LIGHT = "#eaeaea"
    TEXT_MUTED = "#a0aec0"
    GREEN      = "#48bb78"
    BLUE       = "#63b3ed"
    # Consistent gradient: low value → ACCENT, high value → HIGHLIGHT
    CMAP_MAIN  = LinearSegmentedColormap.from_list("main", ["#0f3460", "#e94560"])
    CMAP_WARM  = LinearSegmentedColormap.from_list("warm", ["#0f3460", "#f5a623"])

URGENCY_COLOURS = [
    (7,  C.HIGHLIGHT),   # closes within 7 days  → red
    (14, C.GOLD),        # closes within 14 days → gold
]
URGENCY_DEFAULT = C.BLUE

STATUS_THRESHOLDS = [(10, "Busy", C.HIGHLIGHT), (5, "Normal", C.GOLD)]
STATUS_DEFAULT    = ("Calm", C.GREEN)

DAY_ORDER      = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
DAYS_EN        = {"Monday":"Mon","Tuesday":"Tue","Wednesday":"Wed",
                  "Thursday":"Thu","Friday":"Fri","Saturday":"Sat","Sunday":"Sun"}
TITLE_CANDIDATES = ["Title","Job_Title","Position","Name","Job_Name"]

# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib global theme
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "text.color":       C.TEXT_LIGHT,
    "axes.labelcolor":  C.TEXT_MUTED,
    "xtick.color":      C.TEXT_MUTED,
    "ytick.color":      C.TEXT_MUTED,
    "figure.facecolor": C.PRIMARY,
    "axes.facecolor":   C.SURFACE,
    "axes.edgecolor":   C.ACCENT,
    "grid.color":       C.ACCENT,
    "grid.alpha":       0.25,
    "grid.linewidth":   0.5,
})


# ─────────────────────────────────────────────────────────────────────────────
# Data and Path helpers
# ─────────────────────────────────────────────────────────────────────────────

def resolve_csv_path(filename: str = "cambridge_jobs_archive.csv") -> Path:
    """Helper to dynamic resolution of the CSV file path, with a fallback option."""
    script_dir = Path(__file__).parent
    dynamic_path = script_dir / filename

    if dynamic_path.exists():
        print(f"Loading data from dynamic path: {dynamic_path.resolve()}")
        return dynamic_path

    backup_str = r"C:\Users\ek675\PycharmProjects\EK675\learninng\Job_scrapper&search"
    backup_path = Path(backup_str) / filename
    if backup_path.exists():
        print(f"[warn] Dynamic path failed. Using backup path: {backup_path.resolve()}")
        return backup_path

    raise FileNotFoundError(f"CSV file '{filename}' could not be found dynamically or at backup location.")


def load_data(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    df = _clean_salary(df)
    df = _clean_dates(df)
    df["Day_Name"] = df["Posted"].dt.day_name()
    return df


def _clean_salary(df: pd.DataFrame) -> pd.DataFrame:
    df["Min_Salary_Num"] = (
        df["Min_Salary"].astype(str)
        .str.replace("£", "", regex=False)
        .str.replace(",", "", regex=False)
        .replace({"N/A": pd.NA, "nan": pd.NA, "": pd.NA})
        .pipe(pd.to_numeric, errors="coerce")
    )
    return df


def _clean_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("Closes", "Posted"):
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    missing = df[["Closes","Posted"]].isna().any(axis=1).sum()
    if missing:
        print(f"[warn] {missing} rows removed due to date format errors.")
    df = df.dropna(subset=["Closes","Posted"]).copy()
    df["Days_Open"] = (df["Closes"] - df["Posted"]).dt.days
    return df


def find_title_col(df: pd.DataFrame) -> str:
    for c in TITLE_CANDIDATES:
        if c in df.columns:
            return c
    return df.columns[0]


def dept_id_to_name(df: pd.DataFrame) -> dict:
    return df.drop_duplicates("Department_ID").set_index("Department_ID")["Department"].to_dict()


def truncate(s: str, n: int = 28) -> str:
    return s if len(s) <= n else s[:n-1] + "…"


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helpers (computed once, shared by both renderers)
# ─────────────────────────────────────────────────────────────────────────────

def agg_kpis(df: pd.DataFrame) -> dict:
    today = pd.Timestamp.now().normalize()
    return {
        "total":        len(df),
        "depts":        df["Department"].nunique(),
        "avg_salary":   df["Min_Salary_Num"].mean(),
        "closing_week": int((df["Closes"].between(today, today + pd.Timedelta(days=7))).sum()),
        "posted_today": int((df["Posted"] == today).sum()),
    }


def agg_closing_counts(df: pd.DataFrame, n: int = 10) -> pd.Series:
    return df["Closes"].value_counts().sort_index().head(n)


def agg_top_departments(df: pd.DataFrame, n: int = 10) -> pd.Series:
    return df["Department_ID"].value_counts().head(n)


def agg_daily_avg(df: pd.DataFrame) -> pd.Series:
    avg = (
        df.groupby(["Posted","Day_Name"]).size()
        .reset_index(name="Daily_Count")
        .groupby("Day_Name")["Daily_Count"].mean()
        .reindex(DAY_ORDER)
    )
    avg.index = [DAYS_EN[d] for d in avg.index]
    return avg


def agg_dept_salary(df: pd.DataFrame, n: int = 10) -> pd.Series:
    return (
        df.groupby("Department_ID")["Min_Salary_Num"]
        .mean().dropna().sort_values(ascending=False).head(n)
    )


def agg_daily_counts(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    dc = df["Posted"].value_counts().sort_index(ascending=False).reset_index().head(n)
    dc.columns = ["Date","Count"]
    dc["Day"]       = dc["Date"].dt.day_name().map(DAYS_EN)
    dc["Date_Str"] = dc["Date"].dt.strftime("%d %b")
    return dc


def classify_status(count: int) -> tuple[str, str]:
    for threshold, label, colour in STATUS_THRESHOLDS:
        if count > threshold:
            return label, colour
    return STATUS_DEFAULT


def urgency_colour(close_date: pd.Timestamp) -> str:
    today = pd.Timestamp.now().normalize()
    delta = (close_date - today).days
    for days, colour in URGENCY_COLOURS:
        if delta <= days:
            return colour
    return URGENCY_DEFAULT


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib helpers
# ─────────────────────────────────────────────────────────────────────────────

def style_axis(ax, title: str, xlabel: str = "", ylabel: str = "",
               y_grid_only: bool = True) -> None:
    ax.set_facecolor(C.SURFACE)
    ax.set_title(title, fontsize=13, fontweight="bold", color=C.TEXT_LIGHT,
                 pad=10, loc="left")
    ax.set_xlabel(xlabel, fontsize=10, color=C.TEXT_MUTED, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=10, color=C.TEXT_MUTED, labelpad=6)
    ax.tick_params(labelsize=9, colors=C.TEXT_MUTED, length=3)
    for spine in ax.spines.values():
        spine.set_edgecolor(C.ACCENT)
        spine.set_linewidth(0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if y_grid_only:
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)


def add_hbar_labels(ax, bars, fmt: str = "{:.0f}",
                    color: str = C.TEXT_LIGHT, fontsize: int = 8) -> None:
    for bar in bars:
        w = bar.get_width()
        ax.annotate(fmt.format(w),
                    xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0), textcoords="offset points",
                    fontsize=fontsize, color=color,
                    ha="left", va="center", fontweight="bold")


def gradient_colours(values: np.ndarray, cmap) -> list:
    norm = values / values.max() if values.max() > 0 else values
    return [cmap(v) for v in norm]


# ─────────────────────────────────────────────────────────────────────────────
# KPI card drawing (Matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

def draw_kpi_card(ax, value: str, label: str, accent_colour: str) -> None:
    ax.set_facecolor(C.SURFACE2)
    for spine in ax.spines.values():
        spine.set_edgecolor(accent_colour)
        spine.set_linewidth(2)
    ax.axis("off")
    ax.plot([0.08, 0.92], [0.92, 0.92], color=accent_colour,
            linewidth=3, transform=ax.transAxes, clip_on=False)
    ax.text(0.5, 0.58, value, ha="center", va="center",
            fontsize=22, fontweight="bold", color=C.TEXT_LIGHT,
            transform=ax.transAxes)
    ax.text(0.5, 0.22, label, ha="center", va="center",
            fontsize=9, color=C.TEXT_MUTED, transform=ax.transAxes)


# ─────────────────────────────────────────────────────────────────────────────
# PNG renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_png(df: pd.DataFrame, output_path: str = "cambridge_dashboard.png") -> None:
    kpis       = agg_kpis(df)
    c_counts   = agg_closing_counts(df)
    top_depts  = agg_top_departments(df)
    avg_day    = agg_daily_avg(df)
    dept_sal   = agg_dept_salary(df)
    daily_dc   = agg_daily_counts(df)
    id2dept    = dept_id_to_name(df)

    fig = plt.figure(figsize=(30, 20), dpi=120, facecolor=C.PRIMARY)

    outer = gridspec.GridSpec(2, 1, figure=fig,
                              top=0.91, bottom=0.05,
                              left=0.05, right=0.97,
                              hspace=0.08,
                              height_ratios=[1, 5.5])

    kpi_gs   = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[0], wspace=0.06)
    chart_gs = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer[1], wspace=0.30, hspace=0.44)

    fig.text(0.5, 0.965, "Cambridge University  ·  Jobs Analytics Dashboard",
             ha="center", fontsize=24, fontweight="bold", color=C.TEXT_LIGHT,
             fontfamily="DejaVu Sans")
    fig.text(0.5, 0.942,
             f"{datetime.now().strftime('%d %B %Y')}",
             ha="center", fontsize=11, color=C.TEXT_MUTED)

    avg_sal_str = f"£{kpis['avg_salary']/1000:.1f}k" if pd.notna(kpis['avg_salary']) else "N/A"
    kpi_data = [
        (str(kpis["total"]),           "Total Jobs",          C.HIGHLIGHT),
        (str(kpis["depts"]),           "Active Depts",        C.BLUE),
        (avg_sal_str,                  "Avg Min Salary",      C.GOLD),
        (str(kpis["closing_week"]),    "Closing This Week",   C.HIGHLIGHT),
        (str(kpis["posted_today"]),    "Posted Today",        C.GREEN),
    ]
    for i, (val, lbl, col) in enumerate(kpi_data):
        ax = fig.add_subplot(kpi_gs[i])
        draw_kpi_card(ax, val, lbl, col)

    axes = [[fig.add_subplot(chart_gs[r, c]) for c in range(3)] for r in range(2)]

    ax = axes[0][0]
    bar_colours = [urgency_colour(d) for d in c_counts.index]
    ax.bar(c_counts.index.strftime("%d %b"), c_counts.values,
           color=bar_colours, edgecolor=C.ACCENT, linewidth=0.5, width=0.65)
    style_axis(ax, "Upcoming Closing Dates", ylabel="Job Count")
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylim(0, c_counts.max() * 1.20)
    patches = [mpatches.Patch(color=C.HIGHLIGHT, label="≤7 days"),
               mpatches.Patch(color=C.GOLD,      label="≤14 days"),
               mpatches.Patch(color=C.BLUE,      label=">14 days")]
    ax.legend(handles=patches, fontsize=7.5, loc="upper right",
              facecolor=C.SURFACE, edgecolor=C.ACCENT, labelcolor=C.TEXT_LIGHT)

    ax = axes[0][1]
    dept_full   = [truncate(id2dept.get(did, str(did))) for did in top_depts.index]
    colours2    = gradient_colours(top_depts.values.astype(float), C.CMAP_MAIN)
    bars2       = ax.barh(dept_full, top_depts.values,
                          color=colours2, edgecolor=C.ACCENT, linewidth=0.4, height=0.65)
    add_hbar_labels(ax, bars2)
    style_axis(ax, "Top Recruiting Departments", xlabel="Job Count")
    ax.set_xlim(0, top_depts.max() * 1.22)
    ax.tick_params(axis="y", labelsize=8)
    ax.invert_yaxis()

    ax = axes[0][2]
    ax.fill_between(avg_day.index, avg_day.values, alpha=0.15, color=C.HIGHLIGHT)
    ax.plot(avg_day.index, avg_day.values, marker="o", linewidth=2.5, markersize=8,
            color=C.HIGHLIGHT, markerfacecolor=C.TEXT_LIGHT, markeredgewidth=2)
    for i, (x, y) in enumerate(zip(avg_day.index, avg_day.values)):
        if pd.notna(y):
            ax.annotate(f"{y:.1f}", xy=(i, y), xytext=(0, 9),
                        textcoords="offset points", ha="center",
                        fontsize=8.5, color=C.GOLD, fontweight="bold")
    style_axis(ax, "Daily Posting Rhythm", ylabel="Avg Jobs / Day")
    ax.set_ylim(0, avg_day.max() * 1.28)

    ax = axes[1][0]
    dept_sal_full = [truncate(id2dept.get(did, str(did))) for did in dept_sal.index]
    colours4      = gradient_colours(dept_sal.values, C.CMAP_WARM)
    bars4         = ax.barh(dept_sal_full, dept_sal.values,
                            color=colours4, edgecolor=C.ACCENT, linewidth=0.4, height=0.65)
    for bar in bars4:
        val = bar.get_width()
        ax.annotate(f"£{val/1000:.0f}k",
                    xy=(val + dept_sal.max()*0.01, bar.get_y() + bar.get_height()/2),
                    fontsize=8.5, color=C.GOLD, fontweight="bold", ha="left", va="center")
    style_axis(ax, "Avg Min Salary by Department", xlabel="Min Salary")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x/1000:.0f}k"))
    ax.set_xlim(0, dept_sal.max() * 1.20)
    ax.tick_params(axis="y", labelsize=8)
    ax.invert_yaxis()

    ax = axes[1][1]
    valid = df.dropna(subset=["Min_Salary_Num","Days_Open"])
    if len(valid) > 0:
        sizes  = np.clip(valid["Days_Open"] * 1.8, 20, 300)
        salary_norm = (valid["Min_Salary_Num"] - valid["Min_Salary_Num"].min()) / \
                      (valid["Min_Salary_Num"].max() - valid["Min_Salary_Num"].min() + 1)
        bubble_cols = [C.CMAP_MAIN(v) for v in salary_norm]
        ax.scatter(valid["Min_Salary_Num"], valid["Days_Open"],
                   s=sizes, c=bubble_cols, alpha=0.55,
                   edgecolors=C.ACCENT, linewidths=0.4)
        if len(valid) > 2:
            z = np.polyfit(valid["Min_Salary_Num"], valid["Days_Open"], 1)
            x_line = np.linspace(valid["Min_Salary_Num"].min(), valid["Min_Salary_Num"].max(), 200)
            ax.plot(x_line, np.poly1d(z)(x_line), color=C.GOLD,
                    linewidth=1.8, linestyle="--", label="Trend", alpha=0.85)
            ax.legend(fontsize=8.5, facecolor=C.SURFACE, edgecolor=C.ACCENT, labelcolor=C.TEXT_LIGHT)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x/1000:.0f}k"))
    style_axis(ax, "Salary vs. Visibility Window (size = days open)", xlabel="Min Salary", ylabel="Days Open")
    for size_val, lbl in [(20, "20d"), (100, "60d"), (250, "120d+")]:
        ax.scatter([], [], s=size_val, color=C.BLUE, alpha=0.5, edgecolors=C.ACCENT, label=lbl)

    ax = axes[1][2]
    ax.set_facecolor(C.SURFACE)
    ax.axis("off")
    ax.set_title("Activity Flow (Last 10 Days)", fontsize=13, fontweight="bold", color=C.TEXT_LIGHT, pad=10, loc="left")

    col_labels = ["Date", "Day", "Jobs", "Status"]
    col_widths  = [0.26, 0.20, 0.16, 0.30]
    header_y, row_h = 0.94, 0.082

    xp = 0.03
    for label, w in zip(col_labels, col_widths):
        ax.text(xp + w/2, header_y, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color=C.GOLD, transform=ax.transAxes)
        xp += w
    ax.plot([0.02, 0.98], [header_y - 0.03, header_y - 0.03], color=C.HIGHLIGHT, linewidth=1.0, transform=ax.transAxes, clip_on=False)

    for i, (_, row) in enumerate(daily_dc.iterrows()):
        y   = header_y - 0.055 - i * row_h
        bg = mpatches.FancyBboxPatch(
            (0.02, y - row_h*0.46), 0.96, row_h*0.88,
            boxstyle="round,pad=0.004", linewidth=0,
            facecolor=C.ACCENT if i % 2 == 0 else C.SURFACE,
            transform=ax.transAxes, clip_on=True)
        ax.add_patch(bg)

        label, badge_col = classify_status(int(row["Count"]))
        vals = [row["Date_Str"], row["Day"], str(int(row["Count"])), ""]
        cols = [C.TEXT_LIGHT, C.TEXT_MUTED, C.TEXT_LIGHT, badge_col]
        xp = 0.03
        for j, (val, col, w) in enumerate(zip(vals, cols, col_widths)):
            if j == 3:
                bx = xp + w/2
                badge = mpatches.FancyBboxPatch(
                    (bx - 0.12, y - 0.025), 0.24, 0.052,
                    boxstyle="round,pad=0.005", linewidth=0,
                    facecolor=badge_col, alpha=0.25,
                    transform=ax.transAxes, clip_on=True)
                ax.add_patch(badge)
                ax.text(bx, y, label, ha="center", va="center",
                        fontsize=8.5, color=badge_col, fontweight="bold", transform=ax.transAxes)
            else:
                ax.text(xp + w/2, y, val, ha="center", va="center",
                        fontsize=9, color=col, transform=ax.transAxes)
            xp += w

    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=C.PRIMARY)
    print(f"PNG saved → {output_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Plotly HTML renderer
# ─────────────────────────────────────────────────────────────────────────────

def _urgency_colour_list(index: pd.DatetimeIndex) -> list[str]:
    return [urgency_colour(d) for d in index]


def _gradient_hex(values: pd.Series, cmap) -> list[str]:
    norm = (values - values.min()) / (values.max() - values.min() + 1)
    def to_hex(rgba):
        return "#{:02x}{:02x}{:02x}".format(
            int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
    return [to_hex(cmap(v)) for v in norm]


def _job_table(sub_df: pd.DataFrame, title_col: str,
               header_colour: str, visible: bool) -> go.Table:
    return go.Table(
        columnwidth=[0.50, 0.35, 0.15],
       header=dict(
            values=["<b>Job Title</b>", "<b>Department</b>", "<b>Date</b>"],
            fill_color=header_colour,
            font=dict(color=C.TEXT_LIGHT, size=11),
            align="left",
            line_color=C.PRIMARY,
        ),
        cells=dict(
            values=[
                sub_df[title_col].tolist(),
                sub_df["Department"].tolist(),
                sub_df.iloc[:, 2].dt.strftime("%d %b %Y").tolist(),
            ],
            fill_color=[[C.SURFACE if i%2==0 else C.PRIMARY for i in range(len(sub_df))]],
            font=dict(color=C.TEXT_LIGHT, size=10),
            align=["left","left","center"],
            line_color=C.ACCENT,
        ),
        visible=visible,
    )


def render_html(df: pd.DataFrame, output_path: str = "cambridge_dashboard.html",
                open_browser: bool = True) -> None:
    title_col = find_title_col(df)
    kpis      = agg_kpis(df)
    c_counts  = agg_closing_counts(df)
    top_depts = agg_top_departments(df)
    avg_day   = agg_daily_avg(df)
    dept_sal  = agg_dept_salary(df)
    daily_dc  = agg_daily_counts(df)
    id2dept   = dept_id_to_name(df)

    fig = make_subplots(
        rows=2, cols=3,
        row_heights=[0.5, 0.5],
        subplot_titles=[
            "Upcoming Closing Dates",
            "Top Recruiting Departments",
            "Daily Posting Rhythm",
            "Avg Min Salary by Department",
            "",  # Hidden completely: Controlled outside Plotly container by structural layout framework
            "Activity Flow (Last 10 Days)",
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.16,
        specs=[
            [{"type": "xy"},     {"type": "xy"},     {"type": "xy"}],
            [{"type": "xy"},     {"type": "domain"},  {"type": "domain"}],
        ],
    )

    urg_cols   = _urgency_colour_list(c_counts.index)
    days_until = [(d - pd.Timestamp.now().normalize()).days for d in c_counts.index]
    fig.add_trace(go.Bar(
        x=c_counts.index.strftime("%d %b"),
        y=c_counts.values,
        marker_color=urg_cols,
        customdata=list(zip(c_counts.index.strftime("%d %B %Y"), days_until)),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "%{y} jobs closing · "
            "<i>%{customdata[1]} days left</i><extra></extra>"
        ),
        name="",
    ), row=1, col=1)

    text_depts = [id2dept.get(did, str(did)) for did in top_depts.index]
    short_depts = [truncate(n, 24) for n in text_depts]
    cols_depts  = _gradient_hex(pd.Series(top_depts.values.astype(float)), C.CMAP_MAIN)
    sals_depts  = {did: df[df["Department_ID"] == did]["Min_Salary_Num"].mean() for did in top_depts.index}
    hover_sals  = [f"£{sals_depts[did]/1000:.0f}k" if pd.notna(sals_depts[did]) else "N/A" for did in top_depts.index]
    fig.add_trace(go.Bar(
        x=top_depts.values,
        y=short_depts,
        orientation="h",
        marker_color=cols_depts,
        customdata=list(zip(text_depts, hover_sals)),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "%{x} jobs · Avg salary: %{customdata[1]}<extra></extra>"
        ),
        name="",
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=avg_day.index, y=avg_day.values,
        mode="lines+markers",
        line=dict(color=C.HIGHLIGHT, width=2.5),
        marker=dict(size=9, color=C.TEXT_LIGHT, line=dict(color=C.HIGHLIGHT, width=2)),
        fill="tozeroy", fillcolor="rgba(233,69,96,0.10)",
        hovertemplate="<b>%{x}</b><br>Avg: %{y:.1f} jobs/day<extra></extra>",
        name="",
    ), row=1, col=3)

    dept_sal_names = [truncate(id2dept.get(did, str(did)), 24) for did in dept_sal.index]
    dept_sal_full  = [id2dept.get(did, str(did)) for did in dept_sal.index]
    sal_colours    = _gradient_hex(dept_sal, C.CMAP_WARM)
    fig.add_trace(go.Bar(
        x=dept_sal.values,
        y=dept_sal_names,
        orientation="h",
        marker_color=sal_colours,
        customdata=dept_sal_full,
        hovertemplate="<b>%{customdata}</b><br>Avg: £%{x:,.0f}<extra></extra>",
        name="",
    ), row=2, col=1)

    closes_top10 = (
        df[df["Closes"] >= pd.Timestamp.now()]
        .sort_values("Closes").head(10)
        [[title_col, "Department", "Closes"]].copy()
    )
    posted_top10 = (
        df.sort_values("Posted", ascending=False).head(10)
        [[title_col, "Department", "Posted"]].copy()
    )
    fig.add_trace(_job_table(closes_top10, title_col, C.HIGHLIGHT, visible=True),  row=2, col=2)
    fig.add_trace(_job_table(posted_top10, title_col, C.BLUE,      visible=False), row=2, col=2)

    daily_dc["Status"] = daily_dc["Count"].apply(lambda x: classify_status(int(x))[0])
    fig.add_trace(go.Table(
        header=dict(
            values=["<b>Date</b>", "<b>Day</b>", "<b>Jobs</b>", "<b>Status</b>"],
            fill_color=C.ACCENT,
            font=dict(color=C.GOLD, size=11),
            align="center", line_color=C.PRIMARY,
        ),
        cells=dict(
            values=[daily_dc["Date_Str"], daily_dc["Day"], daily_dc["Count"], daily_dc["Status"]],
            fill_color=[[C.SURFACE if i % 2 == 0 else C.PRIMARY for i in range(len(daily_dc))]],
            font=dict(color=C.TEXT_LIGHT, size=10),
            align="center", line_color=C.ACCENT,
        ),
    ), row=2, col=3)

    fig.update_layout(
        title=None,
        paper_bgcolor=C.PRIMARY,
        plot_bgcolor=C.SURFACE,
        font=dict(color=C.TEXT_LIGHT, family="DejaVu Sans"),
        showlegend=False,
        autosize=True,
        height=800,
        margin=dict(t=10, b=10, l=40, r=40),
        hoverlabel=dict(bgcolor=C.SURFACE, bordercolor=C.HIGHLIGHT,
                        font=dict(color=C.TEXT_LIGHT, size=12)),
    )

    fig.update_xaxes(showgrid=False, gridcolor=C.ACCENT, gridwidth=0.4)
    fig.update_yaxes(showgrid=True,  gridcolor=C.ACCENT, gridwidth=0.4)
    fig.update_xaxes(showgrid=True, row=1, col=3)
    fig.update_yaxes(showticklabels=False, title=None, row=2, col=1)

    plotly_div = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"responsive": True},
    )

    avg_sal_str = f"£{kpis['avg_salary']/1000:.1f}k" if pd.notna(kpis['avg_salary']) else "N/A"
    kpi_html_defs = [
        (str(kpis["total"]),        "Total Jobs",         C.HIGHLIGHT),
        (str(kpis["depts"]),        "Active Depts",       C.BLUE),
        (avg_sal_str,               "Avg Min Salary",     C.GOLD),
        (str(kpis["closing_week"]), "Closing This Week",  C.HIGHLIGHT),
        (str(kpis["posted_today"]), "Posted Today",       C.GREEN),
    ]

    kpi_cards = ""
    for val, lbl, col in kpi_html_defs:
        kpi_cards += f"""
        <div class="kpi-card" style="border-color:{col}">
            <div class="kpi-bar" style="background:{col}"></div>
            <div class="kpi-value" style="color:{col}">{val}</div>
            <div class="kpi-label">{lbl}</div>
        </div>"""

    today_str = datetime.now().strftime('%d %B %Y')
    total_ilan = len(df)
    n_dept = df['Department'].nunique()

    # RESTRUCTURED HTML CONTAINER WITH HYBRID DESIGN PATTERN (FLEX + SEPARATE GRID SYSTEM FOR HEADERS)
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cambridge Jobs Dashboard</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: {C.PRIMARY};
    font-family: 'DejaVu Sans', Arial, sans-serif;
    color: {C.TEXT_LIGHT};
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    overflow-y: auto;
  }}
  .header {{
    text-align: center;
    padding: 18px 20px 8px;
    border-bottom: 1px solid {C.ACCENT};
  }}
  .header h1 {{
    font-size: 22px;
    font-weight: 700;
    color: {C.TEXT_LIGHT};
    letter-spacing: 0.3px;
  }}
  .header .subtitle {{
    font-size: 12px;
    color: {C.TEXT_MUTED};
    margin-top: 4px;
  }}
  .kpi-strip {{
    display: flex;
    gap: 10px;
    padding: 12px 20px;
    background: {C.PRIMARY};
    border-bottom: 1px solid {C.ACCENT};
    flex-shrink: 0;
  }}
  .kpi-card {{
    flex: 1;
    background: {C.SURFACE2};
    border: 1.5px solid;
    border-radius: 6px;
    padding: 10px 12px 12px;
    position: relative;
    overflow: hidden;
    min-width: 0;
  }}
  .kpi-bar {{
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 6px 6px 0 0;
  }}
  .kpi-value {{
    font-size: 28px;
    font-weight: 700;
    line-height: 1.1;
    margin-top: 6px;
  }}
  .kpi-label {{
    font-size: 11px;
    color: {C.TEXT_MUTED};
    margin-top: 5px;
  }}
  
  /* Robust CSS Layout Framework to perfectly coordinate HTML panels with the Plotly Grid */
  .dashboard-container {{
    position: relative;
    width: 100%;
    max-width: 100%;
    margin: 0 auto;
    padding: 0 40px;
    display: flex;
    flex-direction: column;
  }}

  /* Top Overlay Header Panel mapped to Row 2 Grid layout coordinates */
  .row2-headers-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    column-gap: 8%; /* Matches Plotly horizontal_spacing precisely */
    width: 100%;
    margin-top: 425px; /* Safely pushes control panel directly over Row 2 */
    position: absolute;
    left: 0;
    padding: 0 40px;
    z-index: 10;
    pointer-events: none; /* Allows chart hovering beneath empty grid columns */
  }}

  .grid-header-cell {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: {C.PRIMARY};
    padding-bottom: 8px;
    pointer-events: auto; /* Restores interactive behavior on the actual control area */
    flex-wrap: nowrap;
  }}

  .table-title {{
    font-size: 13px;
    font-weight: bold;
    color: {C.TEXT_LIGHT};
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    padding-right: 8px;
  }}

  .btn-group {{
    display: flex;
    gap: 4px;
    flex-shrink: 0;
    white-space: nowrap;
  }}

  .table-toggle-btn {{
    background: {C.ACCENT};
    border: 1px solid {C.ACCENT};
    color: {C.TEXT_MUTED};
    padding: 3px 8px;
    font-size: 10px;
    font-weight: bold;
    cursor: pointer;
    border-radius: 4px;
    transition: all 0.2s ease;
  }}

  .table-toggle-btn:hover {{
    color: {C.TEXT_LIGHT};
    border-color: {C.HIGHLIGHT};
  }}

  .table-toggle-btn.active {{
    background: {C.HIGHLIGHT};
    border-color: {C.HIGHLIGHT};
    color: {C.TEXT_LIGHT};
  }}

  .chart-area {{
    width: 100%;
    height: 800px;
    z-index: 1;
  }}
</style>
</head>
<body>

<div class="header">
  <h1>Cambridge University · Jobs Analytics Dashboard</h1>
  <div class="subtitle">{today_str} &nbsp;·&nbsp; {total_ilan:,} jobs &nbsp;·&nbsp; {n_dept} departments</div>
</div>

<div class="kpi-strip">
{kpi_cards}
</div>

<div class="dashboard-container">
  
  <div class="row2-headers-grid">
    <div></div>
    
    <div class="grid-header-cell">
      <div class="table-title" id="dynamic-panel-title">10 Jobs Closing Soon</div>
      <div class="btn-group">
        <button class="table-toggle-btn active" id="btn-closing" onclick="switchTable('closing')">📅 Closing Soon</button>
        <button class="table-toggle-btn" id="btn-recent" onclick="switchTable('recent')">🆕 Recently Posted</button>
      </div>
    </div>
    
    <div></div>
  </div>

  <div class="chart-area">
  {plotly_div}
  </div>
</div>

<script>
  function switchTable(mode) {{
    var gd = document.querySelector('.plotly-graph-div');
    if (!gd || !window.Plotly) return;
    
    var titleEl = document.getElementById('dynamic-panel-title');
    var btnClosing = document.getElementById('btn-closing');
    var btnRecent = document.getElementById('btn-recent');
    
    if (mode === 'closing') {{
      titleEl.innerText = "10 Jobs Closing Soon";
      btnClosing.classList.add('active');
      btnRecent.classList.remove('active');
      window.Plotly.restyle(gd, {{'visible': [true, true, true, true, true, false, true]}});
    }} else {{
      titleEl.innerText = "10 Most Recently Posted Jobs";
      btnClosing.classList.remove('active');
      btnRecent.classList.add('active');
      window.Plotly.restyle(gd, {{'visible': [true, true, true, true, false, true, true]}});
    }}
  }}

  (function() {{
    var gd = document.querySelector('.plotly-graph-div');
    if (!gd) return;
    window.addEventListener('resize', function() {{
      if (window.Plotly) {{
        window.Plotly.Plots.resize(gd);
      }}
    }});
  }})();
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"HTML saved → {output_path}")

    if open_browser:
        import webbrowser
        webbrowser.open(Path(output_path).absolute().as_uri())


if __name__ == "__main__":
    csv_file = resolve_csv_path("cambridge_jobs_archive.csv")
    df_data  = load_data(csv_file)
    render_png(df_data)
    render_html(df_data, open_browser=True)