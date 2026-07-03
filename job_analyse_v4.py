"""
job_analyse_v4.py  —  Cambridge University Jobs Dashboard
Refactored from v3: modular, DRY, validated, type-annotated.
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
# Constants
# ─────────────────────────────────────────────────────────────────────────────

class Colour:
    PRIMARY    = "#1a1a2e"
    SURFACE    = "#16213e"
    ACCENT     = "#0f3460"
    HIGHLIGHT  = "#e94560"
    GOLD       = "#f5a623"
    TEXT_LIGHT = "#eaeaea"
    TEXT_MUTED = "#a0aec0"
    GREEN      = "#48bb78"
    BLUE       = "#63b3ed"
    CMAP_COOL  = LinearSegmentedColormap.from_list("cool", ["#0f3460", "#e94560"])
    CMAP_WARM  = LinearSegmentedColormap.from_list("warm", ["#1a1a2e", "#f5a623"])


DAYS_TR: dict[str, str] = {
    "Monday": "Pzt", "Tuesday": "Sal", "Wednesday": "Car",
    "Thursday": "Per", "Friday": "Cum", "Saturday": "Cmt", "Sunday": "Paz",
}

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Status thresholds (jobs posted per day)
STATUS_THRESHOLDS = [(10, "Yogun", Colour.HIGHLIGHT), (5, "Normal", Colour.GOLD)]
STATUS_DEFAULT    = ("Sakin", Colour.GREEN)

# Preferred column names for job title, in priority order
TITLE_CANDIDATES = ["Title", "Job_Title", "Position", "Name", "Job_Name"]


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib global theme
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "text.color":       Colour.TEXT_LIGHT,
    "axes.labelcolor":  Colour.TEXT_MUTED,
    "xtick.color":      Colour.TEXT_MUTED,
    "ytick.color":      Colour.TEXT_MUTED,
    "figure.facecolor": Colour.PRIMARY,
    "axes.facecolor":   Colour.SURFACE,
    "axes.edgecolor":   Colour.ACCENT,
    "grid.color":       Colour.ACCENT,
    "grid.alpha":       0.3,
    "grid.linewidth":   0.6,
})


# ─────────────────────────────────────────────────────────────────────────────
# Data loading & cleaning
# ─────────────────────────────────────────────────────────────────────────────

def load_data(csv_path: str | Path) -> pd.DataFrame:
    """Load and clean the jobs CSV. Raises FileNotFoundError if missing."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    df = _clean_salary(df)
    df = _clean_dates(df)
    df["Day_Name"] = df["Posted"].dt.day_name()
    return df


def _clean_salary(df: pd.DataFrame) -> pd.DataFrame:
    """Parse Min_Salary column to numeric, storing result in Min_Salary_Num."""
    df["Min_Salary_Num"] = (
        df["Min_Salary"]
        .astype(str)
        .str.replace("£", "", regex=False)
        .str.replace(",", "", regex=False)
        .replace({"N/A": pd.NA, "nan": pd.NA, "": pd.NA})
        .pipe(pd.to_numeric, errors="coerce")
    )
    return df


def _clean_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse date columns and derive Days_Open; drop rows with missing dates."""
    for col in ("Closes", "Posted"):
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

    missing = df[["Closes", "Posted"]].isna().any(axis=1).sum()
    if missing:
        print(f"[warn] Dropping {missing} rows with unparseable dates.")
    df = df.dropna(subset=["Closes", "Posted"]).copy()
    df["Days_Open"] = (df["Closes"] - df["Posted"]).dt.days
    return df


def find_title_column(df: pd.DataFrame) -> str:
    """Return the best available job-title column name."""
    for candidate in TITLE_CANDIDATES:
        if candidate in df.columns:
            return candidate
    print(f"[warn] No standard title column found; using '{df.columns[0]}'.")
    return df.columns[0]


# ─────────────────────────────────────────────────────────────────────────────
# Shared aggregation helpers (called once, reused by both renderers)
# ─────────────────────────────────────────────────────────────────────────────

def agg_closing_counts(df: pd.DataFrame, n: int = 10) -> pd.Series:
    return df["Closes"].value_counts().sort_index().head(n)


def agg_top_departments(df: pd.DataFrame, n: int = 10) -> pd.Series:
    return df["Department_ID"].value_counts().head(n)


def agg_daily_avg(df: pd.DataFrame) -> pd.Series:
    avg = (
        df.groupby(["Posted", "Day_Name"])
        .size()
        .reset_index(name="Daily_Count")
        .groupby("Day_Name")["Daily_Count"]
        .mean()
        .reindex(DAY_ORDER)
    )
    avg.index = [DAYS_TR[d] for d in avg.index]
    return avg


def agg_dept_salary(df: pd.DataFrame, n: int = 10) -> pd.Series:
    return (
        df.groupby("Department_ID")["Min_Salary_Num"]
        .mean()
        .dropna()
        .sort_values(ascending=False)
        .head(n)
    )


def agg_daily_counts(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    dc = (
        df["Posted"].value_counts()
        .sort_index(ascending=False)
        .reset_index()
        .head(n)
    )
    dc.columns = ["Tarih", "Sayi"]
    dc["Gun"]      = dc["Tarih"].dt.day_name().map(DAYS_TR)
    dc["Tarih_Str"]= dc["Tarih"].dt.strftime("%d %b")
    return dc


def classify_status(count: int) -> tuple[str, str]:
    """Return (label, colour) for a daily posting count."""
    for threshold, label, colour in STATUS_THRESHOLDS:
        if count > threshold:
            return label, colour
    return STATUS_DEFAULT


def dept_id_to_name(df: pd.DataFrame) -> dict:
    return df.drop_duplicates("Department_ID").set_index("Department_ID")["Department"].to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib helpers
# ─────────────────────────────────────────────────────────────────────────────

def style_axis(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(Colour.SURFACE)
    ax.set_title(title, fontsize=15, fontweight="bold", color=Colour.TEXT_LIGHT,
                 pad=14, loc="left")
    ax.set_xlabel(xlabel, fontsize=11, color=Colour.TEXT_MUTED, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=11, color=Colour.TEXT_MUTED, labelpad=8)
    ax.tick_params(labelsize=10, colors=Colour.TEXT_MUTED, length=4)
    for spine in ax.spines.values():
        spine.set_edgecolor(Colour.ACCENT)
        spine.set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_hbar_labels(ax, bars, fmt: str = "{:.0f}",
                    color: str = Colour.TEXT_LIGHT, fontsize: int = 9) -> None:
    """Annotate horizontal bar values at the right edge."""
    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            fmt.format(width),
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(3, 0), textcoords="offset points",
            fontsize=fontsize, color=color, ha="left", va="center", fontweight="bold",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib PNG renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_png(df: pd.DataFrame, output_path: str = "cambridge_dashboard.png") -> None:
    """Render a static PNG dashboard using Matplotlib."""
    c_counts   = agg_closing_counts(df)
    top_depts  = agg_top_departments(df)
    avg_day    = agg_daily_avg(df)
    dept_sal   = agg_dept_salary(df)
    daily_dc   = agg_daily_counts(df)
    id2dept    = dept_id_to_name(df)

    fig = plt.figure(figsize=(28, 18), dpi=120, facecolor=Colour.PRIMARY)
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            left=0.06, right=0.97, top=0.82, bottom=0.07,
                            wspace=0.38, hspace=0.48)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]

    fig.text(0.5, 0.95, "Cambridge University - Is Ilanlari Analiz Paneli",
             ha="center", fontsize=22, fontweight="bold", color=Colour.TEXT_LIGHT)
    fig.text(0.5, 0.90,
             f"Guncelleme: {datetime.now().strftime('%d %B %Y')}  |  "
             f"Toplam Ilan: {len(df):,}  |  "
             f"Benzersiz Bolum: {df['Department'].nunique()}",
             ha="center", fontsize=11, color=Colour.TEXT_MUTED)

    # Panel 1 — Upcoming closing dates
    ax = axes[0][0]
    colours = [Colour.HIGHLIGHT if i == 0 else Colour.BLUE for i in range(len(c_counts))]
    ax.bar(c_counts.index.strftime("%d %b"), c_counts.values,
           color=colours, edgecolor=Colour.ACCENT, linewidth=0.6, width=0.65)
    style_axis(ax, "Yaklasan Ilan Kapanis Tarihleri", ylabel="Ilan Sayisi")
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylim(0, c_counts.max() * 1.18)

    # Panel 2 — Most active departments
    ax = axes[0][1]
    norm = top_depts.values / top_depts.max()
    bars = ax.barh(top_depts.index, top_depts.values,
                   color=[Colour.CMAP_COOL(v) for v in norm],
                   edgecolor=Colour.ACCENT, linewidth=0.5, height=0.65)
    add_hbar_labels(ax, bars)
    style_axis(ax, "En Cok Ilan Veren Bolumler", xlabel="Ilan Sayisi")
    ax.set_xlim(0, top_depts.max() * 1.20)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.invert_yaxis()

    # Panel 3 — Daily posting rhythm
    ax = axes[0][2]
    ax.fill_between(avg_day.index, avg_day.values, alpha=0.18, color=Colour.HIGHLIGHT)
    ax.plot(avg_day.index, avg_day.values, marker="o", linewidth=2.5, markersize=9,
            color=Colour.HIGHLIGHT, markerfacecolor=Colour.TEXT_LIGHT, markeredgewidth=2)
    for i, (x, y) in enumerate(zip(avg_day.index, avg_day.values)):
        if pd.notna(y):
            ax.annotate(f"{y:.1f}", xy=(i, y), xytext=(0, 10),
                        textcoords="offset points", ha="center",
                        fontsize=9, color=Colour.GOLD, fontweight="bold")
    style_axis(ax, "Gunluk Ortalama Ilan Yayini", ylabel="Ort. Ilan / Gun")
    ax.set_ylim(0, avg_day.max() * 1.25)

    # Panel 4 — Department salary
    ax = axes[1][0]
    norm = dept_sal.values / dept_sal.max()
    bars = ax.barh(dept_sal.index, dept_sal.values,
                   color=[Colour.CMAP_WARM(v) for v in norm],
                   edgecolor=Colour.ACCENT, linewidth=0.5, height=0.65)
    for bar in bars:
        val = bar.get_width()
        ax.annotate(f"£{val / 1000:.0f}k",
                    xy=(val + dept_sal.max() * 0.01, bar.get_y() + bar.get_height() / 2),
                    fontsize=9, color=Colour.GOLD, fontweight="bold", ha="left", va="center")
    style_axis(ax, "Bolum Bazli Ortalama Maas (Min.)", xlabel="Maas")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x / 1000:.0f}k"))
    ax.set_xlim(0, dept_sal.max() * 1.18)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.invert_yaxis()

    # Panel 5 — Salary vs days open scatter
    ax = axes[1][1]
    valid = df.dropna(subset=["Min_Salary_Num", "Days_Open"])
    ax.scatter(valid["Min_Salary_Num"], valid["Days_Open"],
               alpha=0.45, s=60, color=Colour.BLUE,
               edgecolors=Colour.ACCENT, linewidths=0.4)
    if len(valid) > 2:
        z      = np.polyfit(valid["Min_Salary_Num"], valid["Days_Open"], 1)
        x_line = np.linspace(valid["Min_Salary_Num"].min(), valid["Min_Salary_Num"].max(), 200)
        ax.plot(x_line, np.poly1d(z)(x_line), color=Colour.HIGHLIGHT,
                linewidth=2, linestyle="--", label="Trend")
        ax.legend(fontsize=9, facecolor=Colour.SURFACE,
                  edgecolor=Colour.ACCENT, labelcolor=Colour.TEXT_LIGHT)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x / 1000:.0f}k"))
    style_axis(ax, "Maas & Ilan Suresi Korelasyonu",
               xlabel="Min. Maas", ylabel="Ilan Suresi (Gun)")

    # Panel 6 — Recent activity table
    ax = axes[1][2]
    ax.set_facecolor(Colour.SURFACE)
    ax.axis("off")
    ax.set_title("Son 15 Gunluk Akis Ozeti", fontsize=15,
                 fontweight="bold", color=Colour.TEXT_LIGHT, pad=14, loc="left")

    col_labels = ["Tarih", "Gun", "Ilan", "Durum"]
    col_widths  = [0.28, 0.20, 0.18, 0.24]
    header_y, row_h = 0.96, 0.058

    xp = 0.02
    for label, w in zip(col_labels, col_widths):
        ax.text(xp + w / 2, header_y, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color=Colour.GOLD,
                transform=ax.transAxes)
        xp += w

    ax.plot([0.01, 0.99], [header_y - 0.025, header_y - 0.025],
            color=Colour.ACCENT, linewidth=1.2, transform=ax.transAxes, clip_on=False)

    for i, (_, row) in enumerate(daily_dc.iterrows()):
        y   = header_y - 0.05 - i * row_h
        rect = mpatches.FancyBboxPatch(
            (0.01, y - row_h * 0.45), 0.98, row_h * 0.9,
            boxstyle="round,pad=0.005", linewidth=0,
            facecolor=Colour.ACCENT if i % 2 == 0 else Colour.SURFACE,
            transform=ax.transAxes, clip_on=True,
        )
        ax.add_patch(rect)
        label, colour = classify_status(int(row["Sayi"]))
        vals   = [row["Tarih_Str"], row["Gun"], str(int(row["Sayi"])), label]
        cols   = [Colour.TEXT_LIGHT, Colour.TEXT_MUTED, Colour.TEXT_LIGHT, colour]
        xp = 0.02
        for val, col, w in zip(vals, cols, col_widths):
            ax.text(xp + w / 2, y, val, ha="center", va="center",
                    fontsize=9.5, color=col, transform=ax.transAxes)
            xp += w

    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=Colour.PRIMARY)
    print(f"PNG kaydedildi -> {output_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Plotly HTML renderer
# ─────────────────────────────────────────────────────────────────────────────

def _make_subplot_specs() -> list:
    """Return subplot specs: top row all xy; bottom row xy + two domain tables."""
    return [
        [{"type": "xy"},     {"type": "xy"},     {"type": "xy"}],
        [{"type": "xy"},     {"type": "domain"},  {"type": "domain"}],
    ]


def render_html(df: pd.DataFrame, output_path: str = "cambridge_dashboard.html",
                open_browser: bool = True) -> None:
    """Render an interactive HTML dashboard using Plotly."""
    title_col  = find_title_column(df)
    c_counts   = agg_closing_counts(df)
    top_depts  = agg_top_departments(df)
    avg_day    = agg_daily_avg(df)
    dept_sal   = agg_dept_salary(df)
    daily_dc   = agg_daily_counts(df)
    id2dept    = dept_id_to_name(df)

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "Yaklasan Ilan Kapanis Tarihleri",
            "En Cok Ilan Veren Bolumler",
            "Gunluk Ortalama Ilan Yayini",
            "Bolum Bazli Ortalama Maas (Min.)",
            "Yakinda Kapanan 10 Ilan",          # toggled by buttons
            "Son 15 Gunluk Akis Ozeti",
        ],
        horizontal_spacing=0.10,
        vertical_spacing=0.18,
        specs=_make_subplot_specs(),
    )

    # ── Trace 0: Closing dates bar ────────────────────────────────────────────
    fig.add_trace(go.Bar(
        x=c_counts.index.strftime("%d %b"),
        y=c_counts.values,
        marker_color=[Colour.HIGHLIGHT if i == 0 else Colour.BLUE
                      for i in range(len(c_counts))],
        customdata=c_counts.index.strftime("%d %B %Y"),
        hovertemplate="<b>%{customdata}</b><br>%{y} ilan kapaniyor<extra></extra>",
        name="",
    ), row=1, col=1)

    # ── Trace 1: Top departments bar ─────────────────────────────────────────
    full_labels_2 = [id2dept.get(did, str(did)) for did in top_depts.index]
    fig.add_trace(go.Bar(
        x=top_depts.values,
        y=top_depts.index.astype(str),
        orientation="h",
        marker_color=Colour.BLUE,
        customdata=full_labels_2,
        hovertemplate="<b>%{customdata}</b><br>%{x} ilan<extra></extra>",
        name="",
    ), row=1, col=2)

    # ── Trace 2: Daily rhythm scatter-line ────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=avg_day.index,
        y=avg_day.values,
        mode="lines+markers",
        line=dict(color=Colour.HIGHLIGHT, width=2.5),
        marker=dict(size=9, color=Colour.TEXT_LIGHT,
                    line=dict(color=Colour.HIGHLIGHT, width=2)),
        hovertemplate="<b>%{x}</b><br>Ort. %{y:.1f} ilan/gun<extra></extra>",
        name="",
    ), row=1, col=3)

    # ── Trace 3: Department salary bar ───────────────────────────────────────
    full_labels_4 = [id2dept.get(did, str(did)) for did in dept_sal.index]
    fig.add_trace(go.Bar(
        x=dept_sal.values,
        y=dept_sal.index.astype(str),
        orientation="h",
        marker_color=Colour.GOLD,
        customdata=full_labels_4,
        hovertemplate="<b>%{customdata}</b><br>Ort. £%{x:,.0f}<extra></extra>",
        name="",
    ), row=2, col=1)

    # ── Traces 4 & 5: Switchable job tables ──────────────────────────────────
    def _job_table(sub_df: pd.DataFrame, header_colour: str, visible: bool) -> go.Table:
        return go.Table(
            header=dict(
                values=["<b>Ilan Basligi</b>", "<b>Bolum</b>"],
                fill_color=header_colour,
                font=dict(color=Colour.TEXT_LIGHT, size=11),
                align="center",
                line_color=Colour.PRIMARY,
            ),
            cells=dict(
                values=[sub_df[title_col].tolist(), sub_df["Department"].tolist()],
                fill_color=[[Colour.SURFACE if i % 2 == 0 else Colour.PRIMARY
                             for i in range(len(sub_df))]],
                font=dict(color=Colour.TEXT_LIGHT, size=10),
                align=["left", "left"],
                line_color=Colour.ACCENT,
            ),
            visible=visible,
        )

    closes_top10 = (
        df[df["Closes"] >= pd.Timestamp.now()]
        .sort_values("Closes")
        .head(10)[[title_col, "Department", "Closes"]]
        .copy()
    )
    posted_top10 = (
        df.sort_values("Posted", ascending=False)
        .head(10)[[title_col, "Department", "Posted"]]
        .copy()
    )

    fig.add_trace(_job_table(closes_top10, Colour.HIGHLIGHT, visible=True),  row=2, col=2)
    fig.add_trace(_job_table(posted_top10, Colour.BLUE,      visible=False), row=2, col=2)

    # ── Trace 6: Recent 15-day activity table ─────────────────────────────────
    daily_dc["Durum"] = daily_dc["Sayi"].apply(lambda x: classify_status(int(x))[0])
    fig.add_trace(go.Table(
        header=dict(
            values=["<b>Tarih</b>", "<b>Gun</b>", "<b>Ilan</b>", "<b>Durum</b>"],
            fill_color=Colour.ACCENT,
            font=dict(color=Colour.GOLD, size=11),
            align="center",
            line_color=Colour.PRIMARY,
        ),
        cells=dict(
            values=[daily_dc["Tarih_Str"], daily_dc["Gun"],
                    daily_dc["Sayi"], daily_dc["Durum"]],
            fill_color=[[Colour.SURFACE if i % 2 == 0 else Colour.PRIMARY
                         for i in range(len(daily_dc))]],
            font=dict(color=Colour.TEXT_LIGHT, size=10),
            align="center",
            line_color=Colour.ACCENT,
        ),
    ), row=2, col=3)

    # ── Layout ────────────────────────────────────────────────────────────────
    # Annotation indices for subplot titles: 0-based, left-to-right, top-to-bottom
    PANEL5_ANNOTATION_IDX = 4   # "Yakinda Kapanan 10 Ilan" / "En Son Yayinlanan 10 Ilan"

    fig.update_layout(
        title=dict(
            text=(
                f"Cambridge University - Is Ilanlari Analiz Paneli<br>"
                f"<sub>Guncelleme: {datetime.now().strftime('%d %B %Y')} | "
                f"Toplam Ilan: {len(df):,} | "
                f"Benzersiz Bolum: {df['Department'].nunique()}</sub>"
            ),
            x=0.5,
            font=dict(size=20, color=Colour.TEXT_LIGHT),
        ),
        paper_bgcolor=Colour.PRIMARY,
        plot_bgcolor=Colour.SURFACE,
        font=dict(color=Colour.TEXT_LIGHT, family="DejaVu Sans"),
        showlegend=False,
        height=950,
        width=1600,
        hoverlabel=dict(
            bgcolor=Colour.SURFACE,
            bordercolor=Colour.HIGHLIGHT,
            font=dict(color=Colour.TEXT_LIGHT, size=12),
        ),
        updatemenus=[dict(
            type="buttons",
            direction="left",
            x=0.63, y=0.5,
            bgcolor=Colour.ACCENT,
            bordercolor=Colour.HIGHLIGHT,
            borderwidth=2,
            pad=dict(l=6, r=6, t=6, b=6),
            font=dict(color=Colour.TEXT_LIGHT, size=12),
            buttons=[
                dict(
                    label="1 — Yakinda Kapanan 10 Ilan",
                    method="update",
                    args=[
                        {"visible": [True, True, True, True, True, False, True]},
                        {f"annotations[{PANEL5_ANNOTATION_IDX}].text":
                             "<b>Yakinda Kapanan 10 Ilan</b>"},
                    ],
                ),
                dict(
                    label="2 — En Son Yayinlanan 10 Ilan",
                    method="update",
                    args=[
                        {"visible": [True, True, True, True, False, True, True]},
                        {f"annotations[{PANEL5_ANNOTATION_IDX}].text":
                             "<b>En Son Yayinlanan 10 Ilan</b>"},
                    ],
                ),
            ],
        )],
    )

    fig.update_xaxes(gridcolor=Colour.ACCENT, gridwidth=0.5)
    fig.update_yaxes(gridcolor=Colour.ACCENT, gridwidth=0.5)

    fig.write_html(output_path)
    print(f"HTML kaydedildi -> {output_path}  (tarayicida ac, hover calısır)")
    if open_browser:
        fig.show()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def create_job_analysis_dashboard(
    csv_path: str,
    png_out:  str = "cambridge_dashboard.png",
    html_out: str = "cambridge_dashboard.html",
    open_browser: bool = True,
) -> None:
    df = load_data(csv_path)
    render_png(df, output_path=png_out)
    render_html(df, output_path=html_out, open_browser=open_browser)


if __name__ == "__main__":
    create_job_analysis_dashboard("cambridge_jobs_archive.csv")
