"""
job_analyse_v6.py  —  Cambridge University Jobs Dashboard
Change vs v5:
  • Job Listings & Activity Flow panels are now REAL HTML <table> elements
    (not Plotly go.Table). Plotly go.Table cannot reliably fire functional
    <a href> clicks in cells (known plotly.js limitation) - a native HTML
    table has no such restriction, links just work.
  • Plotly figure now only holds the 4 real charts (2x2 grid).
  • Toggle between "Closing Soon" / "Recently Posted" is pure DOM (show/hide
    <tbody>), no Plotly.restyle needed anymore for that part.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import html as html_lib
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


class C:
    PRIMARY    = "#1a1a2e"
    SURFACE    = "#16213e"
    SURFACE2   = "#1e2a45"
    ACCENT     = "#0f3460"
    HIGHLIGHT  = "#e94560"
    GOLD       = "#f5a623"
    TEXT_LIGHT = "#eaeaea"
    TEXT_MUTED = "#a0aec0"
    GREEN      = "#48bb78"
    BLUE       = "#63b3ed"
    CMAP_MAIN  = LinearSegmentedColormap.from_list("main", ["#0f3460", "#e94560"])
    CMAP_WARM  = LinearSegmentedColormap.from_list("warm", ["#0f3460", "#f5a623"])

URGENCY_COLOURS = [
    (7,  C.HIGHLIGHT),
    (14, C.GOLD),
]
URGENCY_DEFAULT = C.BLUE

STATUS_THRESHOLDS = [(10, "Busy", C.HIGHLIGHT), (5, "Normal", C.GOLD)]
STATUS_DEFAULT    = ("Calm", C.GREEN)

DAY_ORDER      = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
DAYS_EN        = {"Monday":"Mon","Tuesday":"Tue","Wednesday":"Wed",
                  "Thursday":"Thu","Friday":"Fri","Saturday":"Sat","Sunday":"Sun"}
TITLE_CANDIDATES = ["Title","Job_Title","Position","Name","Job_Name"]

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


def resolve_csv_path(filename: str = "cambridge_jobs_archive.csv") -> Path:
    script_dir = Path(__file__).parent
    dynamic_path = script_dir / filename
    if dynamic_path.exists():
        print(f"Loading data from dynamic path: {dynamic_path.resolve()}")
        return dynamic_path
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


def _urgency_colour_list(index: pd.DatetimeIndex) -> list[str]:
    return [urgency_colour(d) for d in index]


def _gradient_hex(values: pd.Series, cmap) -> list[str]:
    norm = (values - values.min()) / (values.max() - values.min() + 1)
    def to_hex(rgba):
        return "#{:02x}{:02x}{:02x}".format(
            int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
    return [to_hex(cmap(v)) for v in norm]


# ─────────────────────────────────────────────────────────────────────────────
# Native HTML table builders (replace go.Table for anything that needs links)
# ─────────────────────────────────────────────────────────────────────────────

def _esc(val) -> str:
    return html_lib.escape(str(val))


def build_job_rows_html(sub_df: pd.DataFrame, title_col: str, date_col: str) -> str:
    """Builds <tr> rows for one job-listing table body. Real <a href> = real clicks."""
    has_link = "Link" in sub_df.columns
    rows = []
    for i, (_, row) in enumerate(sub_df.iterrows()):
        title_text = _esc(row[title_col])
        dept_text  = _esc(row["Department"])
        date_text  = row[date_col].strftime("%d %b %Y")
        link_val   = row["Link"] if has_link and pd.notna(row["Link"]) else None

        if link_val:
            title_cell = (
                f'<a href="{_esc(link_val)}" target="_blank" rel="noopener" class="job-link">'
                f'🔗 {title_text}</a>'
            )
        else:
            title_cell = title_text

        bg = C.SURFACE if i % 2 == 0 else C.PRIMARY
        rows.append(
            f'<tr style="background:{bg}">'
            f'<td class="col-title">{title_cell}</td>'
            f'<td class="col-dept">{dept_text}</td>'
            f'<td class="col-date">{date_text}</td>'
            f'</tr>'
        )
    return "".join(rows)


def build_activity_rows_html(daily_dc: pd.DataFrame) -> str:
    rows = []
    for i, (_, row) in enumerate(daily_dc.iterrows()):
        label, badge_col = classify_status(int(row["Count"]))
        bg = C.SURFACE if i % 2 == 0 else C.PRIMARY
        rows.append(
            f'<tr style="background:{bg}">'
            f'<td>{_esc(row["Date_Str"])}</td>'
            f'<td>{_esc(row["Day"])}</td>'
            f'<td>{int(row["Count"])}</td>'
            f'<td><span class="status-badge" style="color:{badge_col}; '
            f'background:{badge_col}26; border:1px solid {badge_col}66;">{label}</span></td>'
            f'</tr>'
        )
    return "".join(rows)

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

    # 3 Satır x 2 Sütun düzenine geçiyoruz (Böylece hiçbir grafik kaybolmuyor)
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Upcoming Closing Dates",
            "Top Recruiting Departments",
            "Daily Posting Rhythm",
            "Avg Min Salary by Department",
            "Salary vs. Visibility Window", # Eski kaybolan Bubble/Scatter grafiğiniz
            "" # Bu hücreyi boş bırakıyoruz çünkü HTML tablonuz buranın üzerine oturacak
        ],
        horizontal_spacing=0.10,
        vertical_spacing=0.12,
    )

    # 1. Grafik: Closing Dates (Row 1, Col 1)
    urg_cols   = _urgency_colour_list(c_counts.index)
    days_until = [(d - pd.Timestamp.now().normalize()).days for d in c_counts.index]
    fig.add_trace(go.Bar(
        x=c_counts.index.strftime("%d %b"), y=c_counts.values,
        marker_color=urg_cols,
        customdata=list(zip(c_counts.index.strftime("%d %B %Y"), days_until)),
        hovertemplate="<b>%{customdata[0]}</b><br>%{y} jobs closing · <i>%{customdata[1]} days left</i><extra></extra>",
        name="",
    ), row=1, col=1)

    # 2. Grafik: Top Departments (Row 1, Col 2)
    text_depts = [id2dept.get(did, str(did)) for did in top_depts.index]
    short_depts = [truncate(n, 24) for n in text_depts]
    cols_depts  = _gradient_hex(pd.Series(top_depts.values.astype(float)), C.CMAP_MAIN)
    sals_depts  = {did: df[df["Department_ID"] == did]["Min_Salary_Num"].mean() for did in top_depts.index}
    hover_sals  = [f"£{sals_depts[did]/1000:.0f}k" if pd.notna(sals_depts[did]) else "N/A" for did in top_depts.index]
    fig.add_trace(go.Bar(
        x=top_depts.values, y=short_depts, orientation="h",
        marker_color=cols_depts,
        customdata=list(zip(text_depts, hover_sals)),
        hovertemplate="<b>%{customdata[0]}</b><br>%{x} jobs · Avg salary: %{customdata[1]}<extra></extra>",
        name="",
    ), row=1, col=2)

    # 3. Grafik: Daily Rhythm (Row 2, Col 1)
    fig.add_trace(go.Scatter(
        x=avg_day.index, y=avg_day.values, mode="lines+markers",
        line=dict(color=C.HIGHLIGHT, width=2.5),
        marker=dict(size=9, color=C.TEXT_LIGHT, line=dict(color=C.HIGHLIGHT, width=2)),
        fill="tozeroy", fillcolor="rgba(233,69,96,0.10)",
        hovertemplate="<b>%{x}</b><br>Avg: %{y:.1f} jobs/day<extra></extra>",
        name="",
    ), row=2, col=1)

    # 4. Grafik: Dept Salary (Row 2, Col 2)
    dept_sal_names = [truncate(id2dept.get(did, str(did)), 24) for did in dept_sal.index]
    dept_sal_full  = [id2dept.get(did, str(did)) for did in dept_sal.index]
    sal_colours    = _gradient_hex(dept_sal, C.CMAP_WARM)
    fig.add_trace(go.Bar(
        x=dept_sal.values, y=dept_sal_names, orientation="h",
        marker_color=sal_colours,
        customdata=dept_sal_full,
        hovertemplate="<b>%{customdata}</b><br>Avg: £%{x:,.0f}<extra></extra>",
        name="",
    ), row=2, col=2)

    # 5. Grafik: Eski Kaybolan Salary vs Visibility Bubble Grafiği (Row 3, Col 1)
    valid = df.dropna(subset=["Min_Salary_Num", "Days_Open"])
    if len(valid) > 0:
        sizes = np.clip(valid["Days_Open"] * 1.5, 10, 150)
        fig.add_trace(go.Scatter(
            x=valid["Min_Salary_Num"], y=valid["Days_Open"], mode="markers",
            marker=dict(size=sizes, color=C.HIGHLIGHT, opacity=0.6, line=dict(color=C.ACCENT, width=0.5)),
            hovertemplate="Salary: £%{x:,.0f}<br>Days Open: %{y}<extra></extra>",
            name=""
        ), row=3, col=1)

    # Row 3, Col 2 alanını boş bırakıyoruz (Eksenleri gizliyoruz) çünkü HTML tablomuz buraya gelecek
    fig.update_xaxes(visible=False, row=3, col=2)
    fig.update_yaxes(visible=False, row=3, col=2)

    fig.update_layout(
        title=None, paper_bgcolor=C.PRIMARY, plot_bgcolor=C.SURFACE,
        font=dict(color=C.TEXT_LIGHT, family="DejaVu Sans"),
        showlegend=False, autosize=True,
        height=900, # Yükseklik grafik sayısına göre artırıldı
        margin=dict(t=40, b=10, l=40, r=40),
        hoverlabel=dict(bgcolor=C.SURFACE, bordercolor=C.HIGHLIGHT, font=dict(color=C.TEXT_LIGHT, size=12)),
    )
    fig.update_xaxes(showgrid=False, gridcolor=C.ACCENT, gridwidth=0.4)
    fig.update_yaxes(showgrid=True,  gridcolor=C.ACCENT, gridwidth=0.4)
    fig.update_xaxes(showgrid=True, row=2, col=1)
    fig.update_xaxes(showgrid=True, row=3, col=1)

    plotly_div = fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})

    # Veri hazırlama ve HTML oluşturma kısımları (Aşağısı v6 kodunuzla aynı kalıyor)
    target_cols = [title_col, "Department", "Closes"]
    if "Link" in df.columns: target_cols.append("Link")
    closes_top10 = df[df["Closes"] >= pd.Timestamp.now()].sort_values("Closes").head(10)[target_cols].copy()

    posted_cols = [title_col, "Department", "Posted"]
    if "Link" in df.columns: posted_cols.append("Link")
    posted_top10 = df.sort_values("Posted", ascending=False).head(10)[posted_cols].copy()

    closing_rows_html  = build_job_rows_html(closes_top10, title_col, "Closes")
    posted_rows_html   = build_job_rows_html(posted_top10, title_col, "Posted")
    activity_rows_html = build_activity_rows_html(daily_dc)


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

    today_str  = datetime.now().strftime('%d %B %Y')
    total_ilan = len(df)
    n_dept     = df['Department'].nunique()

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
  }}
  .header {{ text-align: center; padding: 18px 20px 8px; border-bottom: 1px solid {C.ACCENT}; }}
  .header h1 {{ font-size: 22px; font-weight: 700; }}
  .header .subtitle {{ font-size: 12px; color: {C.TEXT_MUTED}; margin-top: 4px; }}

  .kpi-strip {{ display: flex; gap: 10px; padding: 12px 20px; border-bottom: 1px solid {C.ACCENT}; }}
  .kpi-card {{ flex: 1; background: {C.SURFACE2}; border: 1.5px solid; border-radius: 6px;
               padding: 10px 12px 12px; position: relative; overflow: hidden; min-width: 0; }}
  .kpi-bar {{ position: absolute; top: 0; left: 0; right: 0; height: 3px; border-radius: 6px 6px 0 0; }}
  .kpi-value {{ font-size: 28px; font-weight: 700; line-height: 1.1; margin-top: 6px; }}
  .kpi-label {{ font-size: 11px; color: {C.TEXT_MUTED}; margin-top: 5px; }}

  .chart-area {{ width: 100%; padding: 0 20px; }}

  .panels {{ display: flex; gap: 16px; padding: 4px 20px 24px; flex-wrap: wrap; }}
  .panel {{ flex: 1; min-width: 320px; background: {C.SURFACE2}; border: 1px solid {C.ACCENT};
            border-radius: 8px; overflow: hidden; }}
  .panel-header {{ display: flex; align-items: center; justify-content: space-between;
                   padding: 10px 14px; border-bottom: 1px solid {C.ACCENT}; }}
  .panel-title {{ font-size: 13px; font-weight: bold; }}
  .btn-group {{ display: flex; gap: 4px; }}
  .table-toggle-btn {{
    background: {C.ACCENT}; border: 1px solid {C.ACCENT}; color: {C.TEXT_MUTED};
    padding: 3px 8px; font-size: 10px; font-weight: bold; cursor: pointer;
    border-radius: 4px; transition: all 0.2s ease;
  }}
  .table-toggle-btn:hover {{ color: {C.TEXT_LIGHT}; border-color: {C.HIGHLIGHT}; }}
  .table-toggle-btn.active {{ background: {C.HIGHLIGHT}; border-color: {C.HIGHLIGHT}; color: {C.TEXT_LIGHT}; }}

  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  thead th {{
    text-align: left; padding: 8px 12px; background: {C.ACCENT}; color: {C.GOLD};
    font-size: 11px; position: sticky; top: 0;
  }}
  tbody td {{ padding: 7px 12px; border-bottom: 1px solid {C.ACCENT}44; vertical-align: middle; }}
  .table-scroll {{ max-height: 340px; overflow-y: auto; }}
  .col-date, .col-dept {{ color: {C.TEXT_MUTED}; white-space: nowrap; }}
  .job-link {{ color: {C.GOLD}; text-decoration: none; font-weight: bold; }}
  .job-link:hover {{ text-decoration: underline; }}
  .status-badge {{ padding: 2px 8px; border-radius: 10px; font-size: 10px; font-weight: bold; }}
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

<div class="chart-area">
{plotly_div}
</div>

<div class="panels">

  <div class="panel" id="job-panel">
    <div class="panel-header">
      <div class="panel-title" id="dynamic-panel-title">10 Jobs Closing Soon</div>
      <div class="btn-group">
        <button class="table-toggle-btn active" id="btn-closing" onclick="switchTable('closing')">📅 Closing Soon</button>
        <button class="table-toggle-btn" id="btn-recent" onclick="switchTable('recent')">🆕 Recently Posted</button>
      </div>
    </div>
    <div class="table-scroll">
      <table>
        <thead><tr><th>Job Title (click to open)</th><th>Department</th><th>Date</th></tr></thead>
        <tbody id="tbody-closing">{closing_rows_html}</tbody>
        <tbody id="tbody-recent" style="display:none">{posted_rows_html}</tbody>
      </table>
    </div>
  </div>

  <div class="panel" id="activity-panel">
    <div class="panel-header">
      <div class="panel-title">Activity Flow (Last 10 Days)</div>
    </div>
    <div class="table-scroll">
      <table>
        <thead><tr><th>Date</th><th>Day</th><th>Jobs</th><th>Status</th></tr></thead>
        <tbody>{activity_rows_html}</tbody>
      </table>
    </div>
  </div>

</div>

<script>
  function switchTable(mode) {{
    var titleEl = document.getElementById('dynamic-panel-title');
    var btnClosing = document.getElementById('btn-closing');
    var btnRecent = document.getElementById('btn-recent');
    var tbClosing = document.getElementById('tbody-closing');
    var tbRecent = document.getElementById('tbody-recent');

    if (mode === 'closing') {{
      titleEl.innerText = "10 Jobs Closing Soon";
      btnClosing.classList.add('active');
      btnRecent.classList.remove('active');
      tbClosing.style.display = '';
      tbRecent.style.display = 'none';
    }} else {{
      titleEl.innerText = "10 Most Recently Posted Jobs";
      btnClosing.classList.remove('active');
      btnRecent.classList.add('active');
      tbClosing.style.display = 'none';
      tbRecent.style.display = '';
    }}
  }}
</script>
</body>
</html>
"""

    out_p = Path(output_path)
    out_p.write_text(full_html, encoding="utf-8")
    print(f"HTML saved -> {out_p.resolve()}")

    if open_browser:
        import webbrowser
        webbrowser.open(out_p.absolute().as_uri())


if __name__ == "__main__":
    csv_file = resolve_csv_path()
    data = load_data(csv_file)
    render_html(data, open_browser=False)
    print("OK - no errors")
