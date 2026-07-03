import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

plt.rcParams.update({
    "font.family": "DejaVu Sans", "text.color": TEXT_LIGHT,
    "axes.labelcolor": TEXT_MUTED, "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED, "figure.facecolor": PRIMARY,
    "axes.facecolor": SURFACE, "axes.edgecolor": ACCENT,
    "grid.color": ACCENT, "grid.alpha": 0.3, "grid.linewidth": 0.6,
})

def style_axis(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=15, fontweight="bold", color=TEXT_LIGHT, pad=14, loc="left")
    ax.set_xlabel(xlabel, fontsize=11, color=TEXT_MUTED, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=11, color=TEXT_MUTED, labelpad=8)
    ax.tick_params(labelsize=10, colors=TEXT_MUTED, length=4)
    for spine in ax.spines.values():
        spine.set_edgecolor(ACCENT); spine.set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def add_value_labels(ax, bars, fmt="{:.0f}", color=TEXT_LIGHT, fontsize=9, offset=2):
    for bar in bars:
        w = bar.get_width() if bar.get_width() > 0 else bar.get_height()
        x_pos = bar.get_x() + bar.get_width() + offset if bar.get_width() > bar.get_height() \
                else bar.get_x() + bar.get_width() / 2
        y_pos = bar.get_y() + bar.get_height() / 2 if bar.get_width() > bar.get_height() \
                else bar.get_y() + bar.get_height() + offset
        ha = "left" if bar.get_width() > bar.get_height() else "center"
        va = "center" if bar.get_width() > bar.get_height() else "bottom"
        ax.annotate(fmt.format(w), xy=(x_pos, y_pos),
                    fontsize=fontsize, color=color, ha=ha, va=va, fontweight="bold")

def create_job_analysis_dashboard(csv_path):
    df = pd.read_csv(csv_path)
    df["Min_Salary_Num"] = pd.to_numeric(
        df["Min_Salary"].astype(str)
          .str.replace("£","").str.replace(",","")
          .replace("N/A", pd.NA), errors="coerce")
    df["Closes"] = pd.to_datetime(df["Closes"], dayfirst=True, errors="coerce", format="mixed")
    df["Posted"] = pd.to_datetime(df["Posted"], dayfirst=True, errors="coerce", format="mixed")
    df["Days_Open"] = (df["Closes"] - df["Posted"]).dt.days
    df = df.dropna(subset=["Closes","Posted"])
    df["Day_Name"] = df["Posted"].dt.day_name()

    days_tr = {"Monday":"Pzt","Tuesday":"Sal","Wednesday":"Car",
               "Thursday":"Per","Friday":"Cum","Saturday":"Cmt","Sunday":"Paz"}
    id_to_dept = df.drop_duplicates("Department_ID").set_index("Department_ID")["Department"]

    # ── Baslik sutununu bul ───────────────────────────────────────────────────
    title_col = None
    for candidate in ["Title", "Job_Title", "Position", "Name", "Job_Name"]:
        if candidate in df.columns:
            title_col = candidate
            break
    if title_col is None:
        title_col = df.columns[0]  # bulamazsa ilk sutunu kullan

    # ════════════════════════════════════════════════════════════════
    # PNG — matplotlib
    # ════════════════════════════════════════════════════════════════
    fig_mpl = plt.figure(figsize=(28, 18), dpi=120, facecolor=PRIMARY)
    gs = gridspec.GridSpec(2, 3, figure=fig_mpl,
                           left=0.06, right=0.97, top=0.82, bottom=0.07,
                           wspace=0.38, hspace=0.48)
    axes = [[fig_mpl.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]

    fig_mpl.text(0.5, 0.95, "Cambridge University - Is Ilanlari Analiz Paneli",
                 ha="center", fontsize=22, fontweight="bold", color=TEXT_LIGHT)
    fig_mpl.text(0.5, 0.90,
                 f"Guncelleme: {datetime.now().strftime('%d %B %Y')}  |  "
                 f"Toplam Ilan: {len(df):,}  |  "
                 f"Benzersiz Bolum: {df['Department'].nunique()}",
                 ha="center", fontsize=11, color=TEXT_MUTED)

    # 1. Kapanislar
    ax = axes[0][0]

    # 🎯 FIX: Filter out past dates so you only get deadlines starting from today onwards
    # 1. Kapanislar
    ax = axes[0][0]

    # 🎯 1. ADIM: Tarih sütununu sadece "Gün" bazına indirge (Saatleri sıfırla)
    # Bu sayede geçmiş filtrelemesi kusursuz çalışır.
    closes_normalized = pd.to_datetime(df["Closes"]).dt.normalize()

    # 🎯 2. ADIM: Bugünü al ve kesinlikle büyük VEYA eşit olanları filtrele
    today = pd.Timestamp.today().normalize()
    upcoming_closes = df[closes_normalized >= today].copy()

    # 🎯 3. ADIM: Sayma ve Sıralama işlemini temiz normalize edilmiş tarihle yap
    # Böylece her güne ait ilanlar tek bir barda doğru toplanır
    upcoming_closes["Closes_Normalized"] = upcoming_closes["Closes"].dt.normalize()
    closing_counts = upcoming_closes["Closes_Normalized"].value_counts().sort_index().head(10)

    # 🎯 4. ADIM: Matplotlib x-ekseninde string sıralama hatası yapmasın diye
    # Önce barları çizdiriyoruz, label'ları sonra giydiriyoruz
    x_ticks = range(len(closing_counts))
    x_labels = closing_counts.index.strftime("%d %b")

    ax.bar(x_ticks, closing_counts.values,
           color=[HIGHLIGHT if i == 0 else BLUE for i in range(len(closing_counts))],
           edgecolor=ACCENT, linewidth=0.6, width=0.65)

    # Ticks ve Label eşlemesini zorunlu yapıyoruz:
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)

    style_axis(ax, "Yaklasan Ilan Kapanis Tarihleri", ylabel="Ilan Sayisi")
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylim(0, closing_counts.max() * 1.18)

    # 2. En Aktif Bolumler
    ax = axes[0][1]
    top_depts = df["Department_ID"].value_counts().head(10)
    norm_vals = top_depts.values / top_depts.max()
    bars2 = ax.barh(top_depts.index, top_depts.values,
                    color=[CMAP_COOL(v) for v in norm_vals],
                    edgecolor=ACCENT, linewidth=0.5, height=0.65)
    add_value_labels(ax, bars2, offset=0.3)
    style_axis(ax, "En Cok Ilan Veren Bolumler", xlabel="Ilan Sayisi")
    ax.set_xlim(0, top_depts.max() * 1.20)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.invert_yaxis()

    # 3. Haftalik Rutin
    ax = axes[0][2]
    avg_per_day = (
        df.groupby(["Posted","Day_Name"]).size().reset_index(name="Daily_Count")
          .groupby("Day_Name")["Daily_Count"].mean()
          .reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    )
    avg_per_day.index = [days_tr[d] for d in avg_per_day.index]
    ax.fill_between(avg_per_day.index, avg_per_day.values, alpha=0.18, color=HIGHLIGHT)
    ax.plot(avg_per_day.index, avg_per_day.values, marker="o", linewidth=2.5, markersize=9,
            color=HIGHLIGHT, markerfacecolor=TEXT_LIGHT, markeredgewidth=2)
    for i,(x,y) in enumerate(zip(avg_per_day.index, avg_per_day.values)):
        if pd.notna(y):
            ax.annotate(f"{y:.1f}", xy=(i,y), xytext=(0,10),
                        textcoords="offset points", ha="center",
                        fontsize=9, color=GOLD, fontweight="bold")
    style_axis(ax, "Gunluk Ortalama Ilan Yayini", ylabel="Ort. Ilan / Gun")
    ax.set_ylim(0, avg_per_day.max() * 1.25)

    # 4. Bolum Maas
    ax = axes[1][0]
    dept_salary_by_id = (
        df.groupby("Department_ID")["Min_Salary_Num"]
          .mean().dropna().sort_values(ascending=False).head(10)
    )
    norm_vals = dept_salary_by_id.values / dept_salary_by_id.max()
    bars4 = ax.barh(dept_salary_by_id.index, dept_salary_by_id.values,
                    color=[CMAP_WARM(v) for v in norm_vals],
                    edgecolor=ACCENT, linewidth=0.5, height=0.65)
    for bar in bars4:
        val = bar.get_width()
        ax.annotate(f"£{val/1000:.0f}k",
                    xy=(val + dept_salary_by_id.max()*0.01, bar.get_y()+bar.get_height()/2),
                    fontsize=9, color=GOLD, fontweight="bold", ha="left", va="center")
    style_axis(ax, "Bolum Bazli Ortalama Maas (Min.)", xlabel="Maas")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"£{x/1000:.0f}k"))
    ax.set_xlim(0, dept_salary_by_id.max() * 1.18)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.invert_yaxis()

    # 5. Scatter
    ax = axes[1][1]
    valid = df.dropna(subset=["Min_Salary_Num","Days_Open"])
    ax.scatter(valid["Min_Salary_Num"], valid["Days_Open"],
               alpha=0.45, s=60, color=BLUE, edgecolors=ACCENT, linewidths=0.4)
    if len(valid) > 2:
        z = np.polyfit(valid["Min_Salary_Num"], valid["Days_Open"], 1)
        x_line = np.linspace(valid["Min_Salary_Num"].min(), valid["Min_Salary_Num"].max(), 200)
        ax.plot(x_line, np.poly1d(z)(x_line), color=HIGHLIGHT, linewidth=2,
                linestyle="--", label="Trend")
        ax.legend(fontsize=9, facecolor=SURFACE, edgecolor=ACCENT, labelcolor=TEXT_LIGHT)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"£{x/1000:.0f}k"))
    style_axis(ax, "Maas & Ilan Suresi Korelasyonu",
               xlabel="Min. Maas", ylabel="Ilan Suresi (Gun)")

    # 6. Tablo
    ax = axes[1][2]
    ax.set_facecolor(SURFACE); ax.axis("off")
    ax.set_title("Son 15 Gunluk Akis Ozeti", fontsize=15,
                 fontweight="bold", color=TEXT_LIGHT, pad=14, loc="left")
    daily_counts = (df["Posted"].value_counts().sort_index(ascending=False)
                      .reset_index().head(15))
    daily_counts.columns = ["Tarih","Sayi"]
    daily_counts["Gun"] = daily_counts["Tarih"].dt.day_name().map(days_tr)
    daily_counts["Tarih_Str"] = daily_counts["Tarih"].dt.strftime("%d %b")
    def durum(x):
        if x > 10: return ("Yogun", HIGHLIGHT)
        if x > 5:  return ("Normal", GOLD)
        return ("Sakin", GREEN)
    col_labels = ["Tarih","Gun","Ilan","Durum"]
    col_widths  = [0.28, 0.20, 0.18, 0.24]
    header_y = 0.96; row_h = 0.058
    xp = 0.02
    for label,w in zip(col_labels,col_widths):
        ax.text(xp+w/2, header_y, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color=GOLD, transform=ax.transAxes)
        xp += w
    ax.plot([0.01,0.99],[header_y-0.025,header_y-0.025],
            color=ACCENT, linewidth=1.2, transform=ax.transAxes, clip_on=False)
    for i,(_,row) in enumerate(daily_counts.iterrows()):
        y = header_y - 0.05 - i*row_h
        rect = mpatches.FancyBboxPatch(
            (0.01, y-row_h*0.45), 0.98, row_h*0.9,
            boxstyle="round,pad=0.005", linewidth=0,
            facecolor=ACCENT if i%2==0 else SURFACE,
            transform=ax.transAxes, clip_on=True)
        ax.add_patch(rect)
        dl, dc = durum(row["Sayi"])
        vals = [row["Tarih_Str"], row["Gun"], str(int(row["Sayi"])), dl]
        cols = [TEXT_LIGHT, TEXT_MUTED, TEXT_LIGHT, dc]
        xp = 0.02
        for val,col,w in zip(vals,cols,col_widths):
            ax.text(xp+w/2, y, val, ha="center", va="center",
                    fontsize=9.5, color=col, transform=ax.transAxes)
            xp += w

   # plt.savefig("cambridge_dashboard.png", dpi=180, bbox_inches="tight", facecolor=PRIMARY)
    print("PNG kaydedildi -> cambridge_dashboard.png")
    plt.close()

    # ════════════════════════════════════════════════════════════════
    # HTML — plotly (hover + switch buton calısır)
    # ════════════════════════════════════════════════════════════════
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "Yaklasan Ilan Kapanis Tarihleri",
            "En Cok Ilan Veren Bolumler",
            "Gunluk Ortalama Ilan Yayini",
            "Bolum Bazli Ortalama Maas (Min.)",
            "Yakinda Kapanan 10 Ilan",        # 5. baslik (switch ile degisir)
            "Son 15 Gunluk Akis Ozeti",
        ],
        horizontal_spacing=0.10,
        vertical_spacing=0.18,
        specs=[
            [{"type":"xy"},     {"type":"xy"},     {"type":"xy"}],
            [{"type":"xy"},     {"type":"domain"},  {"type":"domain"}],
        ],
    )

    # Trace 0 — 1. Kapanislar
    closing_counts = df["Closes"].value_counts().sort_index().head(10)
    fig.add_trace(go.Bar(
        x=closing_counts.index.strftime("%d %b"),
        y=closing_counts.values,
        marker_color=[HIGHLIGHT if i==0 else BLUE for i in range(len(closing_counts))],
        customdata=closing_counts.index.strftime("%d %B %Y"),
        hovertemplate="<b>%{customdata}</b><br>%{y} ilan kapaniyor<extra></extra>",
        name="",
    ), row=1, col=1)

    # Trace 1 — 2. En Aktif Bolumler
    top_depts     = df["Department_ID"].value_counts().head(10)
    full_labels_2 = [id_to_dept.get(did, str(did)) for did in top_depts.index]
    fig.add_trace(go.Bar(
        x=top_depts.values, y=top_depts.index.astype(str),
        orientation="h", marker_color=BLUE,
        customdata=full_labels_2,
        hovertemplate="<b>%{customdata}</b><br>%{x} ilan<extra></extra>",
        name="",
    ), row=1, col=2)

    # Trace 2 — 3. Haftalik Rutin
    avg_per_day = (
        df.groupby(["Posted","Day_Name"]).size().reset_index(name="Daily_Count")
          .groupby("Day_Name")["Daily_Count"].mean()
          .reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    )
    avg_per_day.index = [days_tr[d] for d in avg_per_day.index]
    fig.add_trace(go.Scatter(
        x=avg_per_day.index, y=avg_per_day.values,
        mode="lines+markers",
        line=dict(color=HIGHLIGHT, width=2.5),
        marker=dict(size=9, color=TEXT_LIGHT, line=dict(color=HIGHLIGHT, width=2)),
        hovertemplate="<b>%{x}</b><br>Ort. %{y:.1f} ilan/gun<extra></extra>",
        name="",
    ), row=1, col=3)

    # Trace 3 — 4. Bolum Maas
    dept_salary = (
        df.groupby("Department_ID")["Min_Salary_Num"]
          .mean().dropna().sort_values(ascending=False).head(10)
    )
    full_labels_4 = [id_to_dept.get(did, str(did)) for did in dept_salary.index]
    fig.add_trace(go.Bar(
        x=dept_salary.values, y=dept_salary.index.astype(str),
        orientation="h", marker_color=GOLD,
        customdata=full_labels_4,
        hovertemplate="<b>%{customdata}</b><br>Ort. £%{x:,.0f}<extra></extra>",
        name="",
    ), row=2, col=1)

    # Trace 4 — 5a. Yakinda Kapanan 10 Ilan (Close Date, varsayilan ACIK)
    closes_top10 = (
        df[df["Closes"] >= pd.Timestamp.now()]
          .sort_values("Closes")
          .head(10)[[title_col, "Department", "Closes"]]
          .copy()
    )
    closes_top10["Closes_Str"] = closes_top10["Closes"].dt.strftime("%d %b %Y")

    fig.add_trace(go.Table(
        header=dict(
            values=["<b>Ilan Basligi</b>", "<b>Bolum</b>", "<b>Kapanis Tarihi</b>"],
            fill_color=HIGHLIGHT,
            font=dict(color=TEXT_LIGHT, size=11),
            align="center",
            line_color=PRIMARY,
        ),
        cells=dict(
            values=[
                closes_top10[title_col].tolist(),
                closes_top10["Department"].tolist(),
                closes_top10["Closes_Str"].tolist(),
            ],
            fill_color=[[SURFACE if i%2==0 else PRIMARY for i in range(len(closes_top10))]],
            font=dict(color=TEXT_LIGHT, size=10),
            align=["left", "left", "center"],
            line_color=ACCENT,
        ),
        visible=True,
    ), row=2, col=2)

    # Trace 5 — 5b. En Son Yayinlanan 10 Ilan (Post Date, varsayilan GIZLI)
    posted_top10 = (
        df.sort_values("Posted", ascending=False)
          .head(10)[[title_col, "Department", "Posted"]]
          .copy()
    )
    posted_top10["Posted_Str"] = posted_top10["Posted"].dt.strftime("%d %b %Y")

    fig.add_trace(go.Table(
        header=dict(
            values=["<b>Ilan Basligi</b>", "<b>Bolum</b>", "<b>Yayin Tarihi</b>"],
            fill_color=BLUE,
            font=dict(color=TEXT_LIGHT, size=11),
            align="center",
            line_color=PRIMARY,
        ),
        cells=dict(
            values=[
                posted_top10[title_col].tolist(),
                posted_top10["Department"].tolist(),
                posted_top10["Posted_Str"].tolist(),
            ],
            fill_color=[[SURFACE if i%2==0 else PRIMARY for i in range(len(posted_top10))]],
            font=dict(color=TEXT_LIGHT, size=10),
            align=["left", "left", "center"],
            line_color=ACCENT,
        ),
        visible=False,
    ), row=2, col=2)

    # Trace 6 — 6. Son 15 Gun Tablo
    daily_counts2 = (df["Posted"].value_counts().sort_index(ascending=False)
                       .reset_index().head(15))
    daily_counts2.columns = ["Tarih","Sayi"]
    daily_counts2["Gun"] = daily_counts2["Tarih"].dt.day_name().map(days_tr)
    daily_counts2["Tarih_Str"] = daily_counts2["Tarih"].dt.strftime("%d %b")
    def durum2(x):
        if x > 10: return "Yogun"
        if x > 5:  return "Normal"
        return "Sakin"
    fig.add_trace(go.Table(
        header=dict(
            values=["<b>Tarih</b>","<b>Gun</b>","<b>Ilan</b>","<b>Durum</b>"],
            fill_color=ACCENT,
            font=dict(color=GOLD, size=11),
            align="center",
            line_color=PRIMARY,
        ),
        cells=dict(
            values=[
                daily_counts2["Tarih_Str"],
                daily_counts2["Gun"],
                daily_counts2["Sayi"],
                daily_counts2["Sayi"].apply(durum2),
            ],
            fill_color=[[SURFACE if i%2==0 else PRIMARY for i in range(len(daily_counts2))]],
            font=dict(color=TEXT_LIGHT, size=10),
            align="center",
            line_color=ACCENT,
        ),
    ), row=2, col=3)

    # Trace sirasi:
    # 0: Bar-Kapanislar  1: Bar-Bolumler  2: Scatter-Haftalik  3: Bar-Maas
    # 4: Table-ClosesTop10  5: Table-PostedTop10  6: Table-Son15Gun

    fig.update_layout(
        title=dict(
            text=(
                f"Cambridge University - Is Ilanlari Analiz Paneli<br>"
                f"<sub>Guncelleme: {datetime.now().strftime('%d %B %Y')} | "
                f"Toplam Ilan: {len(df):,} | "
                f"Benzersiz Bolum: {df['Department'].nunique()}</sub>"
            ),
            x=0.5,
            font=dict(size=20, color=TEXT_LIGHT),
        ),
        paper_bgcolor=PRIMARY,
        plot_bgcolor=SURFACE,
        font=dict(color=TEXT_LIGHT, family="DejaVu Sans"),
        showlegend=False,
        height=950,
        width=1600,
        hoverlabel=dict(
            bgcolor=SURFACE,
            bordercolor=HIGHLIGHT,
            font=dict(color=TEXT_LIGHT, size=12),
        ),
        updatemenus=[dict(
            type="buttons",
            direction="left",
            x=0.63,
            y=0.5,
            bgcolor=ACCENT,
            bordercolor=HIGHLIGHT,
            borderwidth=2,
            pad=dict(l=6, r=6, t=6, b=6),
            font=dict(color=TEXT_LIGHT, size=12),
            buttons=[
                dict(
                    label="1 — Yakinda Kapanan 10 Ilan",
                    method="update",
                    args=[
                        {"visible": [True, True, True, True, True, False, True]},
                        {"annotations[4].text": "<b>Yakinda Kapanan 10 Ilan</b>"},
                    ],
                ),
                dict(
                    label="2 — En Son Yayinlanan 10 Ilan",
                    method="update",
                    args=[
                        {"visible": [True, True, True, True, False, True, True]},
                        {"annotations[4].text": "<b>En Son Yayinlanan 10 Ilan</b>"},
                    ],
                ),
            ],
        )],
    )

    fig.update_xaxes(gridcolor=ACCENT, gridwidth=0.5)
    fig.update_yaxes(gridcolor=ACCENT, gridwidth=0.5)

    #fig.write_html("cambridge_dashboard.html")
    print("HTML kaydedildi -> cambridge_dashboard.html  (tarayicida ac, hover calısır)")
    fig.show()


# ── Calistir ──────────────────────────────────────────────────────────────────
create_job_analysis_dashboard("cambridge_jobs_archive.csv")