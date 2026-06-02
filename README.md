# Cambridge University Job Listings — Scraper & Dashboard

This project consists of two main Python scripts: one scrapes job listings from the University of Cambridge website and saves them to a CSV archive, the other generates a visual analytics dashboard from that archive.

---

## Files

| File | Description |
|---|---|
| `scraper.py` | Web scraper — fetches, cleans, and saves listings to CSV |
| `job_analyse
.py` | Dashboard — generates a PNG and HTML analytics panel from the CSV |
| `job_analyse_v1.py` | Dashboard — generates a PNG and HTML analytics panel from the CSV |
| `cambridge_jobs_archive.csv` | Cumulative job archive (auto-generated) |
| `cambridge_dashboard.png` | Static output (matplotlib) |
| `cambridge_dashboard.html` | Interactive output (Plotly, open in browser) |

---

## Installation

```bash
pip install requests beautifulsoup4 pandas plotly matplotlib numpy
```

---

## Usage

### 1. Scraping Data

```bash
python scraper.py
```

- Scrapes the University of Cambridge job listings page
- Normalises dates to `DD/MM/YYYY` format
- Extracts salaries in `£33,002` format
- Appends new listings to `cambridge_jobs_archive.csv` (duplicate check via `Job_ID`)
- If Excel has the file open, attempts to close it automatically; otherwise saves to a timestamped temporary file

### 2. Generating the Dashboard

```bash
python job_analyse_v3.py
```

Reads `cambridge_jobs_archive.csv` and produces two outputs:

**`cambridge_dashboard.png`** — 6-panel static visual:
1. Upcoming job closing dates
2. Most active departments by listing count
3. Average daily posting volume (weekly pattern)
4. Average minimum salary by department
5. Salary vs. listing duration correlation scatter plot
6. Last 15 days activity summary table

**`cambridge_dashboard.html`** — Same 6 panels, interactive with Plotly:
- Hover tooltips on all charts
- Panel 5 toggle button: **"10 Closing Soon"** / **"10 Most Recently Posted"**

---

## CSV Schema

| Column | Example | Description |
|---|---|---|
| `Job_ID` | `AB12345` | Unique job ID (extracted from URL) |
| `Title` | `Research Associate` | Job title |
| `Department_ID` | `AB` | Short department code |
| `Department` | `Biochemistry` | Full department name |
| `Min_Salary` | `£33,002` | Minimum salary |
| `Posted` | `15/05/2025` | Date posted |
| `Closes` | `30/05/2025` | Closing date |
| `Retrieved_At` | `2025-05-15` | Date the record was fetched |

---

## Notes

- The scraper preserves existing records on every run; listings with a duplicate `Job_ID` are never re-added.
- Dates in older CSV files are also normalised on load (migration).
- The interactive dashboard (`cambridge_dashboard.html`) can be opened directly in any browser — no server required.
