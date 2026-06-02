import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import re
import glob
from datetime import datetime


# --- 1. FONKSİYON: TARİH NORMALIZE ETME ---
def normalize_date(raw_text):
    """
    Herhangi bir formattaki tarih metnini standart 'DD/MM/YYYY' formatına çevirir.
    Tanınamayan değerleri 'N/A' olarak işaretler.

    Desteklenen formatlar:
        '15 May 2025'       → '15/05/2025'
        '15-05-2025'        → '15/05/2025'
        '2025-05-15'        → '15/05/2025'  (ISO)
        '15/05/2025'        → '15/05/2025'  (zaten doğru)
        '05/15/2025'        → algılanır, dönüştürülür
    """
    if not raw_text or str(raw_text).strip() in ('', 'N/A', 'nan'):
        return 'N/A'

    text = str(raw_text).strip()

    # Denenecek formatlar — en yaygından en azına doğru
    FORMATS = [
        '%d/%m/%Y',  # 15/05/2025
        '%d %B %Y',  # 15 May 2025
        '%d %b %Y',  # 15 May 2025 (kısa ay)
        '%Y-%m-%d',  # 2025-05-15  (ISO)
        '%d-%m-%Y',  # 15-05-2025
        '%m/%d/%Y',  # 05/15/2025  (ABD formatı)
        '%d.%m.%Y',  # 15.05.2025
        '%B %d, %Y',  # May 15, 2025
        '%b %d, %Y',  # May 15, 2025 (kısa ay)
    ]

    for fmt in FORMATS:
        try:
            return datetime.strptime(text, fmt).strftime('%d/%m/%Y')
        except ValueError:
            continue

    # Hiçbiri tutmadıysa pandas ile son bir deneme (esnek parser)
    try:
        return pd.to_datetime(text, dayfirst=True).strftime('%d/%m/%Y')
    except Exception:
        print(f"  [UYARI] Tarih dönüştürülemedi: '{text}'")
        return 'N/A'


# --- 2. FONKSİYON: MAAŞ AYIKLAMA VE FORMATLAMA ---
def extract_min_salary_formatted(salary_text):
    """
    Metindeki tüm maaşları bulur, en küçüğünü seçer ve '£33,002' formatında döner.
    """
    if not salary_text or any(x in salary_text for x in ['Competitive', 'not specified']):
        return "N/A"

    numbers = re.findall(r'£(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', salary_text)
    clean_numbers = []

    for num in numbers:
        try:
            val = int(num.split('.')[0].replace(',', ''))
            if 10000 < val < 300000:
                clean_numbers.append(val)
        except ValueError:
            continue

    if clean_numbers:
        min_val = min(clean_numbers)
        return f"£{min_val:,}"
    return "N/A"


# --- 3. FONKSİYON: DİĞER TEMİZLİK ARAÇLARI ---
def clean_job_id(raw_text):
    if not raw_text:
        return "N/A"
    return raw_text.split('-')[-1]


# --- 4. FONKSİYON: VERİ ÇEKME VE İŞLEME ---
def run_scraper():
    URL = "https://www.cam.ac.uk/jobs/search?search_api_views_fulltext=&order=created&sort=desc"
    headers = {'User-Agent': 'Mozilla/5.0'}

    print(f"--- İşlem Başlatıldı: {datetime.now().strftime('%H:%M:%S')} ---")

    try:
        response = requests.get(URL, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
    except Exception as e:
        print(f"Hata: Sayfaya ulaşılamadı -> {e}")
        return

    table = soup.find('table')
    if not table:
        print("Tablo bulunamadı.")
        return

    jobs_list = []
    rows = table.find_all('tr')[1:]

    for i, row in enumerate(rows):
        cols = row.find_all('td')
        if len(cols) >= 6:
            salary_raw = cols[2].get_text(separator=' ', strip=True)
            formatted_min = extract_min_salary_formatted(salary_raw)

            display_parts = re.split(r'\s{2,}', salary_raw)
            salary_display = display_parts[0] if display_parts else salary_raw

            link_tag = cols[0].find('a')
            link_raw = link_tag['href'] if link_tag else ""

            # Tarihleri ham alıp normalize et
            posted_raw = cols[4].get_text(strip=True)
            closes_raw = cols[5].get_text(strip=True)

            posted_norm = normalize_date(posted_raw)
            closes_norm = normalize_date(closes_raw)

            print(f"Satır {i + 1}: {formatted_min} | "
                  f"Posted: {posted_raw!r} → {posted_norm} | "
                  f"Closes: {closes_raw!r} → {closes_norm}")
            print(f"Satır {i + 1} departman ham: {clean_job_id(link_raw)[:2]}")
            jobs_list.append({
                'Job_ID': clean_job_id(link_raw).upper(),
                'Title': cols[0].get_text(strip=True),
                'Department_ID': clean_job_id(link_raw)[:2].upper(),
                'Department': cols[1].get_text(strip=True),
                'Min_Salary': formatted_min,
                'Posted': posted_norm,  # ← normalize edilmiş
                'Closes': closes_norm,  # ← normalize edilmiş
                'Retrieved_At': datetime.now().strftime('%Y-%m-%d'),
            })
    jobs_list.reverse()
    save_to_csv(jobs_list)


# --- 5. FONKSİYON: AKILLI KAYIT, BİRLEŞTİRME VE MİGRASYON ---
def save_to_csv(new_data, main_file='cambridge_jobs_archive.csv'):
    new_df = pd.DataFrame(new_data)

    all_files = glob.glob('cambridge_jobs_*.csv')
    df_list = []

    for f in all_files:
        if os.path.exists(f):
            temp = pd.read_csv(f)

            # Mevcut dosyadaki tarihleri de normalize et (migrasyon)
            for col in ('Posted', 'Closes'):
                if col in temp.columns:
                    temp[col] = temp[col].apply(normalize_date)

            df_list.append(temp)

    df_list.append(new_df)

    final_df = pd.concat(df_list, ignore_index=True)
    final_df = final_df.drop_duplicates(subset=['Job_ID'], keep='first')

    try:
        os.system("taskkill /f /im excel.exe /t >nul 2>&1")
        final_df.to_csv(main_file, index=False, encoding='utf-8-sig')
        print(f"\nBAŞARILI: '{main_file}' güncellendi. "
              f"Toplam kayıt: {len(final_df)}")

        for f in glob.glob('cambridge_jobs_[0-9]*.csv'):
            os.remove(f)

    except PermissionError:
        temp_name = f"cambridge_jobs_{datetime.now().strftime('%H%M%S')}.csv"
        final_df.to_csv(temp_name, index=False, encoding='utf-8-sig')
        print(f"\nUYARI: Dosya açık olduğu için '{temp_name}' adıyla kaydedildi.")


# --- ÇALIŞTIR ---
if __name__ == "__main__":
    run_scraper()
