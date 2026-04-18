import time
import re
import csv
import os
import json
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException

from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service


OUTPUT_FILE = "jkpcb_parallel.csv"
LOG_FILE = "jkpcb_parallel.log"

YEARS = ["2022", "2023", "2024", "2025", "2026"]


# -------------------------------
# DRIVER FACTORY
# -------------------------------
def create_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    return driver


# -------------------------------
# LOG
# -------------------------------
def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"{time.ctime()} | {msg}\n")


# -------------------------------
# EXTRACT
# -------------------------------
pattern = re.compile(r"([A-Za-z]+-\d{4})\s+(\d+)")

def extract_chart(driver):
    data = {}

    elements = driver.find_elements(By.CSS_SELECTOR, "#chartdiv12 g[aria-label]")

    for el in elements:
        try:
            if not el.is_displayed():
                continue

            text = el.get_attribute("aria-label")
            match = pattern.search(text)

            if match:
                month_year = match.group(1)
                value = int(match.group(2))

                month, year = month_year.split("-")
                key = (month, year)

                data[key] = value

        except StaleElementReferenceException:
            continue

    return data


# -------------------------------
# WAIT
# -------------------------------
def wait_chart(driver, wait, year):
    wait.until(
        lambda d: any(
            f"December-{year}" in (el.get_attribute("aria-label") or "")
            for el in d.find_elements(By.CSS_SELECTOR, "#chartdiv12 g[aria-label]")
        )
    )
    time.sleep(0.5)


# -------------------------------
# CLICK
# -------------------------------
def click_show(driver, wait):
    btn = wait.until(EC.element_to_be_clickable((By.ID, "btnshow")))
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
    time.sleep(0.2)
    driver.execute_script("arguments[0].click();", btn)


# -------------------------------
# RESET
# -------------------------------
def reset_chart(driver):
    driver.execute_script("""
        var chart = document.getElementById('chartdiv12');
        if(chart){ chart.innerHTML = ''; }
    """)
    time.sleep(0.2)


# -------------------------------
# WORKER TASK
# -------------------------------
def scrape_task(task):
    region, district, station, year = task

    driver = create_driver()
    wait = WebDriverWait(driver, 25)

    url = "https://jkpcb.jk.gov.in/airquality.aspx"

    parameters = {
        "aqi": "AQI",
        "pm10": "PM10",
        "pm25": "PM2_5",
        "no2": "NO2",
        "so2": "SO2"
    }

    months = [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
    ]

    rows = []

    try:
        driver.get(url)

        # REGION
        Select(wait.until(
            EC.presence_of_element_located((By.ID, "ddregion"))
        )).select_by_value(region)
        wait.until(EC.presence_of_element_located((By.ID, "dddistrict")))
        time.sleep(1)

        # DISTRICT
        Select(driver.find_element(By.ID, "dddistrict")).select_by_value(district)
        wait.until(EC.presence_of_element_located((By.ID, "ddstation")))
        time.sleep(1)

        # STATION
        Select(driver.find_element(By.ID, "ddstation")).select_by_value(station)
        time.sleep(1)

        # YEAR
        Select(driver.find_element(By.ID, "ddyr1")).select_by_value(year)
        time.sleep(1)

        month_data = {
            m: {
                "AQI": "null",
                "PM10": "null",
                "PM2_5": "null",
                "NO2": "null",
                "SO2": "null"
            } for m in months
        }

        # PARAM LOOP
        for param, col in parameters.items():
            reset_chart(driver)

            Select(driver.find_element(By.ID, "ddparam")).select_by_value(param)
            time.sleep(0.4)

            click_show(driver, wait)
            wait_chart(driver, wait, year)

            data = extract_chart(driver)

            for (month, yr), value in data.items():
                if yr == year:
                    if value == 0:
                        value = "null"
                    month_data[month][col] = value

        for month in months:
            rows.append([
                region, district, station, year, month,
                month_data[month]["AQI"],
                month_data[month]["PM10"],
                month_data[month]["PM2_5"],
                month_data[month]["NO2"],
                month_data[month]["SO2"]
            ])

    except Exception as e:
        log(f"FAIL {region}-{district}-{station}-{year} | {e}")

    finally:
        driver.quit()

    return rows


# -------------------------------
# BUILD TASK LIST
# -------------------------------
def build_tasks():
    driver = create_driver()
    wait = WebDriverWait(driver, 20)

    driver.get("https://jkpcb.jk.gov.in/airquality.aspx")

    tasks = []

    region_select = Select(wait.until(
        EC.presence_of_element_located((By.ID, "ddregion"))
    ))

    regions = [
        opt.get_attribute("value")
        for opt in region_select.options if opt.get_attribute("value") != "Select"
    ]

    for region in regions:
        Select(driver.find_element(By.ID, "ddregion")).select_by_value(region)
        wait.until(EC.presence_of_element_located((By.ID, "dddistrict")))
        time.sleep(1)

        districts = [
            opt.get_attribute("value")
            for opt in Select(driver.find_element(By.ID, "dddistrict")).options
            if opt.get_attribute("value") != "Select"
        ]

        for district in districts:
            Select(driver.find_element(By.ID, "dddistrict")).select_by_value(district)
            wait.until(EC.presence_of_element_located((By.ID, "ddstation")))
            time.sleep(1)

            stations = [
                opt.get_attribute("value")
                for opt in Select(driver.find_element(By.ID, "ddstation")).options
                if opt.get_attribute("value") != "Select"
            ]

            for station in stations:
                for year in YEARS:
                    tasks.append((region, district, station, year))

    driver.quit()
    return tasks


# -------------------------------
# SAVE CSV
# -------------------------------
def save_rows(rows):
    file_exists = os.path.exists(OUTPUT_FILE)

    with open(OUTPUT_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "Region","District","Station","Year","Month",
                "AQI","PM10","PM2_5","NO2","SO2"
            ])

        for r in rows:
            writer.writerow(r)


# -------------------------------
# MAIN
# -------------------------------
def main():
    tasks = build_tasks()

    workers = max(2, cpu_count() // 2)

    print(f"🚀 Running with {workers} workers")

    with Pool(workers) as pool:
        for result in tqdm(pool.imap_unordered(scrape_task, tasks), total=len(tasks)):
            if result:
                save_rows(result)


if __name__ == "__main__":
    main()