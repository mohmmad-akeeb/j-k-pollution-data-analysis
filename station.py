import time
import re
import csv
import argparse

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException

from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service


YEARS = ["2022", "2023", "2024", "2025", "2026"]

MONTHS = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]


# -------------------------------
# DRIVER
# -------------------------------
def create_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1080")

    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )


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
                data[(month, year)] = value

        except StaleElementReferenceException:
            continue

    return data


# -------------------------------
# HELPERS
# -------------------------------
def wait_chart(driver, wait, year):
    wait.until(
        lambda d: len(
            d.find_elements(By.CSS_SELECTOR, "#chartdiv12 g[aria-label]")
        ) > 0
    )
    time.sleep(0.5)


def click_show(driver, wait):
    btn = wait.until(EC.element_to_be_clickable((By.ID, "btnshow")))
    driver.execute_script("arguments[0].click();", btn)


def reset_chart(driver):
    driver.execute_script("""
        var chart = document.getElementById('chartdiv12');
        if(chart){ chart.innerHTML = ''; }
    """)
    time.sleep(0.2)


# -------------------------------
# FIND STATION (NO STALE ELEMENTS)
# -------------------------------
def find_station(driver, wait, target_station):
    target_station = target_station.strip().lower()

    wait.until(EC.presence_of_element_located((By.ID, "ddregion")))

    regions = [
        opt.get_attribute("value")
        for opt in Select(driver.find_element(By.ID, "ddregion")).options
        if opt.get_attribute("value") != "Select"
    ]

    for region_val in regions:
        Select(driver.find_element(By.ID, "ddregion")).select_by_value(region_val)

        wait.until(EC.presence_of_element_located((By.ID, "dddistrict")))
        time.sleep(1)

        districts = [
            opt.get_attribute("value")
            for opt in Select(driver.find_element(By.ID, "dddistrict")).options
            if opt.get_attribute("value") != "Select"
        ]

        for district_val in districts:
            Select(driver.find_element(By.ID, "dddistrict")).select_by_value(district_val)

            wait.until(EC.presence_of_element_located((By.ID, "ddstation")))
            time.sleep(1)

            station_count = len(Select(driver.find_element(By.ID, "ddstation")).options)

            for i in range(station_count):
                # re-fetch every iteration
                stations = Select(driver.find_element(By.ID, "ddstation")).options
                s = stations[i]

                name = s.text.strip().lower()

                if target_station in name:
                    return (
                        region_val,
                        district_val,
                        s.get_attribute("value")
                    )

    return None, None, None


# -------------------------------
# SCRAPE
# -------------------------------
def scrape_station(station_name):
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

    driver.get(url)

    region, district, station = find_station(driver, wait, station_name)

    if not station:
        print("Station not found")
        driver.quit()
        return

    print(f"Resolved → Region:{region}, District:{district}, Station:{station}")

    output_file = f"{station_name.replace(' ', '_')}.csv"

    rows = []

    for year in YEARS:

        Select(driver.find_element(By.ID, "ddregion")).select_by_value(region)
        time.sleep(1)

        Select(driver.find_element(By.ID, "dddistrict")).select_by_value(district)
        time.sleep(1)

        Select(driver.find_element(By.ID, "ddstation")).select_by_value(station)
        time.sleep(1)

        Select(driver.find_element(By.ID, "ddyr1")).select_by_value(year)
        time.sleep(1)

        month_data = {
            m: {"AQI":"null","PM10":"null","PM2_5":"null","NO2":"null","SO2":"null"}
            for m in MONTHS
        }

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

        for month in MONTHS:
            if year == "2026" and month not in ["January", "February", "March"]:
                continue

            rows.append([
                region,
                district,
                station_name,
                year,
                month,
                month_data[month]["AQI"],
                month_data[month]["PM10"],
                month_data[month]["PM2_5"],
                month_data[month]["NO2"],
                month_data[month]["SO2"]
            ])

    driver.quit()

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Region","District","Station","Year","Month",
            "AQI","PM10","PM2_5","NO2","SO2"
        ])

        writer.writerows(rows)

    print(f"Saved → {output_file}")


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("station", help="Station name (partial match supported)")

    args = parser.parse_args()

    scrape_station(args.station)