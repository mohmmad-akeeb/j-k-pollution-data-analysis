import time
import re
import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException

from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service


TEST_YEAR = "2025"


class JKPCBTestScraper:

    def __init__(self):
        options = webdriver.ChromeOptions()
        options.add_argument("--start-maximized")

        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )

        self.wait = WebDriverWait(self.driver, 25)
        self.url = "https://jkpcb.jk.gov.in/airquality.aspx"

        self.pattern = re.compile(r"([A-Za-z]+-\d{4})\s+(\d+)")

        self.parameters = {
            "aqi": "AQI",
            "pm10": "PM10",
            "pm25": "PM2_5",
            "no2": "NO2",
            "so2": "SO2"
        }

        self.months = [
            "January","February","March","April","May","June",
            "July","August","September","October","November","December"
        ]

    # -------------------------------
    # STRONG WAIT FOR FINAL RENDER
    # -------------------------------
    def wait_chart_loaded(self):
        self.wait.until(
            EC.presence_of_element_located((By.ID, "chartdiv12"))
        )

        # wait until December specifically appears (critical fix)
        self.wait.until(
            lambda d: any(
                "December-" + TEST_YEAR in (el.get_attribute("aria-label") or "")
                for el in d.find_elements(By.CSS_SELECTOR, "#chartdiv12 g[aria-label]")
            )
        )

        time.sleep(1)

    # -------------------------------
    # STRICT EXTRACTION (VISIBLE + LATEST)
    # -------------------------------
    def extract_chart(self):
        data = {}

        elements = self.driver.find_elements(By.CSS_SELECTOR, "#chartdiv12 g[aria-label]")

        for el in elements:
            try:
                if not el.is_displayed():
                    continue

                text = el.get_attribute("aria-label")
                match = self.pattern.search(text)

                if match:
                    month_year = match.group(1).strip()
                    value = int(match.group(2))

                    month, year = month_year.split("-")
                    key = (month, year)

                    # always overwrite to keep latest render
                    data[key] = value

            except StaleElementReferenceException:
                continue

        return data

    # -------------------------------
    # DROPDOWN SELECTION
    # -------------------------------
    def auto_select(self):

        region_select = Select(self.wait.until(
            EC.presence_of_element_located((By.ID, "ddregion"))
        ))

        region_val = [
            opt.get_attribute("value")
            for opt in region_select.options
            if opt.get_attribute("value") != "Select"
        ][0]

        region_select.select_by_value(region_val)
        self.wait.until(EC.presence_of_element_located((By.ID, "dddistrict")))
        time.sleep(1)

        region_text = Select(self.driver.find_element(By.ID, "ddregion")).first_selected_option.text

        district_select = Select(self.wait.until(
            EC.presence_of_element_located((By.ID, "dddistrict"))
        ))

        district_val = [
            opt.get_attribute("value")
            for opt in district_select.options
            if opt.get_attribute("value") != "Select"
        ][0]

        district_select.select_by_value(district_val)
        self.wait.until(EC.presence_of_element_located((By.ID, "ddstation")))
        time.sleep(1)

        district_text = Select(self.driver.find_element(By.ID, "dddistrict")).first_selected_option.text

        station_select = Select(self.wait.until(
            EC.presence_of_element_located((By.ID, "ddstation"))
        ))

        station_val = [
            opt.get_attribute("value")
            for opt in station_select.options
            if opt.get_attribute("value") != "Select"
        ][0]

        station_select.select_by_value(station_val)
        time.sleep(1)

        station_text = Select(self.driver.find_element(By.ID, "ddstation")).first_selected_option.text

        Select(self.driver.find_element(By.ID, "ddyr1")).select_by_value(TEST_YEAR)
        time.sleep(1)

        print("\n🔍 TESTING COMBINATION:")
        print(f"Region  : {region_text}")
        print(f"District: {district_text}")
        print(f"Station : {station_text}")
        print(f"Year    : {TEST_YEAR}")
        print("-" * 50)

        return region_text, district_text, station_text

    # -------------------------------
    # HARD RESET
    # -------------------------------
    def reset_chart(self):
        self.driver.execute_script("""
            var chart = document.getElementById('chartdiv12');
            if(chart){ chart.innerHTML = ''; }
        """)
        time.sleep(0.5)

    def click_show(self):
        btn = self.wait.until(EC.element_to_be_clickable((By.ID, "btnshow")))
        self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
        time.sleep(0.3)
        self.driver.execute_script("arguments[0].click();", btn)

    # -------------------------------
    # MAIN RUN
    # -------------------------------
    def run(self):
        self.driver.get(self.url)

        region, district, station = self.auto_select()

        month_data = {
            m: {
                "AQI": "null",
                "PM10": "null",
                "PM2_5": "null",
                "NO2": "null",
                "SO2": "null"
            } for m in self.months
        }

        for param, col in self.parameters.items():
            print(f"\n➡️ Fetching {col}")

            self.reset_chart()

            Select(self.driver.find_element(By.ID, "ddparam")).select_by_value(param)

            self.wait.until(
                lambda d: Select(d.find_element(By.ID, "ddparam"))
                .first_selected_option.get_attribute("value") == param
            )
            time.sleep(0.5)

            self.click_show()

            self.wait_chart_loaded()

            data = self.extract_chart()

            for (month, year), value in data.items():
                if year == TEST_YEAR:
                    if value == 0:
                        value = "null"
                    month_data[month][col] = value

        print("\n📊 FINAL VERIFIED OUTPUT:\n")

        rows = []

        for month in self.months:
            row = month_data[month]

            print(
                f"{month} {TEST_YEAR} → "
                f"AQI={row['AQI']} | "
                f"PM10={row['PM10']} | "
                f"PM2.5={row['PM2_5']} | "
                f"NO2={row['NO2']} | "
                f"SO2={row['SO2']}"
            )

            rows.append({
                "Region": region,
                "District": district,
                "Station": station,
                "Year": TEST_YEAR,
                "Month": month,
                "AQI": row["AQI"],
                "PM10": row["PM10"],
                "PM2_5": row["PM2_5"],
                "NO2": row["NO2"],
                "SO2": row["SO2"]
            })

        df = pd.DataFrame(rows)
        filename = f"jkpcb_test_{TEST_YEAR}.csv"
        df.to_csv(filename, index=False)

        print(f"\n💾 Saved to: {filename}")

        self.driver.quit()


if __name__ == "__main__":
    JKPCBTestScraper().run()