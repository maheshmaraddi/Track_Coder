from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime, timedelta

options = Options()
options.add_argument('-private')
options.add_argument('--no-sandbox')

geckodriver_path = './driver/geckodriver'
service = FirefoxService(geckodriver_path)

driver = webdriver.Firefox(service=service, options=options)

try:
    driver.get("https://app.software.com/dashboard/components/active_code_time_graph")

    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, "email")))

    userName_field = driver.find_element(By.ID, "email")
    password_field = driver.find_element(By.ID, "password")
    signIn_btn = driver.find_element(By.CLASS_NAME, "btn-primary")

    userName_field.send_keys("@gmail.com")
    password_field.send_keys("@123")
    signIn_btn.click()

    today = datetime.now()
    yesterday = today - timedelta(days=1)
    yesterday_day = yesterday.strftime('%A')
    day_map = {
        'Monday': 'Mon',
        'Tuesday': 'Tue',
        'Wednesday': 'Wed',
        'Thursday': 'Thu',
        'Friday': 'Fri',
        'Saturday': 'Sat',
        'Sunday': 'Sun'
    }
    yesterday_abbr = day_map[yesterday_day]

    # Active Code Time
    xpath_active = f'//*[contains(@aria-label, "{yesterday_abbr},") and contains(@aria-label, "Active Code Time")]'
    element_active = driver.find_element(By.XPATH, xpath_active)
    aria_label_active = element_active.get_attribute('aria-label')

    # Extract Active Code Time
    site_act = aria_label_active.split(",")[1].strip().split()[0]
    active_code_time = round(float(site_act.rstrip('.')), 2)

    hours_active = int(active_code_time)
    minutes_active = int((active_code_time - hours_active) * 60)
    print(f"Yesterday's ({yesterday_abbr}) Active Code Time: {hours_active:01}:{minutes_active:02}")

    # Code Time
    xpath_code_time = f'//*[contains(@aria-label, "{yesterday_abbr},") and contains(@aria-label, "Code Time")]'
    element_code_time = driver.find_element(By.XPATH, xpath_code_time)
    aria_label_code_time = element_code_time.get_attribute('aria-label')

    # Extract Code Time
    site_ct = aria_label_code_time.split(",")[1].strip().split()[0]
    code_time = round(float(site_ct.rstrip('.')), 2)

    ct_hours = int(code_time)
    minutes_ct = int((code_time - ct_hours) * 60)

    print(f"Yesterday's ({yesterday_abbr}) Code Time: {ct_hours:01}:{minutes_ct:02}")

    # Calculate Total Time
    total_minutes = (hours_active * 60 + minutes_active) + (ct_hours * 60 + minutes_ct)
    total_hours = total_minutes // 60
    remaining_minutes = total_minutes % 60

    print(f"Total Time for Yesterday ({yesterday_abbr}): {total_hours:01}:{remaining_minutes:02}")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    driver.quit()