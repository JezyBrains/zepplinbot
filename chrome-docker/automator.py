import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--load-extension=/opt/extension")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    # Add other necessary options for bot detection bypass if needed
    
    driver = webdriver.Remote(
        command_executor='http://localhost:4444/wd/hub',
        options=chrome_options
    )
    return driver

def login_and_launch():
    username = os.getenv('PLAYER_USERNAME')
    password = os.getenv('PLAYER_PASSWORD')
    dashboard_url = os.getenv('DASHBOARD_URL', 'http://dashboard:8050')
    
    driver = setup_driver()
    try:
        # 1. Configure Extension for Docker Network
        print(f"📡 Configuring extension to use dashboard: {dashboard_url}")
        driver.get("https://www.google.com") # Dummy page to run script
        driver.execute_script(f"""
            chrome.storage.local.set({{'dashboardUrl': '{dashboard_url}'}}, function() {{
                console.log('✅ Extension configured for Docker network');
            }});
        """)
        
        print(f"🚀 Navigating to Sokabet login...")
        driver.get("https://sokabet.co.tz/login")

        wait = WebDriverWait(driver, 30)
        
        try:
            # Sokabet mobile login typically uses a phone/mobile input
            print("🔑 Attempting login...")
            user_field = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='text'], input[placeholder*='Mobile']")))
            pass_field = driver.find_element(By.CSS_SELECTOR, "input[type='password']")
            
            user_field.send_keys(username)
            pass_field.send_keys(password)
            
            # Find and click login button
            login_btn = driver.find_element(By.CSS_SELECTOR, "button.login-btn, button[type='submit']")
            login_btn.click()
            print("✅ Login submitted.")
            
            # Wait for login success (redirect or profile element)
            time.sleep(5)
        except Exception as e:
            print(f"⚠️ Automatic login sequence failed: {e}")
            print("👉 Please finish login manually via VNC if needed.")

        # 2. Navigate to Zeppelin
        print("🎮 Searching for Zeppelin game...")
        try:
            # User specified: <span class="menu_item_title">Zeppelin</span>
            zeppelin_link = wait.until(EC.element_to_be_clickable((By.XPATH, "//span[contains(@class, 'menu_item_title') and text()='Zeppelin']")))
            driver.execute_script("arguments[0].click();", zeppelin_link)
            print("🚀 Zeppelin launched.")
        except Exception as e:
            print(f"⚠️ Could not find Zeppelin link automatically: {e}")
            print("👉 Please navigate to the game manually via VNC.")

        # Keep the browser open
        while True:
            time.sleep(60)

    except Exception as e:
        print(f"🔴 Error in automation: {e}")
    finally:
        # driver.quit() # Keep it open for the extension to work
        pass

if __name__ == "__main__":
    login_and_launch()
