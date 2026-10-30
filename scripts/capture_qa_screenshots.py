import os
import time
from playwright.sync_api import sync_playwright

os.makedirs("reports/screenshots", exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch()
    
    # Function to wait for Streamlit app to be fully idle
    def wait_for_idle(page):
        time.sleep(1.0)
        # Wait until top right running indicator disappears
        page.locator('[data-testid="stStatusWidget"]').wait_for(state="hidden", timeout=30000)

    # 1. Desktop - Tab 1 (Initial Empty State)
    page1 = browser.new_page(viewport={"width": 1440, "height": 1800})
    page1.goto("http://localhost:8501", wait_until="networkidle")
    time.sleep(2)
    page1.screenshot(path="reports/screenshots/01_tab1_initial_empty.png")
    print("Captured: 01_tab1_initial_empty.png")

    # 2. Desktop - Tab 1 (Executed Results)
    page1.locator('button:has-text("Run Simulation")').first.click()
    wait_for_idle(page1)
    time.sleep(1)
    page1.screenshot(path="reports/screenshots/02_tab1_executed_results.png")
    print("Captured: 02_tab1_executed_results.png")
    page1.close()

    # 3. Tab 2: Algorithm Arena
    page2 = browser.new_page(viewport={"width": 1440, "height": 1800})
    page2.goto("http://localhost:8501", wait_until="networkidle")
    time.sleep(2)
    page2.get_by_role("tab", name="Algorithm Arena").click()
    time.sleep(1)
    page2.locator('button:has-text("Run Benchmark Arena")').first.click()
    wait_for_idle(page2)
    time.sleep(1)
    page2.screenshot(path="reports/screenshots/03_tab2_arena.png")
    print("Captured: 03_tab2_arena.png")
    page2.close()

    # 4. Tab 3: Contextual Personalization
    page3 = browser.new_page(viewport={"width": 1440, "height": 1800})
    page3.goto("http://localhost:8501", wait_until="networkidle")
    time.sleep(2)
    page3.get_by_role("tab", name="Contextual Personalization").click()
    time.sleep(1)
    page3.locator('button:has-text("Run LinUCB Personalization")').first.click()
    wait_for_idle(page3)
    time.sleep(1)
    page3.screenshot(path="reports/screenshots/04_tab3_contextual.png")
    print("Captured: 04_tab3_contextual.png")
    page3.close()

    # 5. Tab 4: Non-Stationary Dynamics
    page4 = browser.new_page(viewport={"width": 1440, "height": 1800})
    page4.goto("http://localhost:8501", wait_until="networkidle")
    time.sleep(2)
    page4.get_by_role("tab", name="Non-Stationary Dynamics").click()
    time.sleep(1)
    page4.locator('button:has-text("Run Non-Stationary Simulation")').first.click()
    wait_for_idle(page4)
    time.sleep(1)
    page4.screenshot(path="reports/screenshots/05_tab4_non_stationary.png")
    print("Captured: 05_tab4_non_stationary.png")
    page4.close()

    # 6. Tab 5: Offline Evaluation
    page5 = browser.new_page(viewport={"width": 1440, "height": 1800})
    page5.goto("http://localhost:8501", wait_until="networkidle")
    time.sleep(2)
    page5.get_by_role("tab", name="Offline Evaluation").click()
    time.sleep(1)
    page5.locator('button:has-text("Run Offline Replay Evaluation")').first.click()
    wait_for_idle(page5)
    time.sleep(1)
    page5.screenshot(path="reports/screenshots/06_tab5_offline.png")
    print("Captured: 06_tab5_offline.png")
    page5.close()

    # 7. Tab 6: Methodology & Viva Guide
    page6 = browser.new_page(viewport={"width": 1440, "height": 1800})
    page6.goto("http://localhost:8501", wait_until="networkidle")
    time.sleep(2)
    page6.get_by_role("tab", name="Methodology & Viva Guide").click()
    time.sleep(1)
    page6.screenshot(path="reports/screenshots/07_tab6_methodology.png")
    print("Captured: 07_tab6_methodology.png")
    page6.close()

    # 8. Responsive Mobile Viewport (375x1200)
    page_mob = browser.new_page(viewport={"width": 375, "height": 1200})
    page_mob.goto("http://localhost:8501", wait_until="networkidle")
    time.sleep(2)
    page_mob.screenshot(path="reports/screenshots/08_mobile_responsive.png")
    print("Captured: 08_mobile_responsive.png")
    page_mob.close()

    browser.close()

print("All screenshots successfully captured in reports/screenshots/")
