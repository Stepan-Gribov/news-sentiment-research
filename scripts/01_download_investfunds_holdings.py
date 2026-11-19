from __future__ import annotations

import argparse
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from dissertation_pipeline.paths import PATHS
from dissertation_pipeline.selenium_fast import apply_fast_chrome_options, configure_fast_driver


def build_driver(chromedriver: Path | None, user_data_dir: Path):
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service

    user_data_dir.mkdir(parents=True, exist_ok=True)
    options = Options()
    apply_fast_chrome_options(options)
    options.add_argument(f"--user-data-dir={user_data_dir}")
    if chromedriver:
        driver = webdriver.Chrome(service=Service(str(chromedriver)), options=options)
    else:
        driver = webdriver.Chrome(options=options)
    configure_fast_driver(driver)
    return driver


def main() -> None:
    parser = argparse.ArgumentParser(description="Download mutual fund holdings exports from Investfunds Pro.")
    parser.add_argument("--chromedriver", type=Path, default=None)
    parser.add_argument("--user-data-dir", type=Path, default=PATHS.tmp_dir / "investfunds_chrome_profile")
    parser.add_argument("--login-wait-seconds", type=int, default=60)
    parser.add_argument("--out-note", type=Path, default=PATHS.logs_dir / "investfunds_download_note.txt")
    parser.add_argument("--sample-dates", type=int, default=None)
    parser.add_argument("--sample-assets", type=int, default=None)
    args = parser.parse_args()

    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import Select

    driver = build_driver(args.chromedriver, args.user_data_dir)
    try:
        driver.get("https://pro.investfunds.ru/")
        print(f"Log in if needed. Waiting {args.login_wait_seconds}s...")
        time.sleep(args.login_wait_seconds)

        date_options = [o.text for o in Select(driver.find_element(By.ID, "export_date_combo")).options]
        asset_options = [o.text for o in Select(driver.find_element(By.ID, "prop_type_combo")).options]
        if args.sample_dates:
            date_options = date_options[: args.sample_dates]
        if args.sample_assets:
            asset_options = asset_options[: args.sample_assets]

        for asset in asset_options:
            for date in date_options:
                Select(driver.find_element(By.ID, "export_date_combo")).select_by_visible_text(date)
                time.sleep(1.5)
                try:
                    Select(driver.find_element(By.ID, "prop_type_combo")).select_by_visible_text(asset)
                    time.sleep(1.5)
                except Exception:
                    pass
                driver.find_element(By.ID, "exportAssetsFilesBtn").click()
                time.sleep(3)

        args.out_note.parent.mkdir(parents=True, exist_ok=True)
        args.out_note.write_text(
            f"Downloaded Investfunds exports. dates={len(date_options)}, assets={len(asset_options)}\n",
            encoding="utf-8",
        )
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
