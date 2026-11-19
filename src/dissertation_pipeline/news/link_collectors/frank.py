import os
import pandas as pd
import time
import random
import re

import pandas as pd
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException

from concurrent.futures import ThreadPoolExecutor


# ---------------- Selenium: fetch one page ----------------
def fetch_page_source_selenium(url: str, wait_sec: float = 0.5, page_timeout: int = 10) -> str:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--start-minimized")
    options.add_argument("--window-size=1240,720")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument('--blink-settings=imagesEnabled=false')
    options.add_argument("--no-first-run")
    options.add_argument("--no-default-browser-check")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )

    driver = webdriver.Chrome(options=options)
    # driver.minimize_window()

    driver.set_page_load_timeout(page_timeout)
    driver.set_script_timeout(page_timeout)

    try:
        driver.get(url)
        time.sleep(wait_sec)
        html = driver.page_source or ""

        h = html.lower()
        if (
            "servicepipe.ru" in h
            or "id_spinner" in h
            or "id_captcha_frame_div" in h
            or "checkjs" in h
        ):
            raise RuntimeError("Blocked: challenge page returned")

        return html

    finally:
        try:
            driver.quit()
        except Exception:
            pass


# ---------------- Parse max page ----------------
def frankmedia_max_page(html: str) -> int:
    soup = BeautifulSoup(html, "html.parser")
    nums = set()

    for a in soup.select("nav.pagination a.page"):
        t = a.get_text(strip=True)
        if t.isdigit():
            nums.add(int(t))

    for a in soup.select("nav.pagination a[href]"):
        href = a.get("href", "")
        m = re.search(r"(?:\?|&)(?:p)=(\d+)", href)
        if m:
            nums.add(int(m.group(1)))

    return max(nums) if nums else 1


# ---------------- Retry wrapper ----------------
def fetch_with_retries(url: str, max_tries: int = 10, base_sleep: float = 1.0) -> str:
    last_err = None

    for attempt in range(1, max_tries + 1):
        try:
            return fetch_page_source_selenium(url)

        except (RuntimeError, TimeoutException, WebDriverException) as e:
            msg = str(e).lower()

            retryable = (
                ("blocked" in msg)
                or ("challenge" in msg)
                or ("timeout" in msg)
                or ("disconnected" in msg)
                or ("unknown error" in msg)
            )

            if retryable:
                last_err = e
                sleep_s = min(10.0, base_sleep * attempt + random.uniform(0.2, 1.0))
                print(f"Retry {attempt}/{max_tries} -> sleeping {sleep_s:.1f}s : {url}")
                time.sleep(sleep_s)
                continue

            raise

    raise last_err


# ---------------- Main processing ----------------
input_dir = r'./local/My Drive\4. Disserartion research\calculations\data\news_links\frank_links'
output_dir = r'./local/My Drive\4. Disserartion research\calculations\data\news_links\frank_links_content'
os.makedirs(output_dir, exist_ok=True)

start = 0
end = 40


def worker(link: str) -> str:
    return fetch_with_retries(link)


def main():
    files = sorted(os.listdir(input_dir))
    print(f"Total files: {len(files)}. Starting from index: {start}.")

    for num, filename in enumerate(files[start:end], start=start):
        if not filename.endswith(".csv"):
            print(f"Completed {num + 1}/{len(files)} files.")
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            print(f"\nProcessing: {filename}")
            df = pd.read_csv(input_path, encoding="utf-16", sep="\t")
            links = df["link"].astype(str).tolist()

            max_workers = 8
            total_links = len(links)
            counter = {"done": 0}

            def worker_with_status(link: str) -> str:
                res = worker(link)
                counter["done"] += 1
                print(f"{filename} | {counter['done']}/{total_links}", end="\r", flush=True)
                return res

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(worker_with_status, links))

            print()  # newline after row counter finishes

            df["content"] = results
            df.to_csv(output_path, encoding="utf-16", sep="\t", index=False)

        except Exception as e:
            print(f"\nFAILED {filename}: {e}")

        print(f"Completed {num + 1}/{len(files)} files.")


# if __name__ == "__main__":
#     main()



def extract_article_text_and_category(page_source):
    # handle NaN / None / non-string
    if page_source is None or (isinstance(page_source, float) and pd.isna(page_source)):
        return "", ""
    if not isinstance(page_source, str):
        page_source = str(page_source)

    soup = BeautifulSoup(page_source, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()

    category_tag = soup.find("a", class_=re.compile(r"\bcategory\b"))
    category = category_tag.get_text(strip=True) if category_tag else ""

    article = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", class_=re.compile(r"(article|content|post|entry|body)", re.I))
    )
    if not article:
        article = soup

    paragraphs = article.find_all("p")

    text_chunks = []
    for p in paragraphs:
        txt = p.get_text(strip=True)
        if txt:
            text_chunks.append(txt)

    content = " ".join(text_chunks)
    content = re.sub(r"\s+", " ", content).strip()

    return content, category


def work(name):
    print(f"in Process {name}",flush=True)
    dff = pd.read_csv(output_dir+f'\{name}', encoding="utf-16", sep="\t")
    dff['content_txt'] =  None
    dff['content_txt'], dff['category'] = zip(*dff['content'].apply(extract_article_text_and_category))
    dff.to_csv(output_dir+f'\{name}', encoding="utf-16", sep="\t", index=False)



def main1():
    names = sorted(os.listdir(output_dir))
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(work, names))

if __name__ == "__main__":
    main1()

quit()


# import pandas as pd
# df = pd.read_csv(r"./local/My Drive\4. Disserartion research\calculations\data\news_links\frank_links_content\АВТОВАЗ.csv", encoding="utf-16", sep="\t")
# df.head()