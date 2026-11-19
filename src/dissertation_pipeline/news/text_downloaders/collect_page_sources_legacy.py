# %%
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# %% [markdown]
# Set your input file path.

# %%
FILE_PATH = (
    "./local/Library/CloudStorage/analyst@example.org/"
    "My Drive/4. Disserartion research/calculations/data/news_links/kommersant/АВТОВАЗ.xlsx"
)
URL_COL = "url"
OUTPUT_SUFFIX = "_with_source"
SLEEP_SECONDS = 0.2
OUTPUT_DIR = (
    "./local/Library/CloudStorage/analyst@example.org/"
    "My Drive/4. Disserartion research/calculations/data/pages_source"
)

# %%
def build_session(total_retries: int = 5, backoff_factor: float = 0.5) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0 Safari/537.36"
            )
        }
    )
    return session


def fetch_page_source(session: requests.Session, url: str, timeout: int = 20) -> str | None:
    try:
        resp = session.get(url, timeout=timeout)
    except requests.RequestException:
        return None

    if resp.status_code != 200:
        return None
    return resp.text

# %%
file_path = Path(FILE_PATH)
df = pd.read_excel(file_path)

if URL_COL not in df.columns:
    raise ValueError(f"Missing column '{URL_COL}' in {file_path.name}")

session = build_session()
page_sources: list[str | None] = []

for url in tqdm(df[URL_COL].astype(str).tolist(), desc="Fetching pages"):
    url = url.strip()
    if not url or url.lower() == "nan":
        page_sources.append(None)
        continue
    page_sources.append(fetch_page_source(session, url))
    if SLEEP_SECONDS:
        time.sleep(SLEEP_SECONDS)

df["page_source"] = page_sources

output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"{file_path.stem}{OUTPUT_SUFFIX}.csv"
df.to_csv(output_path, index=False, encoding="utf-16")
output_path
