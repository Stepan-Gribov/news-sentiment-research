from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path


FAST_BLOCKED_URL_PATTERNS = [
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.webp",
    "*.svg",
    "*.ico",
    "*.css",
    "*.woff",
    "*.woff2",
    "*.ttf",
    "*.otf",
    "*.mp4",
    "*.webm",
    "*.mp3",
    "*.avi",
    "*.mov",
    "*google-analytics.com*",
    "*googletagmanager.com*",
    "*mc.yandex.ru*",
    "*metrika.yandex.ru*",
    "*adfox.ru*",
    "*doubleclick.net*",
]


def find_chrome_executable() -> Path:
    env_value = os.environ.get("CHROME_EXECUTABLE")
    if env_value:
        path = Path(env_value).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"CHROME_EXECUTABLE is set but file does not exist: {path}")

    system = platform.system().lower()
    candidates: list[Path] = []
    if system == "darwin":
        candidates += [
            Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            Path.home() / "Applications" / "Google Chrome.app" / "Contents" / "MacOS" / "Google Chrome",
        ]
    elif system == "windows":
        for root_var in ["PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"]:
            root = os.environ.get(root_var)
            if root:
                candidates.append(Path(root) / "Google" / "Chrome" / "Application" / "chrome.exe")
        candidates.append(Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"))
        candidates.append(Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"))
    else:
        candidates += [Path("/usr/bin/google-chrome"), Path("/usr/bin/chromium"), Path("/usr/bin/chromium-browser")]

    for command in ["google-chrome", "chrome", "chromium", "chromium-browser"]:
        found = shutil.which(command)
        if found:
            candidates.append(Path(found))

    for path in candidates:
        if path.exists():
            return path

    checked = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Google Chrome executable was not found. Set CHROME_EXECUTABLE to the full chrome path. "
        f"Checked:\n{checked}"
    )


def fast_chrome_args() -> list[str]:
    return [
        "--blink-settings=imagesEnabled=false",
        "--disable-background-networking",
        "--disable-background-timer-throttling",
        "--disable-client-side-phishing-detection",
        "--disable-default-apps",
        "--disable-dev-shm-usage",
        "--disable-extensions",
        "--disable-features=Translate,OptimizationHints,MediaRouter,AutofillServerCommunication",
        "--disable-popup-blocking",
        "--disable-renderer-backgrounding",
        "--disable-sync",
        "--metrics-recording-only",
        "--mute-audio",
        "--no-default-browser-check",
        "--no-first-run",
        "--password-store=basic",
    ]


def fast_chrome_prefs(extra: dict | None = None) -> dict:
    prefs = {
        "profile.default_content_setting_values.images": 2,
        "profile.managed_default_content_settings.images": 2,
        "profile.default_content_setting_values.notifications": 2,
        "profile.default_content_setting_values.popups": 2,
        "profile.default_content_setting_values.geolocation": 2,
        "credentials_enable_service": False,
        "profile.password_manager_enabled": False,
    }
    if extra:
        prefs.update(extra)
    return prefs


def apply_fast_chrome_options(options, *, page_load_strategy: str = "eager", prefs: dict | None = None):
    options.page_load_strategy = page_load_strategy
    for arg in fast_chrome_args():
        options.add_argument(arg)
    options.add_experimental_option("prefs", fast_chrome_prefs(prefs))
    return options


def configure_fast_driver(driver) -> None:
    """Block heavy resources after WebDriver attaches, including remote Chrome sessions."""
    try:
        driver.execute_cdp_cmd("Network.enable", {})
        driver.execute_cdp_cmd("Network.setBlockedURLs", {"urls": FAST_BLOCKED_URL_PATTERNS})
    except Exception:
        pass
