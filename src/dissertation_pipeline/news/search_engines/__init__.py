"""Search-engine based news link discovery."""

from dissertation_pipeline.news.search_engines.gdelt import (
    DEFAULT_GDELT_PROFILES,
    GdeltDocClient,
    GdeltWindowCollector,
)
from dissertation_pipeline.news.search_engines.yandex import (
    DEFAULT_QUERY_PROFILES,
    YandexSearchClient,
    YandexWindowCollector,
)

__all__ = [
    "DEFAULT_GDELT_PROFILES",
    "DEFAULT_QUERY_PROFILES",
    "GdeltDocClient",
    "GdeltWindowCollector",
    "YandexSearchClient",
    "YandexWindowCollector",
]
