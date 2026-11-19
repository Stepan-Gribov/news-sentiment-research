from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_supp_root() -> Path:
    env_value = os.environ.get("DISS_SUPP_ROOT")
    if env_value:
        return Path(env_value).expanduser()

    return DEFAULT_PROJECT_ROOT


DEFAULT_SUPP_ROOT = _default_supp_root()


@dataclass(frozen=True)
class PipelinePaths:
    project_root: Path = DEFAULT_PROJECT_ROOT
    supp_root: Path = DEFAULT_SUPP_ROOT

    @property
    def config_dir(self) -> Path:
        return self.project_root / "config"

    @property
    def company_names_file(self) -> Path:
        top500_file = self.config_dir / "company_lists" / "top500_securities_news_search_names.txt"
        return top500_file if top500_file.exists() else self.config_dir / "company_names.txt"

    @property
    def notebooks_dir(self) -> Path:
        return self.project_root / "notebooks"

    @property
    def docs_dir(self) -> Path:
        return self.project_root / "docs"

    @property
    def raw_dir(self) -> Path:
        return self.supp_root / "_legacy_raw"

    @property
    def news_data_dir(self) -> Path:
        return self.supp_root / "news_data"

    @property
    def news_links_history_dir(self) -> Path:
        return self.news_data_dir / "links" / "history"

    @property
    def news_links_runs_dir(self) -> Path:
        return self.news_data_dir / "links" / "runs"

    @property
    def news_pages_history_dir(self) -> Path:
        return self.news_data_dir / "article_pages" / "history"

    @property
    def news_pages_runs_dir(self) -> Path:
        return self.news_data_dir / "article_pages" / "runs"

    @property
    def news_merged_dir(self) -> Path:
        return self.news_data_dir / "merged"

    @property
    def interim_dir(self) -> Path:
        return self.supp_root / "_interim"

    @property
    def processed_dir(self) -> Path:
        return self.supp_root / "_processed"

    @property
    def outputs_dir(self) -> Path:
        return self.supp_root / "analysis_outputs"

    @property
    def logs_dir(self) -> Path:
        return self.supp_root / "logs"

    @property
    def tmp_dir(self) -> Path:
        return self.supp_root / "_tmp"

    @property
    def news_links_dir(self) -> Path:
        return self.news_links_history_dir

    @property
    def news_pages_dir(self) -> Path:
        return self.news_pages_history_dir

    @property
    def llm_parts_dir(self) -> Path:
        return self.supp_root / "llm_outputs" / "parts_100"

    @property
    def funds_dir(self) -> Path:
        return self.supp_root / "funds_data"

    @property
    def market_dir(self) -> Path:
        return self.supp_root / "market_data"

    def ensure_base_dirs(self) -> None:
        for path in [
            self.outputs_dir,
            self.logs_dir,
            self.news_data_dir,
            self.news_links_dir,
            self.news_pages_dir,
            self.news_links_history_dir,
            self.news_links_runs_dir,
            self.news_pages_history_dir,
            self.news_pages_runs_dir,
            self.news_merged_dir,
            self.funds_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


PATHS = PipelinePaths()
