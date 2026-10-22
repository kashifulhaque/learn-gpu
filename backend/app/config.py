"""Application settings, read from the environment or a .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Caddy sends these after Authelia has authenticated the request.
    sso_proxy_secret: str = ""
    # Maps an Authelia username to the name that holds that reader's progress,
    # such as "ifkash:enderboi,ananya:michu". Unmapped users keep their own name.
    sso_users: str = ""

    @property
    def sso_names(self) -> dict[str, str]:
        pairs = (item.split(":", 1) for item in self.sso_users.split(",") if ":" in item)
        return {
            sso.strip().lower(): name.strip()
            for sso, name in pairs
            if sso.strip() and name.strip()
        }

    # App
    data_dir: Path = REPO_ROOT / "data"
    public_url: str = "http://localhost:8000"
    content_dir: Path = REPO_ROOT / "content"
    frontend_dist: Path = REPO_ROOT / "frontend" / "dist"
    # The PDF book and its manifest, written by frontend/scripts/build-book.mjs
    # on the deploying machine and mounted into the container.
    book_dir: Path = REPO_ROOT / "book"

    # GPU providers
    gpu_provider: str = "runpod"
    # The accelerator the labs ask for, for display.
    gpu_type: str = "A100-80GB"
    runpod_api_key: str = ""
    runpod_endpoint_id: str = ""

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
