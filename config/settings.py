from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    # LLM
    anthropic_api_key: str

    # Whisper
    whisper_model: Literal["tiny", "base", "small", "medium", "large-v3"] = "medium"
    whisper_device: Literal["cpu", "cuda", "auto"] = "auto"

    # Storage
    storage_backend: Literal["local"] = "local"
    local_storage_root: str = "./data"
    temp_dir: str = "./tmp"

    # Celery
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"

    # Export
    output_dir: str = "./output"

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
