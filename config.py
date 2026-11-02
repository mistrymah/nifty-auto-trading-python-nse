import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).parent
ENV_FILE = PROJECT_ROOT / ".env"


class Config(BaseModel):
    upstox_api_key: str = ""
    upstox_access_token: str = ""
    glm_api_key: str = ""
    groq_api_key: str = ""
    start_time: str = "09:15:00"
    end_time: str = "15:30:00"
    max_trades_per_day: int = 3
    lot_size: int = 65
    webhook_url: str = ""
    webhook_enabled: bool = False
    webhook_events: str = "signal,trade,exit"
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    fcc_api_key: str = "freecc"
    fcc_url: str = "http://127.0.0.1:8082/v1"

    @classmethod
    def load(cls) -> "Config":
        if ENV_FILE.exists():
            with open(ENV_FILE) as f:
                lines = f.readlines()
                data = {}
                for line in lines:
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        data[k.lower()] = v.strip().strip("'\"")
                return cls(**data)
        return cls()


config = Config.load()

for k, v in config.model_dump().items():
    if v:
        os.environ.setdefault(k.upper(), str(v))
