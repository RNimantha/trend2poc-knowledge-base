from pydantic import BaseSettings, Field, validator
from typing import Any


class Settings(BaseSettings):
    ORCHESTRATOR_PORT: int = Field(..., description="Port number where the orchestrator FastAPI server listens")
    AGENT1_PORT: int = Field(..., description="Port number where Agent 1 listens")
    AGENT2_PORT: int = Field(..., description="Port number where Agent 2 listens")

    @validator("ORCHESTRATOR_PORT", "AGENT1_PORT", "AGENT2_PORT")
    def port_must_be_valid(cls, v: int) -> int:
        if not (1024 <= v <= 65535):
            raise ValueError(f"Port {v} is not in valid range 1024-65535")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    settings = Settings()
    return settings
