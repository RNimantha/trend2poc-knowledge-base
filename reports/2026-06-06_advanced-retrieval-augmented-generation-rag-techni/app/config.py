from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=True,
    )


settings = Settings()
