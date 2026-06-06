from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OPENAI_API_KEY: str

    model_name: str = "claude-3-5-sonnet-latest"
    embedding_model_name: str = "text-embedding-3-large"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
