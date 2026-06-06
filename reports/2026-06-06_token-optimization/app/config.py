from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    model_api_key: Optional[str] = None
    use_mock_model: bool = True

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
    )

    def __init__(self, **values) -> None:
        super().__init__(**values)
        # Convert use_mock_model from str if needed
        if isinstance(self.use_mock_model, str):
            self.use_mock_model = self.use_mock_model.lower() in ("true", "1", "yes")


settings = Settings()
