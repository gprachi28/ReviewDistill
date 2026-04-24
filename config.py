from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    llm_base_url: str = "http://localhost:8001/v1"
    llm_model: str = "mlx-community/Qwen2.5-7B-Instruct-4bit"
    chroma_path: str = "./chroma_db"
    chroma_collection: str = "yelp_reviews_nola"
    embed_model: str = "nomic-text-v1.5"
    embed_dimensions: int = 256
    sqlite_path: str = "./yelp_reviews.db"
    session_ttl: int = 1800  # 30 minutes in seconds


settings = Settings()
