from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    chroma_path: str = "./chroma_db"
    chroma_collection: str = "yelp_reviews"
    embed_model: str = "nomic-text-v1.5"
    embed_dimensions: int = 256

    class Config:
        env_file = ".env"


settings = Settings()
