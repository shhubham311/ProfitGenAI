import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Paths
    DATA_DIR: str = "data"
    AMAZON_PRODUCTS_PATH: str = "data/amazon_products.csv"
    AMAZON_CATEGORIES_PATH: str = "data/amazon_categories.csv"
    CLICKSTREAM_PATH: str = "data/e-shop clothing 2008.csv"
    
    # Model Settings
    EMBEDDING_MODEL: str = 'all-MiniLM-L6-v2'
    SAMPLE_SIZE: int = 50000  # Keep this manageable for Render's free tier RAM
    
    # Business Logic
    MARGIN_WEIGHT: float = 0.3
    SIMILARITY_WEIGHT: float = 0.5
    BEHAVIOR_WEIGHT: float = 0.2
    MAX_UPSELL_RATIO: float = 1.5
    
    # API Keys
    GROQ_API_KEY: str = ""
    LLM_MODEL: str = "meta-llama/llama-4-maverick-17b-128e-instruct"

    class Config:
        env_file = ".env"

config = Settings()