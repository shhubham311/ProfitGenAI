import pandas as pd
import numpy as np
import logging
from src.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    @staticmethod
    def load_amazon_catalog():
        logger.info("Loading Amazon catalog...")
        # Load Products
        products = pd.read_csv(config.AMAZON_PRODUCTS_PATH, nrows=config.SAMPLE_SIZE)
        
        # Load Categories
        categories = pd.read_csv(config.AMAZON_CATEGORIES_PATH)
        
        # Cleaning
        products = products.dropna(subset=['title', 'price', 'category_id'])
        products = products[products['price'] > 0]
        
        # Merge
        # Note: Amazon 'category_id' maps to 'id' in categories
        merged = products.merge(
            categories, 
            left_on='category_id', 
            right_on='id', 
            how='left'
        )
        
        # Feature Engineering
        # 1. Cost Price Proxy (Assume 30% margin baseline)
        merged['cost_price'] = merged['price'] * 0.7
        
        # 2. Quality Score (Normalized Stars)
        # Fill missing stars with median to avoid bias
        median_stars = merged['stars'].median()
        merged['stars'] = merged['stars'].fillna(median_stars)
        merged['quality_score'] = merged['stars'] / 5.0
        
        logger.info(f"Catalog loaded: {len(merged)} products.")
        return merged.reset_index(drop=True)

    @staticmethod
    def load_clickstream():
        logger.info("Loading UCI Clickstream...")
        # UCI data is typically semicolon separated
        df = pd.read_csv(config.CLICKSTREAM_PATH, sep=';')
        
        # Normalize column names (Lowercase, remove spaces)
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(' ', '_')
            .str.replace('(', '')
            .str.replace(')', '')
        )
        
        logger.info(f"Clickstream loaded: {len(df)} records.")
        return df