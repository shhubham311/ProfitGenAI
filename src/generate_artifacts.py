# generate_artifacts.py
import pandas as pd
import pickle
import sys
import os
from sentence_transformers import SentenceTransformer

# Setup imports from your existing project
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.config import config
from src.data_loader import DataLoader

def generate():
    print("1. Loading Catalog (Full 50k rows)...")
    # Temporarily override config to ensure full load if needed
    # config.SAMPLE_SIZE = 50000 
    
    # Reuse your existing robust loader
    df = DataLoader.load_amazon_catalog()
    
    print(f"   Loaded {len(df)} products.")

    print("2. Generating Embeddings (This may take a minute)...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    
    # Create the search text field exactly like before
    df['search_text'] = (
        df['title'].fillna('') + " " + 
        df['category_name'].fillna('')
    )
    
    embeddings = model.encode(df['search_text'].tolist(), show_progress_bar=True)
    
    print("3. Saving Artifacts...")
    # We save a dictionary containing the Dataframe and the Embeddings
    with open('startups_data.pkl', 'wb') as f:
        pickle.dump({'df': df, 'embeddings': embeddings}, f)
        
    print("Done! 'startups_data.pkl' is ready to upload.")

if __name__ == "__main__":
    generate()