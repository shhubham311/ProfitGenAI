import pickle
import faiss
import pandas as pd
import os
from src.config import config

class ContentEngine:
    def __init__(self, products_df=None):
        self.index = None
        self.df = None
        self.model = None  # Initialize as None
        self._load_artifacts()

    def _load_artifacts(self):
        # Robust path finding for Render
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'startups_data.pkl')
        
        if not os.path.exists(file_path):
             # Fallback check
             file_path = 'startups_data.pkl'
             if not os.path.exists(file_path):
                 raise FileNotFoundError(f"Could not find startups_data.pkl in src/ or root.")
            
        print(f"Loading pre-computed artifacts from {file_path}...")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        self.df = data['df']
        embeddings = data['embeddings']
        
        print("Building FAISS Index...")
        faiss.normalize_L2(embeddings)
        
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)
        print("Engine Ready.")

    def search_by_asin(self, asin: str, k: int = 20):
        product_row = self.df[self.df['asin'] == asin]
        if product_row.empty:
            return pd.DataFrame()
        
        idx = product_row.index[0]
        
        # FIX: Explicit int cast for FAISS
        query_vec = self.index.reconstruct(int(idx)).reshape(1, -1)
        
        distances, indices = self.index.search(query_vec, k + 1)
        
        results = []
        for i in range(len(indices[0])):
            original_idx = indices[0][i]
            if original_idx == int(idx): continue 
            item = self.df.iloc[original_idx].to_dict()
            item['similarity_score'] = float(distances[0][i])
            results.append(item)
            
        return pd.DataFrame(results)

    def search_by_text(self, query: str, k: int = 20):
        # --- OPTIMIZATION START ---
        # Only load the model if it hasn't been loaded yet
        if self.model is None:
            print("Loading Embedding Model (One-time operation)...")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        # --- OPTIMIZATION END ---
        
        query_vec = self.model.encode([query])
        faiss.normalize_L2(query_vec)
        
        distances, indices = self.index.search(query_vec, k)
        
        results = []
        for i in range(len(indices[0])):
            original_idx = indices[0][i]
            item = self.df.iloc[original_idx].to_dict()
            item['similarity_score'] = float(distances[0][i])
            results.append(item)
            
        return pd.DataFrame(results)