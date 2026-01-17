import pickle
import faiss
import pandas as pd
import os
from src.config import config

class ContentEngine:
    def __init__(self, products_df=None):
        # Note: products_df arg is kept for compatibility but ignored
        self.index = None
        self.df = None
        self._load_artifacts()

    def _load_artifacts(self):
        file_path = 'startups_data.pkl'
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found! Did you run generate_artifacts.py?")
            
        print(f"Loading pre-computed artifacts from {file_path}...")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        self.df = data['df']
        embeddings = data['embeddings']
        
        print("Building FAISS Index...")
        # Normalize for Cosine Similarity
        faiss.normalize_L2(embeddings)
        
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)
        print("Engine Ready.")

    def search_by_asin(self, asin: str, k: int = 20):
        product_row = self.df[self.df['asin'] == asin]
        if product_row.empty:
            return pd.DataFrame()
        
        # We need the embedding for this ASIN. 
        # Since we pre-calculated, we can look it up by index.
        idx = product_row.index[0]
        
        # Reconstruct the vector from the index (FAISS allows this)
        query_vec = self.index.reconstruct(int(idx)).reshape(1, -1)
        
        distances, indices = self.index.search(query_vec, k + 1)
        
        results = []
        for i in range(len(indices[0])):
            original_idx = indices[0][i]
            if original_idx == idx: continue 
            item = self.df.iloc[original_idx].to_dict()
            item['similarity_score'] = float(distances[0][i])
            results.append(item)
            
        return pd.DataFrame(results)

    def search_by_text(self, query: str, k: int = 20):
        # For text queries, we still need the model to encode the *query*
        # But we load it strictly for this one operation, which is light on RAM
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        query_vec = model.encode([query])
        faiss.normalize_L2(query_vec)
        
        distances, indices = self.index.search(query_vec, k)
        
        results = []
        for i in range(len(indices[0])):
            original_idx = indices[0][i]
            item = self.df.iloc[original_idx].to_dict()
            item['similarity_score'] = float(distances[0][i])
            results.append(item)
            
        return pd.DataFrame(results)