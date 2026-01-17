import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.config import config

class ContentEngine:
    def __init__(self, products_df: pd.DataFrame):
        self.df = products_df
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.index = None
        self.mapping = []
        self._build_index()

    def _build_index(self):
        print("Generating Embeddings...")
        self.df['search_text'] = (
            self.df['title'].fillna('') + " " + 
            self.df['category_name'].fillna('')
        )
        
        embeddings = self.model.encode(self.df['search_text'].tolist(), show_progress_bar=True)
        faiss.normalize_L2(embeddings)
        
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)
        
        self.mapping = np.arange(len(self.df))
        print("Vector Index Built.")

    def search_by_asin(self, asin: str, k: int = 20):
        """Finds similar products given a specific ASIN ID."""
        product_row = self.df[self.df['asin'] == asin]
        if product_row.empty:
            return pd.DataFrame()
        
        idx = product_row.index[0]
        query_text = self.df.iloc[idx]['search_text']
        
        query_vec = self.model.encode([query_text])
        faiss.normalize_L2(query_vec)
        
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
        """Searches products based on a natural text query."""
        if not self.index:
            return pd.DataFrame()

        # Encode the user's text query
        query_vec = self.model.encode([query])
        faiss.normalize_L2(query_vec)
        
        # Search FAISS index
        distances, indices = self.index.search(query_vec, k)
        
        results = []
        for i in range(len(indices[0])):
            original_idx = indices[0][i]
            item = self.df.iloc[original_idx].to_dict()
            item['similarity_score'] = float(distances[0][i])
            results.append(item)
            
        return pd.DataFrame(results)