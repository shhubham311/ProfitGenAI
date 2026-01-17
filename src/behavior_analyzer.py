import pandas as pd
import numpy as np

class BehaviorAnalyzer:
    def __init__(self, clickstream_df: pd.DataFrame):
        self.df = clickstream_df
        self.persona_rules = self._analyze()

    def _analyze(self):
        # Aggregate by session
        session_stats = self.df.groupby('session_id').agg(
            session_length=('order', 'max'),
            avg_price=('price', 'mean'),
            distinct_cats=('page_1_main_category', 'nunique')
        ).reset_index()
        
        # Define Personas based on Price Quantiles
        q33 = session_stats['avg_price'].quantile(0.33)
        q66 = session_stats['avg_price'].quantile(0.66)
        
        def get_persona(avg):
            if avg <= q33: return "Budget Conscious"
            if avg >= q66: return "Premium Shopper"
            return "Standard Shopper"
        
        session_stats['persona'] = session_stats['avg_price'].apply(get_persona)
        
        # Calculate rules (max recommended price per persona)
        rules = session_stats.groupby('persona')['avg_price'].agg(['mean', 'max']).to_dict('index')
        
        # Add a buffer to max price (upsell potential)
        for p in rules:
            rules[p]['max_suggested_price'] = rules[p]['max'] * 1.2
            
        print(f"Persona Rules Generated: {rules}")
        return rules

    def get_rules(self):
        return self.persona_rules