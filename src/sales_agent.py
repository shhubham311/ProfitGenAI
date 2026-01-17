import pandas as pd
from groq import Groq
from src.config import config

class SalesAgent:
    def __init__(self, persona_rules):
        self.rules = persona_rules
        self.client = Groq(api_key=config.GROQ_API_KEY) if config.GROQ_API_KEY else None

    def rerank(self, candidates: pd.DataFrame, current_price: float, persona: str, limit: int = None):
        """Re-ranks items based on Profit, Similarity, and Constraints."""
        scored = []
        
        # 1. Get Persona Constraints
        p_rules = self.rules.get(persona, self.rules.get("Standard Shopper"))
        max_price_suggestion = p_rules['max_suggested_price']
        
        # 2. Global Upsell Cap
        global_cap = current_price * config.MAX_UPSELL_RATIO
        
        for _, row in candidates.iterrows():
            # Constraint: Price Cap
            if row['price'] > max(global_cap, max_price_suggestion):
                continue
                
            # Calculate Profit Score
            margin = row['price'] - row['cost_price']
            margin_pct = (margin / row['price']) * 100
            
            # Normalize Profit (0-1 range, assuming 80% is max high margin)
            norm_profit = min(margin_pct / 80.0, 1.0)
            
            # Calculate Similarity Score
            norm_sim = row['similarity_score']
            
            # Final Weighted Score
            final_score = (
                (norm_sim * config.SIMILARITY_WEIGHT) + 
                (norm_profit * config.MARGIN_WEIGHT)
            )
            
            row['final_score'] = final_score
            scored.append(row)
            
        # Sort by score descending
        sorted_df = pd.DataFrame(scored).sort_values(by='final_score', ascending=False)
        
        # Apply Limit if provided
        if limit:
            return sorted_df.head(limit)
        
        return sorted_df

    def generate_pitch(self, context, recs, persona):
        if not self.client:
            return self._mock_pitch(context, recs, persona)
            
        recs_text = "\n".join([f"- {r['title']} (${r['price']})" for _, r in recs.iterrows()])
        
        prompt = f"""
        You are a helpful Sales Executive.
        User Persona: {persona}.
        User is viewing: {context['title']} (${context['price']}).
        
        Recommended items:
        {recs_text}
        
        Write a persuasive, 2-sentence pitch for these items.
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=config.LLM_MODEL,
                temperature=0.5
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM Error: {e}")
            return self._mock_pitch(context, recs, persona)

    def _mock_pitch(self, context, recs, persona):
        if recs.empty:
            return "I'm looking for the best options for you right now."
        top = recs.iloc[0]
        return f"Since you're looking at {context['title']}, I highly recommend {top['title']}. It fits your {persona} profile perfectly."