from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
import sys
import os
# --- FIX FOR OMP ERROR #15 ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# -----------------------------
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config
from src.data_loader import DataLoader
from src.behavior_analyzer import BehaviorAnalyzer
from src.content_engine import ContentEngine
from src.sales_agent import SalesAgent
from src import db

app = FastAPI(title="ProfitGenAI")

# --- Frontend Configuration ---
templates = Jinja2Templates(directory="src/templates")
app.mount("/static", StaticFiles(directory="src/static"), name="static")

# --- Global State ---
behavior_analyzer = None
content_engine = None
sales_agent = None

# --- Pydantic Models ---
class RecommendationRequest(BaseModel):
    user_email: Optional[str] = None
    asin: str

class SearchRequest(BaseModel):
    user_email: Optional[str] = None
    query: str = ""
    user_persona: str = "Standard Shopper"

class CartActionRequest(BaseModel):
    email: str
    asin: str

class CheckoutRequest(BaseModel):
    email: str

class AuthRequest(BaseModel):
    email: str
    password: str
    persona: str = "Standard Shopper"

class UserDataRequest(BaseModel):
    email: str

class PersonaUpdateRequest(BaseModel):
    email: str
    persona: str

# --- Startup Event ---
@app.on_event("startup")
def startup_event():
    global behavior_analyzer, content_engine, sales_agent
    
    print("--- Starting ProfitGenAI System ---")
    
    # 1. Initialize Database (Persistent)
    print("Initializing SQLite Database...")
    db.init_db()
    
    # 2. Load Data Models
    
    # --- [OLD APPROACH] High Memory Usage (Commented Out) ---
    # products_df = DataLoader.load_amazon_catalog() # <--- This caused the crash on Render
    # clickstream_df = DataLoader.load_clickstream()
    
    # print("Initializing Behavior Analyzer...")
    # behavior_analyzer = BehaviorAnalyzer(clickstream_df)
    
    # print("Initializing Content Engine...")
    # content_engine = ContentEngine(products_df) # <--- Processed embeddings in RAM
    
    # --- [NEW APPROACH] Optimized for Render Free Tier ---
    # We still load clickstream data as it is small and needed for rules
    clickstream_df = DataLoader.load_clickstream()
    
    print("Initializing Behavior Analyzer...")
    behavior_analyzer = BehaviorAnalyzer(clickstream_df)
    
    print("Initializing Content Engine (Loading Artifacts)...")
    # Initialize without arguments. 
    # This triggers the new _load_artifacts() method in ContentEngine 
    # to read your 'startups_data.pkl' file instead of calculating in RAM.
    content_engine = ContentEngine() 
    
    print("Initializing Sales Agent...")
    sales_agent = SalesAgent(behavior_analyzer.get_rules())
    
    print("--- System Ready ---")

# --- Helper Functions ---
def get_user_by_email(email: str) -> Optional[dict]:
    """Wrapper to fetch user data safely."""
    user = db.get_user_by_email(email)
    return user

# --- Endpoints ---
@app.api_route("/", methods=["GET", "HEAD"])
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/signup")
async def signup(req: AuthRequest):
    try:
        user = db.create_user_secure(req.email, req.password, req.persona)
        return {"message": "User created successfully", "email": user["email"]}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/login")
async def login_user(req: AuthRequest):
    """Logs in an existing user."""
    # Verify login (checks password hash internally)
    user = db.verify_login(req.email, req.password)
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    # Return user data (cart, history, etc.)
    full_user = get_user_by_email(req.email)
    return {
        "message": "Login successful",
        "email": full_user["email"],
        "persona": full_user["persona"],
        "last_login": full_user.get("last_login")
    }

@app.post("/update_persona")
async def update_persona(req: PersonaUpdateRequest):
    """Updates user persona and persists it."""
    user = get_user_by_email(req.email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    db.update_user_persona(req.email, req.persona)
    return {"message": f"Persona updated to {req.persona}", "persona": req.persona}

@app.post("/get_user_data")
async def get_user_data(req: UserDataRequest):
    """Returns current user state (cart, history)."""
    user = get_user_by_email(req.email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.post("/get_history")
async def get_history(req: CheckoutRequest):
    """Returns the purchase history for a user, enriched with product details."""
    user = get_user_by_email(req.email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    history_asins = user.get("history", [])
    
    # Enrich history with product titles for better frontend display
    enriched_history = []
    if content_engine:
        for asin in history_asins:
            # Use content_engine.df (loaded from pickle)
            product_details = content_engine.df[content_engine.df['asin'] == asin]
            if not product_details.empty:
                enriched_history.append({
                    "asin": asin,
                    "title": product_details.iloc[0]['title'],
                    "price": product_details.iloc[0]['price']
                })
            else:
                # Handle case where a historical ASIN might not be in the current catalog
                enriched_history.append({"asin": asin, "title": "Product no longer available", "price": 0})
    
    return {"history": enriched_history}


@app.post("/get_cart")
async def get_cart(req: UserDataRequest):
    """Returns the user's current cart, enriched with product details."""
    user = get_user_by_email(req.email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    cart_asins = user.get("cart", [])
    
    enriched_cart = []
    if content_engine:
        for asin in cart_asins:
            product_details = content_engine.df[content_engine.df['asin'] == asin]
            if not product_details.empty:
                enriched_cart.append({
                    "asin": asin,
                    "title": product_details.iloc[0]['title'],
                    "price": product_details.iloc[0]['price']
                })
    
    return {"cart": enriched_cart}


@app.post("/add_to_cart")
async def add_to_cart(req: CartActionRequest):
    """Adds item to user's cart."""
    user = get_user_by_email(req.email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if product exists in catalog
    if content_engine.df[content_engine.df['asin'] == req.asin].empty:
        raise HTTPException(status_code=404, detail="Product ASIN not found")
    
    # Add to cart
    db.add_to_cart(user["id"], req.asin)
    
    # Return updated user data
    updated_user = get_user_by_email(req.email)
    return {"message": "Item added to cart", "cart": updated_user["cart"]}

@app.post("/remove_from_cart")
async def remove_from_cart(req: CartActionRequest):
    """Removes item from user's cart."""
    user = get_user_by_email(req.email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    db.remove_from_cart(user["id"], req.asin)
    
    updated_user = get_user_by_email(req.email)
    return {"message": "Item removed from cart", "cart": updated_user["cart"]}

@app.post("/checkout")
async def checkout(req: CheckoutRequest):
    """Purchases all items in cart."""
    user = get_user_by_email(req.email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if not user["cart"]:
        raise HTTPException(status_code=400, detail="Cart is empty")
    
    items_count = db.checkout(user["id"])
    
    return {"message": "Purchase successful!", "total_items": items_count}

@app.post("/buy_item")
async def buy_single_item(req: CartActionRequest):
    """Immediately purchases a single item (no cart)."""
    user = get_user_by_email(req.email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if content_engine.df[content_engine.df['asin'] == req.asin].empty:
        raise HTTPException(status_code=404, detail="Product ASIN not found")
    conn = db.get_db_connection() # Manual connection for direct insert
    cursor = conn.cursor()
    cursor.execute("INSERT INTO purchase_history (user_id, asin) VALUES (?, ?)", (user["id"], req.asin))
    conn.commit()
    conn.close()
    
    # Return updated history
    updated_user = get_user_by_email(req.email)
    return {"message": "Item purchased!", "history": updated_user["history"]}

@app.post("/search")
async def search_products(req: SearchRequest):
    """Searches catalog and returns ALL results."""
    if not content_engine or not sales_agent:
        raise HTTPException(status_code=503, detail="System not ready yet")
    
    if not req.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # 1. Search by Text (Increased k for full results)
    # Optimized ContentEngine handles the query encoding internally
    raw_results = content_engine.search_by_text(req.query, k=50)
    
    if raw_results.empty:
        return {"results": []}

    # 2. Rerank - NO LIMIT (Show all)
    ranked_items = sales_agent.rerank(
        candidates=raw_results,
        current_price=raw_results['price'].mean(), 
        persona=req.user_persona,
        limit=None 
    )

    # 3. Format Response
    recommendations = [
        {
            "asin": row["asin"],
            "title": row["title"],
            "price": row["price"],
            "final_score": row["final_score"]
        }
        for _, row in ranked_items.iterrows()
    ]
    
    return {
        "query": req.query,
        "results": recommendations
    }

@app.post("/recommend")
async def get_recommendation(req: RecommendationRequest):
    """Recommends upsell items based on context."""
    if not content_engine or not sales_agent:
        raise HTTPException(status_code=503, detail="System not ready yet")

    # Identify user ID
    email = req.user_email if req.user_email else None
    user = get_user_by_email(email) if email else None
    
    # CONTEXT SELECTION LOGIC
    context_asin = req.asin
    context_price = 0
    
    if user and user["cart"]:
        # Priority 1: Last item in cart
        context_asin = user["cart"][-1]
    elif user and user["history"]:
        # Priority 2: Last item in history
        context_asin = user["history"][-1]
    
    # Fetch Context Product
    # Using the pre-loaded dataframe in ContentEngine
    context_row = content_engine.df[content_engine.df['asin'] == context_asin]
    if context_row.empty:
        raise HTTPException(status_code=404, detail="Product ASIN not found")
    
    context_item = context_row.iloc[0].to_dict()
    context_price = context_item['price']
    
    # Get Similar Items
    similar_items = content_engine.search_by_asin(context_asin, k=20)
    
    # Rerank - LIMIT to 3 (Upsell)
    ranked_items = sales_agent.rerank(
        candidates=similar_items,
        current_price=context_price,
        persona=user["persona"] if user else "Standard Shopper",
        limit=3
    )
    
    # Generate Pitch
    pitch = sales_agent.generate_pitch(
        context=context_item,
        recs=ranked_items,
        persona=user["persona"] if user else "Standard Shopper"
    )
    
    # Format Response
    recommendations = [
        {
            "asin": row["asin"],
            "title": row["title"],
            "price": row["price"],
            "final_score": row["final_score"]
        }
        for _, row in ranked_items.iterrows()
    ]

    return {
        "context_product": {
            "asin": context_item['asin'],
            "title": context_item['title'],
            "price": context_item['price']
        },
        "sales_pitch": pitch,
        "recommendations": recommendations
    }