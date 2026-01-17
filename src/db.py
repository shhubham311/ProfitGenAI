import sqlite3
import bcrypt
import datetime
from typing import List, Optional, Dict
import os

DB_NAME = os.getenv("DB_PATH", "profitgenai.db")

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row # Access columns by name
    return conn

def init_db():
    """Initializes database and creates tables."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Create Secure Users Table (Check IF NOT EXISTS)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            persona TEXT NOT NULL DEFAULT 'Standard Shopper',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')
    
    # 2. Create Cart Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cart_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            asin TEXT NOT NULL,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    
    # 3. Create Purchase History Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS purchase_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            asin TEXT NOT NULL,
            purchased_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database initialized successfully.")

# --- AUTH OPERATIONS ---

def create_user_secure(email: str, plain_password: str, persona: str):
    """Creates a user with hashed password."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Hash password
    password_hash = bcrypt.hashpw(
        plain_password.encode('utf-8'), 
        bcrypt.gensalt()
    ).decode('utf-8') # Store as string
    
    try:
        cursor.execute(
            "INSERT INTO users (email, password_hash, persona) VALUES (?, ?, ?)",
            (email, password_hash, persona)
        )
        conn.commit()
        user_id = cursor.lastrowid
        return {"id": user_id, "email": email, "persona": persona}
    except sqlite3.IntegrityError:
        conn.close()
        raise ValueError("Email already exists")
    finally:
        conn.close()

def verify_login(email: str, plain_password: str) -> Optional[Dict]:
    """Verifies email and password."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Fetch user by email
    cursor.execute("SELECT id, email, password_hash, persona, last_login FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    
    if not user:
        conn.close()
        return None
    
    # 2. Verify Password Hash
    if not bcrypt.checkpw(
        plain_password.encode('utf-8'), 
        user["password_hash"].encode('utf-8')
    ):
        conn.close()
        return None
    
    # 3. Update Last Login (Activity Tracking)
    cursor.execute(
        "UPDATE users SET last_login = ? WHERE id = ?",
        (datetime.datetime.now(), user["id"])
    )
    conn.commit()
    conn.close()
    
    return {
        "id": user["id"],
        "email": user["email"],
        "persona": user["persona"],
        "last_login": user["last_login"]
    }

def get_user_by_email(email: str) -> Optional[Dict]:
    """Fetches user data (including cart/history)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get User Details
    cursor.execute("SELECT id, email, persona FROM users WHERE email = ?", (email,))
    user_row = cursor.fetchone()
    
    if not user_row:
        conn.close()
        return None
    
    user_id = user_row["id"]
    
    # Get Cart
    cursor.execute("SELECT asin FROM cart_items WHERE user_id = ?", (user_id,))
    cart_rows = cursor.fetchall()
    cart = [row["asin"] for row in cart_rows]
    
    # Get History
    cursor.execute("SELECT asin FROM purchase_history WHERE user_id = ? ORDER BY purchased_at DESC", (user_id,))
    history_rows = cursor.fetchall()
    history = [row["asin"] for row in history_rows]
    
    conn.close()
    
    return {
        "id": user_id,
        "email": user_row["email"],
        "persona": user_row["persona"],
        "cart": cart,
        "history": history
    }

def update_user_persona(email: str, new_persona: str):
    """Updates the user's shopper persona."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE users SET persona = ? WHERE email = ?",
        (new_persona, email)
    )
    conn.commit()
    conn.close()

# --- CART OPERATIONS ---

def add_to_cart(user_id: int, asin: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO cart_items (user_id, asin) VALUES (?, ?)", (user_id, asin))
    conn.commit()
    conn.close()

def remove_from_cart(user_id: int, asin: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM cart_items WHERE user_id = ? AND asin = ?", (user_id, asin))
    conn.commit()
    conn.close()

def checkout(user_id: int) -> int:
    """Moves cart items to history and clears cart."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Get Cart Items
    cursor.execute("SELECT asin FROM cart_items WHERE user_id = ?", (user_id,))
    cart_items = cursor.fetchall()
    count = len(cart_items)
    
    # 2. Insert into History
    for item in cart_items:
        cursor.execute("INSERT INTO purchase_history (user_id, asin) VALUES (?, ?)", (user_id, item["asin"]))
    
    # 3. Clear Cart
    cursor.execute("DELETE FROM cart_items WHERE user_id = ?", (user_id,))
    
    conn.commit()
    conn.close()
    return count