# ProfitGenAI: AI-Powered E-commerce Sales Agent | [Live](https://huggingface.co/spaces/shub3Ed/ProfitGenAI)

## Overview

ProfitGenAI is an advanced e-commerce platform that demonstrates the power of AI in creating a personalized and profitable shopping experience. It goes beyond standard recommendation systems by incorporating a "sales agent" that understands user behavior and re-ranks product suggestions to maximize business value while still being relevant to the user.

## Features

- **AI-Powered Recommendations**: Utilizes a sales agent to re-rank product suggestions based on user personas to maximize profit.
- **Personalized Sales Pitches**: Generates dynamic sales pitches for recommended products using the Groq API.
- **User Personas**: Analyzes user behavior to classify them into personas like "Budget Conscious" or "Premium Shopper".
- **Product Search**: Fast, semantic search for products using sentence transformers and FAISS.
- **User Authentication**: Secure user signup and login.
- **Shopping Cart**: Fully functional shopping cart with add, remove, and view capabilities.
- **Purchase History**: Users can view their past purchases.
- **Interactive Frontend**: A single-page application built with HTML, CSS, and vanilla JavaScript.

## Core Concepts

### User Personas

At the heart of ProfitGenAI's intelligence is the concept of user personas. The `BehaviorAnalyzer` module processes clickstream data to classify users into one of three categories:
- **Budget Conscious**: Users who are sensitive to price.
- **Standard Shopper**: The average user.
- **Premium Shopper**: Users who are willing to spend more for higher quality or premium products.

These personas are determined by analyzing metrics like the average price of products viewed and the number of unique categories explored in a session.

### Profit-Driven Reranking

Standard recommendation engines often prioritize similarity. ProfitGenAI takes it a step further. The `SalesAgent` module implements a reranking algorithm that considers:
- **Similarity Score**: How similar a candidate product is to the user's current context (e.g., last item added to cart).
- **Price Delta**: The difference in price between the candidate product and the context product.
- **User Persona**: The user's sensitivity to price changes.

The final score for each recommendation is a weighted combination of these factors, allowing the system to strategically upsell or cross-sell products in a way that aligns with the user's likely spending habits.

## High-Level Architecture

The application is built with a Python/FastAPI backend and a vanilla JavaScript frontend, consisting of three main layers:

1.  **Data Layer**:
    -   **SQLite Database (`db.py`)**: Persists user data, including login credentials, shopping carts, and purchase history.
    -   **Data Loaders (`data_loader.py`)**: Loads and preprocesses product catalogs and behavioral data from CSV files at startup.

2.  **Backend Logic Layer**:
    -   **Content Engine (`content_engine.py`)**: Creates vector embeddings of product titles using sentence transformers and uses a FAISS index for efficient similarity searches. It powers both text-based search and finding items similar to a given product.
    -   **Behavior Analyzer (`behavior_analyzer.py`)**: Determines the rules for user personas from historical data.
    -   **Sales Agent (`sales_agent.py`)**: The core AI component. It takes product candidates and a user context to rerank them for profitability and uses the Groq API to generate persuasive sales pitches.

3.  **API & Presentation Layer**:
    -   **FastAPI App (`api.py`)**: Exposes a RESTful API for all frontend operations, including user authentication, search, cart management, and recommendations.
    -   **Frontend (`index.html`)**: A dynamic single-page application that interacts with the backend API to provide a seamless user experience.

## Workflow Details

1.  **Initialization**: On startup, the server loads all necessary data into memory, builds the FAISS index for product search, and determines the persona-based rules.
2.  **Authentication**: A user signs up or logs in. Their session is maintained by passing their email (as a user identifier) in subsequent API calls.
3.  **Interaction**:
    -   When a user **searches**, the `ContentEngine` provides a list of relevant products.
    -   When a user **adds an item to their cart**, this action can trigger a new recommendation from the `SalesAgent`. The last item added becomes the primary "context" for the recommendation.
    -   The `SalesAgent` gets similar items from the `ContentEngine`, re-ranks them based on the user's persona and the context item's price, and generates a sales pitch for the top recommendations.
4.  **Checkout**: The user can view their cart (with enriched product data fetched from a dedicated endpoint) and "purchase" the items, which moves them to their permanent purchase history in the SQLite database.

This combination of data analysis, machine learning for search, and generative AI for persuasion creates a powerful and intelligent e-commerce system.

## How to Run

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set up environment variables**:
    Create a `.env` file in the root directory and add your Groq API key:
    ```
    GROQ_API_KEY=your_api_key_here
    ```

3.  **Run the application**:
    ```bash
    uvicorn src.api:app --reload
    ```
    The application will be available at `http://127.0.0.1:8000`.

## Project Structure

```
├───data/                 # Sample datasets
├───src/                  # Source code
│   ├───templates/        # Frontend HTML
│   │   └───index.html
│   ├───static/           # Frontend CSS
│   │   └───style.css
│   ├───api.py            # FastAPI application
│   ├───behavior_analyzer.py # User persona analysis
│   ├───content_engine.py # Product search and similarity
│   ├───data_loader.py    # Data loading and preprocessing
│   ├───db.py             # SQLite database management
│   └───sales_agent.py    # Recommendation and sales pitch logic
├───.env                  # Environment variables
├───README.md             # This file
├───requirements.txt      # Python dependencies
└───Summary.md            # Detailed project summary
```
