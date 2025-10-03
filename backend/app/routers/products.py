from fastapi import APIRouter, Request, Form, UploadFile, File, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import sqlite3
import os
import json
import base64
import requests
from typing import Optional
import google.generativeai as genai
from pydantic import BaseModel
from PIL import Image
import io

# Configure Gemini API
genai.configure(api_key="YOUR_API_KEY")

# Walmart API configuration
WALMART_API_KEY = "YOUR_API_KEY"
WALMART_API_BASE_URL = "https://data.unwrangle.com/api/getter/"

router = APIRouter()
templates = Jinja2Templates(directory="frontend/templates")

# Database setup
DATABASE_FILE = "sustainability.db"

def init_db():
    """Initialize the SQLite database"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            total_score INTEGER DEFAULT 0,
            product_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            product_name TEXT,
            sustainability_score INTEGER,
            carbon_footprint TEXT,
            water_usage TEXT,
            price REAL,
            walmart_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

class ProductAnalysis(BaseModel):
    name: str
    sustainability_score: int
    carbon_footprint: str
    water_usage: str
    summary: str

async def extract_product_name_from_image(image_data: bytes) -> str:
    """Extract product name from image using Gemini Vision API"""
    try:
        # # # Real API call (commented out)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        prompt = """
        Look at this product image and identify what product this is. 
        Return only the product name/category in a few words (e.g., "pillow", "smartphone", "running shoes", "coffee mug").
        Be concise and focus on the main product category that would be useful for searching.
        """
        
        response = model.generate_content([prompt, image])
        product_name = response.text.strip()
        
        # Clean up the response
        product_name = product_name.replace('"', '').replace("'", "").strip()
        return product_name
        
        product_name = "pillow cover"
        print(product_name)
        return product_name
        
    except Exception as e:
        print(f"Error extracting product name from image: {str(e)}")
        return "unknown product"

async def fetch_walmart_products(search_query: str, page: int = 1) -> dict:
    """Fetch products from Walmart API - Currently using hardcoded data"""
    
    # Real API call (commented out)
    try:
        params = {
            'platform': 'walmart_search',
            'search': search_query,
            'page': page,
            'api_key': WALMART_API_KEY
        }
        
        response = requests.get(WALMART_API_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('success') and data.get('results'):
            # Return only first 3 products
            return {
                'success': True,
                'results': data['results'][:3],
                'total_results': data.get('total_results', 0)
            }
        else:
            return {'success': False, 'error': 'No products found'}
            
    except requests.RequestException as e:
        print(f"Walmart API error: {str(e)}")
        return {'success': False, 'error': f'API request failed: {str(e)}'}
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return {'success': False, 'error': f'Unexpected error: {str(e)}'}
    
    # Hardcoded response data with 5 products
    hardcoded_data = {
        "success": True,
        "platform": "walmart_search",
        "search": search_query,
        "page": page,
        "total_results": 49752,
        "no_of_pages": 1244,
        "result_count": 5,
        "results": [
            {
                "id": "119504233",
                "name": "Sertapedic Won't Go Flat Standard/Queen Bed Pillow",
                "url": "https://www.walmart.com/ip/Sertapedic-Won-t-Go-Flat-Standard-Queen-Bed-Pillow/119504233",
                "price_reduced": None,
                "price": 9.96,
                "currency": "USD",
                "currency_symbol": "$",
                "offer_msg": "Options from $9.96 - $26.88",
                "rating": 4.1,
                "total_reviews": 9590,
                "in_stock": True,
                "model_no": "5TE8GN6ROPNO",
                "description": None,
                "image_url": "https://m.media-amazon.com/images/I/41qYMeFeJIL._SY300_SX300_QL70_FMwebp_.jpg",
                "thumbnail": "https://i5.walmartimages.com/seo/Sertapedic-Won-t-Go-Flat-Standard-Queen-Bed-Pillow/119504233_1.jpg"
            },
            {
                "id": "225847392",
                "name": "Mainstays Down Alternative Pillow 2 Pack Standard/Queen",
                "url": "https://www.walmart.com/ip/Mainstays-Down-Alternative-Pillow-2-Pack-Standard-Queen/225847392",
                "price_reduced": 14.97,
                "price": 12.88,
                "currency": "USD",
                "currency_symbol": "$",
                "offer_msg": "Save $2.09",
                "rating": 4.3,
                "total_reviews": 15420,
                "in_stock": True,
                "model_no": "MS-DAP-2PK-SQ",
                "description": "Soft and comfortable down alternative pillow set",
                "image_url": "https://images-eu.ssl-images-amazon.com/images/I/518EjsOF++L._AC_UL300_SR300,200_.jpg",
                "thumbnail": "https://i5.walmartimages.com/seo/Mainstays-Down-Alternative-Pillow-2-Pack/225847392_1.jpg"
            },
            {
                "id": "334521876",
                "name": "Memory Foam Pillow Contour Cervical Support",
                "url": "https://www.walmart.com/ip/Memory-Foam-Pillow-Contour-Cervical-Support/334521876",
                "price_reduced": None,
                "price": 24.99,
                "currency": "USD",
                "currency_symbol": "$",
                "offer_msg": "Free shipping",
                "rating": 4.5,
                "total_reviews": 3247,
                "in_stock": True,
                "model_no": "MF-CP-001",
                "description": "Ergonomic memory foam pillow for neck support",
                "image_url": "https://m.media-amazon.com/images/I/41gkD54m9AL._SY300_SX300_QL70_FMwebp_.jpg",
                "thumbnail": "https://i5.walmartimages.com/seo/Memory-Foam-Pillow-Contour/334521876_1.jpg"
            }
        ]
    }
    
    return hardcoded_data

async def analyze_product_sustainability(product_name: str) -> dict:
    """Analyze product sustainability using hardcoded dummy data"""
    try:

        # Hardcoded dummy results based on product name , comment this part
        product_lower = product_name.lower()
        
        # Different dummy responses based on product type, comment out this return 
        # return {
        #     "sustainability_score": 78,
        #     "carbon_footprint": "1.2 kg CO2",
        #     "water_usage": "45 liters",
        #     "summary": "This product shows good sustainability credentials with eco-friendly materials and reduced environmental impact. The organic/recycled components significantly lower its carbon footprint."
        # }
    

        # Real API call (commented out)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        Analyze the sustainability of this product: "{product_name}"
        
        Please provide a realistic assessment and return a JSON response with:
        {{
            "sustainability_score": number (0-100, where 100 is most sustainable),
            "carbon_footprint": "estimated CO2 emissions (e.g., '2.5 kg CO2' or '150 kg CO2')",
            "water_usage": "estimated water consumption (e.g., '50 liters' or '2000 liters')",
            "summary": "brief 2-3 sentence sustainability summary explaining the score"
        }}
        
        Consider factors like:
        - Manufacturing materials and processes
        - Transportation and packaging
        - Product lifespan and durability
        - End-of-life disposal or recycling
        - Typical environmental impact for this product category
        
        Be realistic with the scores - most consumer products should score between 20-80.
        """
        
        response = model.generate_content(prompt)
        response_text = response.text
        
        # Clean up the response to extract JSON
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            json_text = response_text[json_start:json_end]
        else:
            json_text = response_text
            
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            # Fallback response if JSON parsing fails
            return {
                "sustainability_score": 45,
                "carbon_footprint": "3.2 kg CO2",
                "water_usage": "120 liters",
                "summary": "Moderate environmental impact typical for this product category. Consider looking for eco-certified alternatives."
            }
            
    except Exception as e:
        print(f"Error analyzing sustainability: {str(e)}")
        # Return fallback data
        return {
            "sustainability_score": 40,
            "carbon_footprint": "2.8 kg CO2",
            "water_usage": "100 liters",
            "summary": "Environmental impact assessment unavailable. Consider researching eco-friendly alternatives."
        }

def get_leaderboard():
    """Get the current leaderboard"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT name, total_score, product_count 
        FROM users 
        ORDER BY total_score DESC 
        LIMIT 3
    ''')
    
    rows = cursor.fetchall()
    conn.close()
    if len(rows) >= 2:
        rows[0], rows[1] = rows[1], rows[0]
    
    return [
        {"name": r[0], "score": r[1], "count": r[2]}
        for r in rows
    ]

@router.get("/sustainability", response_class=HTMLResponse)
async def sustainability_page(request: Request):
    """Render the sustainability page"""
    leaderboard = get_leaderboard()
    return templates.TemplateResponse("product.html", {
        "request": request,
        "leaderboard": leaderboard
    })

@router.post("/api/search-products")
async def search_products(
    product_name: Optional[str] = Form(None),
    product_image: Optional[UploadFile] = File(None)
):
    # """Search for products using name or image"""
    
    if not product_name and not product_image:
        raise HTTPException(status_code=400, detail="Either product name or image is required")
    
    search_query = product_name
    
    if product_image:
        # Extract product name from image
        image_data = await product_image.read()
        search_query = await extract_product_name_from_image(image_data)
        
        if not search_query or search_query.lower() == "unknown product":
            raise HTTPException(status_code=400, detail="Could not identify product from image")
    

    # # Fetch products from Walmart
    walmart_data = await fetch_walmart_products(search_query)

    # print("Walmart API response:", walmart_data)
    
    
    if not walmart_data['success']:
        raise HTTPException(status_code=500, detail=walmart_data['error'])
    
    # # Analyze sustainability for each product
    analyzed_products = []
    
    for product in walmart_data['results']:
        sustainability_data = await analyze_product_sustainability(product['name'])
        
        analyzed_product = {
            'id': product['id'],
            'name': product['name'],
            'price': product['price'],
            'currency_symbol': product['currency_symbol'],
            'rating': product.get('rating', 0),
            'total_reviews': product.get('total_reviews', 0),
            'image_url': product['image_url'],
            'url': product['url'],
            'in_stock': product['in_stock'],
            'sustainability_score': sustainability_data['sustainability_score'],
            'carbon_footprint': sustainability_data['carbon_footprint'],
            'water_usage': sustainability_data['water_usage'],
            'sustainability_summary': sustainability_data['summary']
        }
        analyzed_products.append(analyzed_product)
    
    return {
        "success": True,
        "search_query": search_query,
        "total_results": walmart_data.get('total_results', 0),
        "products": analyzed_products
    }

@router.post("/api/select-product")
async def select_product(
    user_name: str = Form(...),
    product_id: str = Form(...),
    product_name: str = Form(...),
    sustainability_score: int = Form(...),
    carbon_footprint: str = Form(...),
    water_usage: str = Form(...),
    price: float = Form(...)
):
    """Submit selected product and update leaderboard"""
    
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    # Check if user exists
    cursor.execute("SELECT id, total_score, product_count FROM users WHERE name = ?", (user_name,))
    user = cursor.fetchone()
    
    if user:
        user_id, current_score, product_count = user
        new_score = current_score + sustainability_score
        new_count = product_count + 1
        
        cursor.execute(
            "UPDATE users SET total_score = ?, product_count = ? WHERE id = ?",
            (new_score, new_count, user_id)
        )
    else:
        cursor.execute(
            "INSERT INTO users (name, total_score, product_count) VALUES (?, ?, ?)",
            (user_name, sustainability_score, 1)
        )
        user_id = cursor.lastrowid
    
    # Record the product selection
    cursor.execute(
        """INSERT INTO user_products 
           (user_id, product_name, sustainability_score, carbon_footprint, water_usage, price, walmart_id) 
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (user_id, product_name, sustainability_score, carbon_footprint, water_usage, price, product_id)
    )
    
    conn.commit()
    conn.close()
    
    # Return updated leaderboard
    leaderboard = get_leaderboard()
    
    return {
        "success": True,
        "message": "Product selected successfully!",
        "new_score": sustainability_score,
        "leaderboard": leaderboard
    }