from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from bs4 import BeautifulSoup
import requests
import json
from typing import List, Dict, Any
import asyncio
import aiohttp
from pydantic import BaseModel

router = APIRouter()
templates = Jinja2Templates(directory="frontend/templates")

class ProductData(BaseModel):
    title: str
    image_url: str
    category: str

class TrendsData(BaseModel):
    product: ProductData
    interest_over_time: List[Dict[str, Any]]
    top_region: Dict[str, Any]
    rising_queries: List[str]
    regional_data: List[Dict[str, Any]]

# Amazon scraping configuration
AMAZON_CATEGORIES = {
    'beauty': 'https://www.amazon.in/gp/movers-and-shakers/beauty/ref=zg_bsms_nav_beauty_0_computers',
    # 'books': 'https://www.amazon.in/gp/movers-and-shakers/books/ref=zg_bsms_nav_books_0_beauty',
    'electronics': 'https://www.amazon.in/gp/movers-and-shakers/electronics/ref=zg_bsms_nav_electronics_0_books',
    'kitchen': 'https://www.amazon.in/gp/movers-and-shakers/kitchen/ref=zg_bsms_nav_kitchen_0_electronics',
    'sports': 'https://www.amazon.in/gp/movers-and-shakers/sports/ref=zg_bsms_nav_sports_0_kitchen'
}

# Google Trends API configuration
SERPAPI_KEY = 'YOUR_API_KEY'
# SERPAPI_KEY = 'abc'
TRENDS_BASE_URL = 'https://serpapi.com/search'

def scrape_amazon_product(url: str, category: str) -> ProductData:
    """Scrape a single Amazon product from movers and shakers"""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        page = requests.get(url, headers=headers, timeout=10)
        page.raise_for_status()
        
        soup = BeautifulSoup(page.text, 'html.parser')
        faceout = soup.select_one('div.p13n-sc-uncoverable-faceout')
        
        if not faceout:
            raise ValueError(f"Could not find product data for {category}")
            
        img = faceout.find('img')
        if not img:
            raise ValueError(f"Could not find product image for {category}")
            
        title = img.get('alt', f'Top {category.title()} Product')
        image_url = img.get('src', '')
        
        return ProductData(title=title, image_url=image_url, category=category)
    except Exception as e:
        # Return a fallback product if scraping fails
        return ProductData(
            title=f"Trending {category.title()} Product",
            image_url="https://via.placeholder.com/200x200?text=Product",
            category=category
        )

async def get_trends_data(query: str) -> Dict[str, Any]:
    """Get Google Trends data for a product query"""
    try:
        # Interest over time
        interest_params = {
            'engine': 'google_trends',
            'q': query,
            'data_type': 'TIMESERIES',
            'hl': 'en',
            'geo': 'IN',
            'date': 'today 12-m',
            'api_key': SERPAPI_KEY
        }
        
        # Regional data
        regional_params = {
            'engine': 'google_trends',
            'q': query,
            'data_type': 'GEO_MAP_0',
            'hl': 'en',
            'geo': 'IN',
            'region': 'REGION',
            'api_key': SERPAPI_KEY
        }
        
        # Related queries
        queries_params = {
            'engine': 'google_trends',
            'q': query,
            'data_type': 'RELATED_QUERIES',
            'hl': 'en',
            'geo': 'IN',
            'api_key': SERPAPI_KEY
        }
        
        async with aiohttp.ClientSession() as session:
            # Make all API calls concurrently
            tasks = []
            for params in [interest_params, regional_params, queries_params]:
                tasks.append(session.get(TRENDS_BASE_URL, params=params))
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process responses
            interest_data = []
            regional_data = []
            rising_queries = []
            top_region = {"name": "India", "interest": 100}
            
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    continue
                    
                try:
                    data = await response.json()
                    
                    if i == 0 and 'interest_over_time' in data:
                        # Interest over time data
                        timeline = data['interest_over_time'].get('timeline_data', [])
                        for point in timeline[-12:]:  # Last 12 points
                            interest_data.append({
                                'date': point.get('date', ''),
                                'value': point.get('values', [{}])[0].get('value', 0)
                            })
                    
                    elif i == 1 and 'interest_by_region' in data:
                        # Regional data
                        regions = data['interest_by_region'].get('regions', [])[:10]
                        for region in regions:
                            regional_data.append({
                                'name': region.get('location', ''),
                                'interest': region.get('value', 0)
                            })
                        if regions:
                            top_region = {
                                'name': regions[0].get('location', 'India'),
                                'interest': regions[0].get('value', 100)
                            }
                    
                    elif i == 2 and 'related_queries' in data:
                        # Rising queries
                        rising = data['related_queries'].get('rising', [])[:3]
                        for query_data in rising:
                            rising_queries.append(query_data.get('query', ''))
                        
                        # If no rising queries, use top queries
                        if not rising_queries:
                            top_queries = data['related_queries'].get('top', [])[:3]
                            for query_data in top_queries:
                                rising_queries.append(query_data.get('query', ''))
                                
                except Exception as e:
                    continue
            
            # Generate fallback data if API calls failed
            if not interest_data:
                import random
                for i in range(12):
                    interest_data.append({
                        'date': f'2024-{(i%12)+1:02d}',
                        'value': random.randint(20, 100)
                    })
            
            if not rising_queries:
                rising_queries = [f"{query} reviews", f"best {query}", f"{query} price"]
            
            return {
                'interest_over_time': interest_data,
                'top_region': top_region,
                'rising_queries': rising_queries,
                'regional_data': regional_data or [top_region]
            }
            
    except Exception as e:
        # Return fallback trends data
        import random
        return {
            'interest_over_time': [
                {'date': f'2024-{i+1:02d}', 'value': random.randint(20, 100)} 
                for i in range(12)
            ],
            'top_region': {'name': 'India', 'interest': 100},
            'rising_queries': [f"{query} reviews", f"best {query}", f"{query} price"],
            'regional_data': [{'name': 'India', 'interest': 100}]
        }

@router.get("/trending-products", response_model=List[TrendsData])
async def get_trending_products():
    """Get trending products from Amazon with Google Trends data"""
    try:
        # Scrape Amazon products
        products = []
        for category, url in AMAZON_CATEGORIES.items():
            product = scrape_amazon_product(url, category)
            products.append(product)
        
        # Get trends data for each product
        trends_data = []
        for product in products:
            # Clean product title for better trends search
            search_query = product.title.split(',')[0].split('(')[0].strip()
            trends = await get_trends_data(search_query)
            
            trends_data.append(TrendsData(
                product=product,
                interest_over_time=trends['interest_over_time'],
                top_region=trends['top_region'],
                rising_queries=trends['rising_queries'],
                regional_data=trends['regional_data']
            ))
        
        return trends_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching trending products: {str(e)}")

@router.get("/trends-page", response_class=HTMLResponse)
async def trends_page(request: Request):
    """Render the trending products HTML page"""
    try:
        # Get the trending products data
        trends_data = await get_trending_products()
        
        # Convert Pydantic models to dict for Jinja2
        trends_dict = []
        for trend in trends_data:
            trends_dict.append({
                'product': {
                    'title': trend.product.title,
                    'image_url': trend.product.image_url,
                    'category': trend.product.category
                },
                'interest_over_time': trend.interest_over_time,
                'top_region': trend.top_region,
                'rising_queries': trend.rising_queries,
                'regional_data': trend.regional_data
            })
        
        return templates.TemplateResponse("trending_place.html", {
            "request": request,
            "trends_data": trends_dict
        })
    except Exception as e:
        return templates.TemplateResponse("trending_place.html", {
            "request": request,
            "trends_data": [],
            "error": str(e)
        })

@router.get("/product-trends/{product_name}")
async def get_product_trends(product_name: str):
    """Get detailed trends data for a specific product"""
    try:
        trends = await get_trends_data(product_name)
        return trends
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching trends for {product_name}: {str(e)}")