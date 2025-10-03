from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from PIL import Image
import io
import json
from typing import Optional
import base64
import requests

# import your settings
from ..core.config import settings


router = APIRouter()
templates = Jinja2Templates(directory="frontend/templates")

# Gemini AI configuration
GEMINI_API_KEY = 'YOUR_API_KEY'
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

@router.get("/image-analysis", response_class=HTMLResponse)
async def get_image_analysis_page(request: Request):
    return templates.TemplateResponse("image_analysis.html", {"request": request})

def encode_image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')

def call_gemini_api(image_base64: str, prompt: str) -> dict:
    """Call Gemini API directly using REST API"""
    headers = {
        'Content-Type': 'application/json',
    }
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_base64
                        }
                    }
                ]
            }
        ]
    }
    
    url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

@router.post("/analyze-image")
async def analyze_image(
    request: Request,
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    item_type: str = Form(...),
    product_age: str = Form(...),
    product_description: str = Form(...)
):
    try:
        # Handle image input (either file upload or URL)
        image_data = None
        if file:
            # Process uploaded file
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_data = image
        elif image_url:
            # For URL, we'd need to fetch it - simplified for this example
            raise HTTPException(status_code=400, detail="URL upload not implemented yet")
        else:
            raise HTTPException(status_code=400, detail="No image provided")
        
        # Create the prompt for Gemini
        prompt = f"""
        You're a circular‑economy expert with vision capabilities.  
        Analyze the attached image and the following metadata, then recommend **up to three** actions—Resell at Discount, Refurbish, Recycle Material, Donate to Charity, or Dispose.  
        
        Metadata:
        - Item Type: {item_type}
        - Product Age: {product_age}
        - Product Description: {product_description}
        
        For each action, output in JSON format:
        {{
            "recommendations": [
                {{
                    "action": "Action name",
                    "confidence": 0.85,
                    "rationale": "One‑sentence rationale citing visual cues, product description, and metadata factors"
                }}
            ],
            "overall_assessment": "Brief overall condition assessment considering the product description",
            "key_observations": ["observation1", "observation2", "observation3"]
        }}
        
        Please respond with valid JSON only.
        """
        
        # Convert image to base64
        image_base64 = encode_image_to_base64(image_data)
        
        # Call Gemini API
        try:
            gemini_response = call_gemini_api(image_base64, prompt)
            
            # Extract text from Gemini response
            if 'candidates' in gemini_response and len(gemini_response['candidates']) > 0:
                response_text = gemini_response['candidates'][0]['content']['parts'][0]['text']
            else:
                raise Exception("No response from Gemini API")
            
            # Parse the JSON response
            try:
                # Find JSON in the response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response_text[start_idx:end_idx]
                    analysis_result = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
            except (json.JSONDecodeError, ValueError) as e:
                # Fallback if JSON parsing fails
                analysis_result = {
                    "recommendations": [
                        {
                            "action": "Manual Review Required",
                            "confidence": 0.5,
                            "rationale": "AI analysis completed but response format needs manual review"
                        }
                    ],
                    "overall_assessment": response_text[:200] + "..." if len(response_text) > 200 else response_text,
                    "key_observations": ["Analysis completed", "Response format needs adjustment", "Manual review recommended"]
                }
        
        except Exception as api_error:
            # Fallback analysis if API fails
            analysis_result = {
                "recommendations": [
                    {
                        "action": "Resell at Discount" if product_age in ["Brand New", "Less than 6 months"] else "Recycle Material",
                        "confidence": 0.7,
                        "rationale": f"Based on {item_type} category and {product_age} age, this appears suitable for this action"
                    },
                    {
                        "action": "Donate to Charity",
                        "confidence": 0.6,
                        "rationale": "Alternative option considering the product category and estimated condition"
                    }
                ],
                "overall_assessment": f"Analysis for {item_type} item aged {product_age}. API service temporarily unavailable, showing fallback recommendations.",
                "key_observations": [
                    f"Product category: {item_type}",
                    f"Estimated age: {product_age}",
                    "Visual analysis pending - API service issue"
                ]
            }
        
        return templates.TemplateResponse("analysis_results.html", {
            "request": request,
            "analysis": analysis_result,
            "item_type": item_type,
            "product_age": product_age,
            "product_description": product_description,
            "filename": file.filename if file else "URL Image"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
