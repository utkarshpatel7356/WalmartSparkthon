from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from ml_models.catboost_price_delta_model import predict_price_delta

router = APIRouter()
templates = Jinja2Templates(directory="frontend/templates")

class PriceDeltaRequest(BaseModel):
    location: str
    effective_demand: float
    stock_level: float
    competitor_price: float
    weather: str
    product_name: str
    category: str
    base_price: float
    shelf_life_days: float

@router.get("/price-delta", name="Price Delta Form")
def get_form(request: Request):
    return templates.TemplateResponse("price_delta.html", {"request": request})

@router.post("/price-delta")
def post_form(request_data: PriceDeltaRequest):
    # Convert Pydantic model to dict
    input_data = request_data.dict()
    
    print(input_data)
    
    # Call your model
    pred = predict_price_delta(input_data) * 100  # if predict returns fraction
    
    # Return JSON response for the JavaScript to handle
    return {"prediction": round(pred, 2)}