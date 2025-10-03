from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from ml_models.acquisitions import predict_acquisition_quantity

router = APIRouter()
templates = Jinja2Templates(directory="frontend/templates")

class AcquisitionRequest(BaseModel):
    demand: float
    stock_level: float
    shelf_life_days: float
    price: float
    spoilage_cost: float

@router.get("/acquisition", name="Acquisition Form")
def get_form(request: Request):
    return templates.TemplateResponse("acquisition.html", {"request": request})

@router.post("/acquisition")
def post_form(request_data: AcquisitionRequest):
    input_data = request_data.dict()
    #consol.log(input_data)
    quantity = predict_acquisition_quantity(input_data)
    return {"prediction": quantity}
