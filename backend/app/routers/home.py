from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from ml_models.dummy_model import predict_dummy
from backend.app.database import SessionLocal, init_db

router = APIRouter()
templates = Jinja2Templates(directory="frontend/templates")

# ensure DB & tables exist before first request
@router.on_event("startup")
def startup_event():
    init_db()

@router.get("/")
def get_dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

