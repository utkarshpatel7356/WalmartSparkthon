from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from backend.app.database import init_db

router = APIRouter()
templates = Jinja2Templates(directory="frontend/templates")

# Ensure DB tables are created on startup
@router.on_event("startup")
def startup_event():
    init_db()

@router.get("/inventory", name="Inventory Dashboard")
def inventory_dashboard(request: Request):
    # If you had real data you’d pull it here; for now we just render the template
    return templates.TemplateResponse(
        "inventory_dashboard.html",
        {"request": request}
    )
