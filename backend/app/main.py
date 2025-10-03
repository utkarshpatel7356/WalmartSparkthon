from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from backend.app.routers import home
from backend.app.routers import inventory
from backend.app.routers import price_delta
from backend.app.routers import acquisition
from backend.app.routers import image_analysis
from backend.app.routers import trends
from backend.app.routers import trnf_back
from backend.app.routers import mapft  
from backend.app.routers import prediction
from backend.app.routers import products  # Add this import

app = FastAPI(
    title="Walmart AI Starter",
    description="Enhanced AI platform with Supply-Demand Mapping",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Include all routers
app.include_router(home.router)
app.include_router(inventory.router)
app.include_router(price_delta.router)
app.include_router(acquisition.router)
app.include_router(image_analysis.router)
app.include_router(trends.router)
app.include_router(trnf_back.router)
app.include_router(mapft.router)
app.include_router(prediction.router)
app.include_router(products.router)  # Add this line

# Serve static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Optional: Add middleware for CORS if needed
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional: Add a root endpoint that shows available endpoints
@app.get("/")
async def root():
    return {
        "message": "Walmart AI Starter API",
        "endpoints": {
            "docs": "/docs",
            "map_dashboard": "/map",
            "api_data": "/api/data",
            "add_giver": "/api/givers",
            "add_needy": "/api/needies"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "walmart-ai-starter"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)