from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import httpx
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OpenRouteService API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your OpenRouteService API key
ORS_API_KEY = "YOUR_API_KEY"
ORS_BASE_URL = "https://api.openrouteservice.org/v2/directions"

class RouteRequest(BaseModel):
    coordinates: List[List[float]]  # [[lng, lat], [lng, lat], ...]
    profile: str = "driving-car"  # driving-car, foot-walking, cycling-regular, etc.

class RouteResponse(BaseModel):
    distance: float
    duration: float
    geometry: dict
    summary: dict

@app.get("/")
async def root():
    return {"message": "OpenRouteService API Backend"}

@app.post("/api/route", response_model=dict)
async def get_route(request: RouteRequest):
    """
    Get route from OpenRouteService API
    """
    try:
        # Validate coordinates
        if len(request.coordinates) < 2:
            raise HTTPException(
                status_code=400, 
                detail="At least 2 coordinates are required"
            )
        
        # Validate coordinate format
        for coord in request.coordinates:
            if len(coord) != 2:
                raise HTTPException(
                    status_code=400,
                    detail="Each coordinate must be [longitude, latitude]"
                )
            if not (-180 <= coord[0] <= 180) or not (-90 <= coord[1] <= 90):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid coordinate values"
                )
        
        # Prepare request to OpenRouteService
        ors_url = f"{ORS_BASE_URL}/{request.profile}"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": ORS_API_KEY
        }
        
        payload = {
            "coordinates": request.coordinates
        }
        
        logger.info(f"Making request to ORS: {ors_url}")
        logger.info(f"Payload: {payload}")
        
        # Make request to OpenRouteService
        async with httpx.AsyncClient() as client:
            response = await client.post(
                ors_url,
                json=payload,
                headers=headers,
                timeout=30.0
            )
            
            logger.info(f"ORS Response status: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"ORS Error: {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"OpenRouteService error: {response.text}"
                )
            
            data = response.json()
            
            if "routes" not in data or not data["routes"]:
                raise HTTPException(
                    status_code=404,
                    detail="No route found"
                )
            
            route = data["routes"][0]
            
            # Return processed route data
            return {
                "success": True,
                "route": {
                    "distance": route["summary"]["distance"],
                    "duration": route["summary"]["duration"],
                    "geometry": route["geometry"],
                    "summary": route["summary"]
                },
                "original_response": data  # Include full response for debugging
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "OpenRouteService API"}

# Test endpoint with sample coordinates
@app.get("/api/test-route")
async def test_route():
    """Test route with sample coordinates"""
    sample_request = RouteRequest(
        coordinates=[
            [8.681495, 49.41461],
            [8.686507, 49.41943],
            [8.687872, 49.420318]
        ]
    )
    
    return await get_route(sample_request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)