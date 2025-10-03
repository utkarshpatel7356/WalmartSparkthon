from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid
import logging
import os
import httpx

router = APIRouter()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# Data storage (in production, use a database)
givers = []
needies = []

class Giver(BaseModel):
    product: str
    capacity: int
    lat: float
    lng: float

class Needy(BaseModel):
    product: str
    need: int
    lat: float
    lng: float

class RouteRequest(BaseModel):
    coordinates: List[List[float]]
    profile: str = "driving-car"


# Your OpenRouteService API key
ORS_API_KEY = "YOUR_API_KEY"

@router.get("/map", response_class=HTMLResponse)
async def get_map_page():
    """Serve the supply-demand map dashboard"""
    try:
        # Get the path to the HTML template
        template_path = os.path.join("frontend", "templates", "mapht.html")
        
        # Read the HTML file
        with open(template_path, "r", encoding="utf-8") as file:
            html_content = file.read()
        
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        print('error in loading map')
        raise HTTPException(status_code=404, detail="Map template not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading map page: {str(e)}")

@router.post("/api/givers")
async def add_giver(giver: Giver):
    """Add a new giver (supplier) to the system"""
    try:
        # Validate input data
        if giver.capacity <= 0:
            raise HTTPException(status_code=400, detail="Capacity must be greater than 0")
        
        if not (-90 <= giver.lat <= 90):
            raise HTTPException(status_code=400, detail="Invalid latitude")
        
        if not (-180 <= giver.lng <= 180):
            raise HTTPException(status_code=400, detail="Invalid longitude")
        
        # Create giver data
        giver_data = {
            "id": str(uuid.uuid4()),
            "product": giver.product,
            "capacity": giver.capacity,
            "lat": giver.lat,
            "lng": giver.lng,
            "created_at": str(uuid.uuid1().time)  # Simple timestamp
        }
        
        givers.append(giver_data)
        
        return {
            "success": True, 
            "message": "Giver registered successfully",
            "giver": giver_data
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding giver: {str(e)}")

@router.post("/api/needies")
async def add_needy(needy: Needy):
    """Add a new needy (buyer) to the system"""
    try:
        # Validate input data
        if needy.need <= 0:
            raise HTTPException(status_code=400, detail="Need must be greater than 0")
        
        if not (-90 <= needy.lat <= 90):
            raise HTTPException(status_code=400, detail="Invalid latitude")
        
        if not (-180 <= needy.lng <= 180):
            raise HTTPException(status_code=400, detail="Invalid longitude")
        
        # Create needy data
        needy_data = {
            "id": str(uuid.uuid4()),
            "product": needy.product,
            "need": needy.need,
            "lat": needy.lat,
            "lng": needy.lng,
            "created_at": str(uuid.uuid1().time)  # Simple timestamp
        }
        
        needies.append(needy_data)
        
        return {
            "success": True,
            "message": "Needy registered successfully", 
            "needy": needy_data
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding needy: {str(e)}")

@router.get("/api/data")
async def get_all_data():
    """Get all givers, needies, and calculate matches"""
    try:
        # Calculate matches between givers and needies
        matches = []
        
        for giver in givers:
            for needy in needies:
                # Match products
                if giver["product"] == needy["product"]:
                    # Calculate coverage percentage
                    coverage = min(giver["capacity"] / needy["need"], 1.0)
                    match_type = "full" if coverage >= 1.0 else "partial"
                    
                    # Calculate distance (simple Euclidean distance for demo)
                    lat_diff = giver["lat"] - needy["lat"]
                    lng_diff = giver["lng"] - needy["lng"]
                    distance = (lat_diff**2 + lng_diff**2)**0.5
                    
                    match_data = {
                        "giver_id": giver["id"],
                        "needy_id": needy["id"],
                        "product": giver["product"],
                        "coverage": coverage,
                        "type": match_type,
                        "giver_lat": giver["lat"],
                        "giver_lng": giver["lng"],
                        "needy_lat": needy["lat"],
                        "needy_lng": needy["lng"],
                        "distance": round(distance, 4),
                        "giver_capacity": giver["capacity"],
                        "needy_need": needy["need"]
                    }
                    
                    matches.append(match_data)
        
        # Sort matches by coverage (full matches first) and then by distance
        matches.sort(key=lambda x: (-x["coverage"], x["distance"]))
        
        return {
            "givers": givers,
            "needies": needies,
            "matches": matches,
            "summary": {
                "total_givers": len(givers),
                "total_needies": len(needies),
                "total_matches": len(matches),
                "full_matches": len([m for m in matches if m["type"] == "full"]),
                "partial_matches": len([m for m in matches if m["type"] == "partial"])
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving data: {str(e)}")

@router.get("/api/givers")
async def get_givers():
    """Get all givers"""
    try:
        return {
            "success": True,
            "givers": givers,
            "count": len(givers)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving givers: {str(e)}")

@router.get("/api/needies")
async def get_needies():
    """Get all needies"""
    try:
        return {
            "success": True,
            "needies": needies,
            "count": len(needies)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving needies: {str(e)}")

@router.delete("/api/givers/{giver_id}")
async def delete_giver(giver_id: str):
    """Delete a giver by ID"""
    try:
        global givers
        original_count = len(givers)
        givers = [g for g in givers if g["id"] != giver_id]
        
        if len(givers) == original_count:
            raise HTTPException(status_code=404, detail="Giver not found")
        
        return {
            "success": True,
            "message": "Giver deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting giver: {str(e)}")

@router.delete("/api/needies/{needy_id}")
async def delete_needy(needy_id: str):
    """Delete a needy by ID"""
    try:
        global needies
        original_count = len(needies)
        needies = [n for n in needies if n["id"] != needy_id]
        
        if len(needies) == original_count:
            raise HTTPException(status_code=404, detail="Needy not found")
        
        return {
            "success": True,
            "message": "Needy deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting needy: {str(e)}")

@router.get("/api/matches")
async def get_matches():
    """Get only the matches without full data"""
    try:
        matches = []
        
        for giver in givers:
            for needy in needies:
                if giver["product"] == needy["product"]:
                    coverage = min(giver["capacity"] / needy["need"], 1.0)
                    match_type = "full" if coverage >= 1.0 else "partial"
                    
                    # Calculate distance
                    lat_diff = giver["lat"] - needy["lat"]
                    lng_diff = giver["lng"] - needy["lng"]
                    distance = (lat_diff**2 + lng_diff**2)**0.5
                    
                    matches.append({
                        "giver_id": giver["id"],
                        "needy_id": needy["id"],
                        "product": giver["product"],
                        "coverage": coverage,
                        "type": match_type,
                        "distance": round(distance, 4)
                    })
        
        # Sort by coverage and distance
        matches.sort(key=lambda x: (-x["coverage"], x["distance"]))
        
        return {
            "success": True,
            "matches": matches,
            "count": len(matches)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving matches: {str(e)}")

@router.delete("/api/clear")
async def clear_all_data():
    """Clear all data (for testing purposes)"""
    try:
        global givers, needies
        givers.clear()
        needies.clear()
        
        return {
            "success": True,
            "message": "All data cleared successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing data: {str(e)}")

@router.get("/api/stats")
async def get_statistics():
    """Get system statistics"""
    try:
        # Product distribution
        giver_products = {}
        needy_products = {}
        
        for giver in givers:
            product = giver["product"]
            if product in giver_products:
                giver_products[product] += giver["capacity"]
            else:
                giver_products[product] = giver["capacity"]
        
        for needy in needies:
            product = needy["product"]
            if product in needy_products:
                needy_products[product] += needy["need"]
            else:
                needy_products[product] = needy["need"]
        
        # Calculate matches
        matches = []
        for giver in givers:
            for needy in needies:
                if giver["product"] == needy["product"]:
                    coverage = min(giver["capacity"] / needy["need"], 1.0)
                    matches.append({"coverage": coverage, "product": giver["product"]})
        
        return {
            "success": True,
            "statistics": {
                "total_givers": len(givers),
                "total_needies": len(needies),
                "total_matches": len(matches),
                "full_matches": len([m for m in matches if m["coverage"] >= 1.0]),
                "partial_matches": len([m for m in matches if m["coverage"] < 1.0]),
                "giver_products": giver_products,
                "needy_products": needy_products,
                "total_supply": sum(giver_products.values()),
                "total_demand": sum(needy_products.values())
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving statistics: {str(e)}")
    
@router.post("/api/route")
async def get_route(request: RouteRequest):
    """Get route from OpenRouteService API using GET request with query parameters"""
    try:
        # Validate coordinates
        if len(request.coordinates) != 2:
            raise HTTPException(
                status_code=400, 
                detail="Exactly 2 coordinates are required (start and end)"
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
        
        # Format coordinates for the API (lng,lat format)
        start_coord = f"{request.coordinates[0][1]},{request.coordinates[0][0]}"
        end_coord = f"{request.coordinates[1][1]},{request.coordinates[1][0]}"
        
        # Construct URL with query parameters
        ors_url = f"https://api.openrouteservice.org/v2/directions/{request.profile}"

        logger.info(f"Start: {start_coord}, End: {end_coord}")
        
        params = {
            "api_key": ORS_API_KEY,
            "start": start_coord,
            "end": end_coord,
        }
        
        logger.info(f"Making request to ORS: {ors_url}")
        logger.info(f"Parameters: {params}")
        
        # Make request to OpenRouteService
        async with httpx.AsyncClient() as client:
            response = await client.get(
                ors_url,
                params=params,
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
            
            if "features" not in data or not data["features"]:
                raise HTTPException(
                    status_code=404,
                    detail="No route found"
                )
            
            route = data["features"][0]
            
            # Extract route information
            route_properties = route.get("properties", {})
            route_segments = route_properties.get("segments", [])
            
            # Calculate total distance and duration
            total_distance = 0
            total_duration = 0
            
            if route_segments:
                for segment in route_segments:
                    total_distance += segment.get("distance", 0)
                    total_duration += segment.get("duration", 0)
            else:
                # Fallback if segments not available
                total_distance = route_properties.get("distance", 0)
                total_duration = route_properties.get("duration", 0)
            
            # Convert geometry to GeoJSON format if needed
            geometry = route.get("geometry", {})
            
            # Return processed route data
            return {
                "success": True,
                "route": {
                    "distance": total_distance,
                    "duration": total_duration,
                    "geometry": geometry,
                    "summary": {
                        "distance": total_distance,
                        "duration": total_duration
                    },
                    "full_response": data  # Include full response for debugging
                }
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "OpenRouteService Route Planner"}
