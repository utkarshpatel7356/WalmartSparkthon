from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
import httpx
import logging
import uuid
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Configure templates
# templates = Jinja2Templates(directory="frontend/templates")

# Your OpenRouteService API key
ORS_API_KEY = "YOUR_API_KEY"

# In-memory storage
donations = {}
needs = {}
matches = {}

class DonationRequest(BaseModel):
    item: str
    quantity: int
    location: List[float]  # [lng, lat]
    donor_name: Optional[str] = "Anonymous"

class NeedRequest(BaseModel):
    item: str
    quantity: int
    location: List[float]  # [lng, lat]
    receiver_name: Optional[str] = "Anonymous"

class RouteRequest(BaseModel):
    coordinates: List[List[float]]  # [[lng, lat], [lng, lat]]
    profile: str = "driving-car"

# @router.get("/map", response_class=HTMLResponse)
# async def serve_map_page(request: Request):
#     """Serve the map HTML page"""
#     return templates.TemplateResponse("mapht.html", {"request": request})

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
        raise HTTPException(status_code=404, detail="Map template not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading map page: {str(e)}")

@router.post("/api/donate")
async def add_donation(request: DonationRequest):
    """Add a new donation"""
    try:
        donation_id = str(uuid.uuid4())
        donations[donation_id] = {
            "id": donation_id,
            "item": request.item,
            "quantity": request.quantity,
            "location": request.location,
            "donor_name": request.donor_name,
            "timestamp": datetime.now().isoformat(),
            "status": "available"
        }
        
        # Check for matches
        new_matches = find_matches()
        
        return {
            "success": True,
            "donation_id": donation_id,
            "matches": new_matches
        }
    except Exception as e:
        logger.error(f"Error adding donation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/need")
async def add_need(request: NeedRequest):
    """Add a new need"""
    try:
        need_id = str(uuid.uuid4())
        needs[need_id] = {
            "id": need_id,
            "item": request.item,
            "quantity": request.quantity,
            "location": request.location,
            "receiver_name": request.receiver_name,
            "timestamp": datetime.now().isoformat(),
            "status": "needed"
        }
        
        # Check for matches
        new_matches = find_matches()
        
        return {
            "success": True,
            "need_id": need_id,
            "matches": new_matches
        }
    except Exception as e:
        logger.error(f"Error adding need: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def find_matches():
    """Find matches between donations and needs"""
    new_matches = []
    
    for donation_id, donation in donations.items():
        if donation["status"] != "available":
            continue
            
        for need_id, need in needs.items():
            if need["status"] != "needed":
                continue
                
            # Check if item matches and quantity is sufficient
            if (donation["item"].lower() == need["item"].lower() and 
                donation["quantity"] >= need["quantity"]):
                
                match_id = str(uuid.uuid4())
                match_data = {
                    "id": match_id,
                    "donation_id": donation_id,
                    "need_id": need_id,
                    "item": donation["item"],
                    "quantity": need["quantity"],
                    "donor_name": donation["donor_name"],
                    "receiver_name": need["receiver_name"],
                    "donor_location": donation["location"],
                    "receiver_location": need["location"],
                    "timestamp": datetime.now().isoformat(),
                    "status": "pending"
                }
                
                matches[match_id] = match_data
                new_matches.append(match_data)
    
    return new_matches

@router.get("/api/matches")
async def get_matches():
    """Get all pending matches"""
    pending_matches = [match for match in matches.values() if match["status"] == "pending"]
    return {"matches": pending_matches}

@router.post("/api/accept-match/{match_id}")
async def accept_match(match_id: str):
    """Accept a match and update statuses"""
    try:
        if match_id not in matches:
            raise HTTPException(status_code=404, detail="Match not found")
        
        match = matches[match_id]
        match["status"] = "accepted"
        
        # Update donation and need statuses
        donation_id = match["donation_id"]
        need_id = match["need_id"]
        
        if donation_id in donations:
            donations[donation_id]["status"] = "matched"
        if need_id in needs:
            needs[need_id]["status"] = "matched"
        
        return {"success": True, "match": match}
    except Exception as e:
        logger.error(f"Error accepting match: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
        start_coord = f"{request.coordinates[0][0]},{request.coordinates[0][1]}"
        end_coord = f"{request.coordinates[1][0]},{request.coordinates[1][1]}"
        
        # Construct URL with query parameters
        ors_url = f"https://api.openrouteservice.org/v2/directions/{request.profile}"
        
        params = {
            "api_key": ORS_API_KEY,
            "start": start_coord,
            "end": end_coord
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
                    }
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
    return {"status": "healthy", "service": "Map Features Route Planner"}

@router.get("/api/donations")
async def get_donations():
    """Get all donations"""
    return {"donations": list(donations.values())}

@router.get("/api/needs")
async def get_needs():
    """Get all needs"""
    return {"needs": list(needs.values())}

@router.get("/api/stats")
async def get_stats():
    """Get statistics about donations, needs, and matches"""
    return {
        "total_donations": len(donations),
        "total_needs": len(needs),
        "total_matches": len(matches),
        "pending_matches": len([m for m in matches.values() if m["status"] == "pending"]),
        "accepted_matches": len([m for m in matches.values() if m["status"] == "accepted"]),
        "available_donations": len([d for d in donations.values() if d["status"] == "available"]),
        "unfulfilled_needs": len([n for n in needs.values() if n["status"] == "needed"])
    }

# Additional endpoints for better integration with your main app
@router.get("/api/data")
async def get_all_data():
    """Get all data for the map dashboard"""
    return {
        "donations": list(donations.values()),
        "needs": list(needs.values()),
        "matches": list(matches.values()),
        "stats": {
            "total_donations": len(donations),
            "total_needs": len(needs),
            "total_matches": len(matches),
            "pending_matches": len([m for m in matches.values() if m["status"] == "pending"]),
            "accepted_matches": len([m for m in matches.values() if m["status"] == "accepted"])
        }
    }

# Walmart-specific endpoints to match your main app structure
@router.post("/api/givers")
async def add_giver(request: DonationRequest):
    """Add a new giver (donation) - alias for add_donation"""
    return await add_donation(request)

@router.post("/api/needies")
async def add_needy(request: NeedRequest):
    """Add a new needy person - alias for add_need"""
    return await add_need(request)