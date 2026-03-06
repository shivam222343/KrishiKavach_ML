import requests
from bs4 import BeautifulSoup
import json
import re

def get_osm_facilities(lat, lon, radius_km=50, city=None):
    """
    Query Overpass API for agricultural facilities (mills, warehouses, etc.)
    """
    radius_meters = radius_km * 1000
    
    # Improved Query: searching for more specific agricultural and industrial tags
    query = f"""
    [out:json][timeout:8];
    (
      node["man_made"="silo"](around:{radius_meters},{lat},{lon});
      node["industrial"~"factory|mill"](around:{radius_meters},{lat},{lon});
      way["industrial"~"factory|mill"](around:{radius_meters},{lat},{lon});
      node["landuse"="industrial"]["name"~"Mill|Warehouse|Storage|Agri|Cotton"](around:{radius_meters},{lat},{lon});
      way["landuse"="industrial"]["name"~"Mill|Warehouse|Storage|Agri|Cotton"](around:{radius_meters},{lat},{lon});
      node["shop"="agriculture"](around:{radius_meters},{lat},{lon});
    );
    out body;
    >;
    out skel qt;
    """
    
    # Mapping facility types to relevant images
    IMAGE_MAP = {
        "Ginning Mill": "https://images.unsplash.com/photo-1590633717560-49651582e3b2?q=80&w=400&h=300&auto=format&fit=crop",
        "Warehouse": "https://images.unsplash.com/photo-1586528116311-ad8dd3c8310d?q=80&w=400&h=300&auto=format&fit=crop",
        "Processing Center": "https://images.unsplash.com/photo-1541888946425-d81bb19480c5?q=80&w=400&h=300&auto=format&fit=crop",
        "Cold Storage": "https://images.unsplash.com/photo-1580674684081-7617fbf3d745?q=80&w=400&h=300&auto=format&fit=crop"
    }

    url = "https://overpass-api.de/api/interpreter"
    try:
        response = requests.post(url, data={'data': query}, timeout=10)
        if response.status_code != 200:
            print(f"OSM API Error: Status {response.status_code}")
            return []
        
        # Check if response is actually JSON
        if "application/json" not in response.headers.get("Content-Type", "").lower():
            print(f"OSM API Error: Non-JSON response received")
            return []

        data = response.json()
        
        facilities = []
        for element in data.get('elements', []):
            if 'tags' in element:
                name = element['tags'].get('name', 'Agricultural Facility')
                # Determine specific type from name or tags
                facility_type = "Processing Center"
                if "ginning" in name.lower() or "spinning" in name.lower(): facility_type = "Ginning Mill"
                elif "warehouse" in name.lower() or "storage" in name.lower(): facility_type = "Warehouse"
                elif "cold storage" in name.lower(): facility_type = "Cold Storage"
                
                lat_val = element.get('lat')
                lon_val = element.get('lon')
                
                if not lat_val and 'center' in element:
                    lat_val = element['center']['lat']
                    lon_val = element['center']['lon']
                
                if lat_val and lon_val:
                    facilities.append({
                        "id": f"osm_{element['id']}",
                        "name": name,
                        "type": facility_type,
                        "location": [lon_val, lat_val],
                        "city": city or element['tags'].get('addr:city', 'Local Area'),
                        "contact": element['tags'].get('phone', element['tags'].get('contact:phone', '+91 (OSM Verified)')),
                        "source": "OpenStreetMap",
                        "image": IMAGE_MAP.get(facility_type, IMAGE_MAP["Processing Center"])
                    })
        return facilities
    except Exception as e:
        print(f"OSM Error: {e}")
        return []

def get_hybrid_facilities(lat, lon, radius_km=50, city=None):
    # Try fetching from OSM
    osm_data = get_osm_facilities(lat, lon, radius_km, city)
    return osm_data
