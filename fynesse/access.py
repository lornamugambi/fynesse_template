import pandas as pd
import osmnx as ox

def _bbox_from_latlon(latitude: float, longitude: float, box_size_km: float):
    box_size_degrees = box_size_km / 111.0
    north = latitude + box_size_degrees / 2
    south = latitude - box_size_degrees / 2
    west = longitude - box_size_degrees / 2
    east = longitude + box_size_degrees / 2
    return (west, south, east, north)

def get_osm_datapoints(latitude: float, longitude: float, box_size_km: float = 2, poi_tags=None):
    if poi_tags is None:
        poi_tags = {"amenity": True}
    bbox = _bbox_from_latlon(latitude, longitude, box_size_km)
    pois = ox.features_from_bbox(bbox, poi_tags)
    return pois
