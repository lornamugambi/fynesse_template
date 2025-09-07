from typing import List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
import osmnx as ox

def _bbox_from_latlon(latitude: float, longitude: float, box_size_km: float):
    box_size_degrees = box_size_km / 111.0
    north = latitude + box_size_degrees / 2
    south = latitude - box_size_degrees / 2
    west = longitude - box_size_degrees / 2
    east = longitude + box_size_degrees / 2
    return (west, south, east, north)

def plot_city_map(place_name: str, latitude: float, longitude: float, box_size_km: float = 2, poi_tags=None):
    if poi_tags is None:
        poi_tags = {"amenity": True}
    bbox = _bbox_from_latlon(latitude, longitude, box_size_km)
    graph = ox.graph_from_bbox(bbox)
    area = ox.geocode_to_gdf(place_name)
    nodes, edges = ox.graph_to_gdfs(graph)
    buildings = ox.features_from_bbox(bbox, tags={"building": True})
    pois = ox.features_from_bbox(bbox, tags=poi_tags)

    fig, ax = plt.subplots(figsize=(6, 6))
    area.plot(ax=ax, color="tan", alpha=0.5)
    buildings.plot(ax=ax, facecolor="gray", edgecolor="gray")
    edges.plot(ax=ax, linewidth=1, edgecolor="black", alpha=0.3)
    nodes.plot(ax=ax, color="black", markersize=1, alpha=0.3)
    pois.plot(ax=ax, color="green", markersize=5, alpha=1)
    west, south, east, north = bbox
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_title(place_name, fontsize=14)
    plt.show()

def get_osm_features(latitude: float, longitude: float, box_size_km: float = 2, tags=None) -> pd.DataFrame:
    if tags is None:
        tags = {"amenity": True}
    bbox = _bbox_from_latlon(latitude, longitude, box_size_km)
    gdf = ox.features_from_bbox(bbox, tags)
    return pd.DataFrame(gdf)

def get_feature_vector(latitude: float, longitude: float, box_size_km: float = 2, features: Optional[List[Tuple[str, Optional[str]]]] = None):
    if features is None:
        features = [
            ("amenity", None),
            ("amenity", "school"),
            ("amenity", "hospital"),
            ("amenity", "restaurant"),
            ("amenity", "cafe"),
            ("shop", None),
            ("tourism", None),
            ("tourism", "hotel"),
            ("tourism", "museum"),
            ("leisure", None),
            ("leisure", "park"),
            ("historic", None),
            ("amenity", "place_of_worship"),
        ]
    bbox = _bbox_from_latlon(latitude, longitude, box_size_km)
    tags = {}
    for key, _ in features:
        if key not in tags:
            tags[key] = True
    try:
        pois = ox.features_from_bbox(bbox, tags)
        if len(pois) > 0:
            df = pd.DataFrame(pois)
        else:
            df = pd.DataFrame(columns=[key for key, _ in features])
    except Exception:
        return {f"{key}:{value}" if value else key: 0 for key, value in features}

    counts = {}
    for key, value in features:
        if key in df.columns:
            if value:
                counts[f"{key}:{value}"] = (df[key] == value).sum()
            else:
                counts[key] = df[key].notnull().sum()
        else:
            counts[f"{key}:{value}" if value else key] = 0
    return counts

def visualize_feature_space(X, y, method: str = 'PCA'):
    if method.upper() != 'PCA':
        raise ValueError("Only PCA is implemented in this minimal version.")
    from sklearn.decomposition import PCA
    import numpy as np
    pca = PCA(n_components=2)
    X_proj = pca.fit_transform(X)
    labels = list(sorted(set(y)))
    colors = ["green", "blue", "orange", "purple"]
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    for idx, label in enumerate(labels):
        mask = (y == label) if not isinstance(y, list) else np.array(y) == label
        plt.scatter(X_proj[mask, 0], X_proj[mask, 1], label=label, color=colors[idx % len(colors)], s=100, alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("2D projection of feature vectors")
    plt.legend()
    plt.show()
