#%% Preprocessing
print("x")
import geopandas as gpd
import os
os.chdir("C:/Users/wojci/Desktop/IV_rok/data/data")

granica = gpd.read_file("Preprocess/granica.gpkg")

def preprocess(obj):
    obj = obj.to_crs("EPSG:2180")
    obj = gpd.clip(obj, granica)
    return obj

# POI- preprocess
POI = gpd.read_file("wielkopolskie-latest-free.shp/gis_osm_pois_free_1.shp")
POI = preprocess(POI)
POI.fclass.unique()
POI = POI[POI['fclass'].isin(["bakery", "supermarket", "food_court"])]

axes = granica.plot(facecolor = "grey")
POI.plot(ax = axes)

POI.to_file("Preprocess/POIS.gpkg", driver = "GPKG")

# Preprocess roads
roads = gpd.read_file("wielkopolskie-latest-free.shp/gis_osm_roads_free_1.shp")
roads = preprocess(roads)
roads = roads[roads['fclass'].isin(["footway", "path", "cycleway"])]
roads.plot()

roads.to_file("Preprocess/roads.gpkg", driver = "GPKG")

##### OBLICZENIE MAP ODLEGLOSCI I RASTERYZACJA ZOSTALA DOKONANA W R - plik Jankowiak_Skrypt

#%% Import narzedzi
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import rasterio as rio
import numpy as np
import pandas as pd
import random

#%% Wczytanie danych
shops = rio.open("shops.tif")
drogi = rio.open("drogi.tif")
granica = gpd.read_file("Preprocess/granica.gpkg")
ludnosc = gpd.read_file("ludnosc_poznan_250.gpkg")
a = granica.bounds

minx = int(a['minx'].min())
maxx = int(a['maxx'].max())
miny = int(a['miny'].min())
maxy = int(a['maxy'].max())


#%% Generowanie lokalizacji sklepów
sklepy = []
sillShop = 600
sillRoad = 50

def replace_nan(value):
    if pd.isna(value):
        return "> {}".format(round(shops.read().max()), 2)
    else:
        return value

def sprawdz_przeciecie(strefy, nowa_strefa):
    for strefa in strefy:
        if strefa.intersects(nowa_strefa):
            return True
    return False


def probkiRoad(x, y, roadValue):
    if roadValue[0][0] > sillRoad:
        return True
    return False


def probkiShop(x, y, shopValue):
    if shopValue[0][0] < sillShop:
        return True
    return False

def ludzie(prog):
    if prog < 1000:
        return True
    return False
    

def generate_shops(n):
    sklepy = []
    while len(sklepy) < n:
        lon, lat = random.uniform(minx, maxx), random.uniform(miny, maxy)
        point = Point(lon, lat)
        strefa_wplywu = point.buffer(1500)  # Strefa wpływu o promieniu 1,5 km
        x, y = point.x, point.y
        shopValue = list(shops.sample([(x, y)]))
        roadValue = list(drogi.sample([(x, y)]))
        
        # Sprawdzenie czy punkt znajduje sie w granicach
        if not granica.contains(point).any():
            continue
        
        # Sprawdzenie czy bufor punktu nachodzi na inne bufory
        if sprawdz_przeciecie([x.buffer(1500) for x in sklepy], strefa_wplywu):
            continue
        
        if probkiShop(x, y, shopValue):
            continue
        
        if probkiRoad(x, y, roadValue):
            continue
        sklepy.append(point)
        lokalizacje = gpd.GeoDataFrame(geometry = sklepy, crs = "epsg:2180")
        coord_list = [(x, y) for x, y in zip(lokalizacje["geometry"].x, lokalizacje["geometry"].y)]
        lokalizacje["shopDist"] = [x for x in shops.sample(coord_list)]
        lokalizacje["pathDist"] = [x for x in drogi.sample(coord_list)]
        lokalizacje = lokalizacje.applymap(replace_nan)
    return lokalizacje


def findShops(n = 6, t = 200000):
    score = 0
    while score < t:
        lokale = generate_shops(n)
        buff = lokale.buffer(1500).unary_union
        przeciecie = ludnosc[ludnosc.geometry.intersects(buff)]
        score = przeciecie['ludnosc'].sum()
        print(score)
        axes = granica.plot(facecolor = "grey")
        lokale.plot(ax = axes, facecolor = "red")
    return lokale, score

result, score = findShops(6, 110000)
# =============================================================================
# axes = granica.plot(facecolor = "grey")
# result.plot(ax = axes, facecolor = "red")
# result.head()
# =============================================================================











import geopandas as gpd
import os

os.chdir("C:/Users/wojci/Desktop/IV_rok/data/data")

#granica = gpd.read_file("Poznan.gpkg")
#agranica = granica['geometry']
#agranica.to_file("Preprocess/granica.gpkg", driver = "GPKG")

#%% preprocessing
granica = gpd.read_file("Preprocess/granica.gpkg")
def preprocess(obj):
    obj = obj.to_crs("EPSG:2180")
    obj = gpd.clip(obj, granica)
    return obj

# POI- preprocess
POI = gpd.read_file("wielkopolskie-latest-free.shp/gis_osm_pois_free_1.shp")
POI = preprocess(POI)
POI = POI[POI.fclass == "supermarket"]
POI.fclass.unique()
POI = POI[POI['fclass'].isin(["bakery", "supermarket", "food_court"])]
POI.plot()


POI_buff = POI.buffer(1000)
POI_buff.plot()
axes = granica.plot(facecolor = "grey")
POI_buff.plot(ax = axes)

POI.to_file("Preprocess/POIS.gpkg", driver = "GPKG")



#%% Preprocess roads
roads = gpd.read_file("wielkopolskie-latest-free.shp/gis_osm_roads_free_1.shp")
roads = preprocess(roads)
roads = roads[roads['fclass'].isin(["footway", "path", "cycleway"])]
roads.plot()
roadsBuff = roads.buffer(100)
roadsBuff.plot()

roads.to_file("Preprocess/roads.gpkg", driver = "GPKG")
roadsBuff.to_file("Preprocess/roadsBuff.gpkg", driver = "GPKG")



#%% Przeskalowanie wartosci ludnosc
#lud = gpd.read_file("ludnosc_poznan_250.gpkg")
#lud.plot()
import rasterio as rio
ludRast = rio.open("ludnosc.tif")
buff = df.buffer(1500)
buff.plot()
from rasterio.mask import mask
out_image, transformed = mask(ludRast, buff.geometry, crop = True, filled = True)
from rasterio.plot import show
show(out_image)
out_image

import numpy as np
from sklearn import preprocessing
ludnoscOG = ludRast.read()
ludnoscOG.shape
ludnosc = np.reshape(ludnoscOG, (ludnoscOG.shape[1], ludnoscOG.shape[2]))
ludnosc.shape
#lud_scaled = preprocessing.MinMaxScaler().fit_transform(ludnosc)
scaler = preprocessing.MinMaxScaler()
ludnosc = scaler.fit_transform(ludnosc)
ludnosc = np.reshape(ludnosc, (ludnoscOG.shape[0], ludnoscOG.shape[1], ludnoscOG.shape[2]))
ludnosc.shape


profile = ludRast.profile
#with rio.open('example.tif', 'w', **profile) as dst:
#        dst.write(ludnosc.astype(ludRast.dtypes[0]))


#%% Preprocessing
import geopandas as gpd
import os
os.chdir("C:/Users/wojci/Desktop/IV_rok/data/data")

granica = gpd.read_file("Preprocess/granica.gpkg")

def preprocess(obj):
    obj = obj.to_crs("EPSG:2180")
    obj = gpd.clip(obj, granica)
    return obj

# POI- preprocess
POI = gpd.read_file("wielkopolskie-latest-free.shp/gis_osm_pois_free_1.shp")
POI = preprocess(POI)
POI.fclass.unique()
POI = POI[POI['fclass'].isin(["bakery", "supermarket", "food_court"])]
POI.plot()

axes = granica.plot(facecolor = "grey")
POI.plot(ax = axes)

POI.to_file("Preprocess/POIS.gpkg", driver = "GPKG")



# Preprocess roads
roads = gpd.read_file("wielkopolskie-latest-free.shp/gis_osm_roads_free_1.shp")
roads = preprocess(roads)
roads = roads[roads['fclass'].isin(["footway", "path", "cycleway"])]
roads.plot()

roads.to_file("Preprocess/roads.gpkg", driver = "GPKG")

##### OBLICZENIE MAP ODLEGLOSCI I RASTERYZACJA ZOSTALA DOKONANA W R - plik Jankowiak_Skrypt

#%% Import narzedzi
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import rasterio as rio
import numpy as np
import pandas as pd
import random

#%% Wczytanie danych
shops = rio.open("shops.tif")
drogi = rio.open("drogi.tif")
granica = gpd.read_file("Preprocess/granica.gpkg")
ludnosc = gpd.read_file("ludnosc_poznan_250.gpkg")
a = granica.bounds

minx = int(a['minx'].min())
maxx = int(a['maxx'].max())
miny = int(a['miny'].min())
maxy = int(a['maxy'].max())


#%% Generowanie lokalizacji sklepÃ³w
sklepy = []
sillShop = 600
sillRoad = 50

def replace_nan(value):
    if pd.isna(value):
        return "> {}".format(round(shops.read().max()), 2)
    else:
        return value

def sprawdz_przeciecie(strefy, nowa_strefa):
    for strefa in strefy:
        if strefa.intersects(nowa_strefa):
            return True
    return False


def probkiRoad(x, y, roadValue):
    if roadValue[0][0] > sillRoad:
        return True
    return False


def probkiShop(x, y, shopValue):
    if shopValue[0][0] < sillShop:
        return True
    return False

def ludzie(prog):
    if prog < 1000:
        return True
    return False
    

def generate_shops(n):
    sklepy = []
    while len(sklepy) < n:
        lon, lat = random.uniform(minx, maxx), random.uniform(miny, maxy)
        point = Point(lon, lat)
        strefa_wplywu = point.buffer(1500)  # Strefa wpÅ‚ywu o promieniu 1,5 km
        x, y = point.x, point.y
        shopValue = list(shops.sample([(x, y)]))
        roadValue = list(drogi.sample([(x, y)]))
        
        # Sprawdzenie czy punkt znajduje sie w granicach
        if not granica.contains(point).any():
            continue
        
        # Sprawdzenie czy bufor punktu nachodzi na inne bufory
        if sprawdz_przeciecie([x.buffer(1500) for x in sklepy], strefa_wplywu):
            continue
        
        if probkiShop(x, y, shopValue):
            continue
        
        if probkiRoad(x, y, roadValue):
            continue
        sklepy.append(point)
        lokalizacje = gpd.GeoDataFrame(geometry = sklepy, crs = "epsg:2180")
        coord_list = [(x, y) for x, y in zip(lokalizacje["geometry"].x, lokalizacje["geometry"].y)]
        lokalizacje["shopDist"] = [x for x in shops.sample(coord_list)]
        lokalizacje["pathDist"] = [x for x in drogi.sample(coord_list)]
        lokalizacje = lokalizacje.applymap(replace_nan)
    return lokalizacje

# =============================================================================
# score = 0
# while score < 200000:
#     lokale = generate_shops()
#     buff = lokale.buffer(1500).unary_union
#     przeciecie = ludnosc[ludnosc.geometry.intersects(buff)]
#     score = przeciecie['ludnosc'].sum()
#     print(lokale)
#     print(score)
# =============================================================================

def findShops(n = 6, t = 200000):
    score = 0
    while score < t:
        lokale = generate_shops(n)
        buff = lokale.buffer(1500).unary_union
        przeciecie = ludnosc[ludnosc.geometry.intersects(buff)]
        score = przeciecie['ludnosc'].sum()
        print(score)
    return lokale, score

result, score = findShops(6, 100000)
axes = granica.plot(facecolor = "grey")
result.plot(ax = axes, facecolor = "red")
result.head()
