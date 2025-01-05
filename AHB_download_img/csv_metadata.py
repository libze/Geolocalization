import msgpack
import pickle
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from pathlib import Path
from geopy.geocoders import Nominatim
import csv
import geopandas as gpd
from shapely.geometry import Point


path = ".\\Data_test2\\"
DTU_BLACKHOLE = "/dtu/blackhole/05/146725/shards/shards/"
#path = DTU_BLACKHOLE
save_path = DTU_BLACKHOLE + "/mp16/" #Path where images are stored.

file_path = "ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp"
world = gpd.read_file(file_path)

def get_country(lat, lon):
    # Create a point geometry
    point = Point(lon, lat)  # Note: lon comes before lat
    # Find the country containing the point
    for i, country in world.iterrows():
        if country['geometry'].contains(point):
            return country['ADMIN']  # Replace 'ADMIN' with the relevant column for country names

    return get_country_from_latlon(lat,lon)

def get_country_from_latlon(lat, lon, city=False):
    # Initialize Nominatim API
    geolocator = Nominatim(user_agent="reading_files_IM2GPS")

    # Reverse geocode (get location details from lat/lon)
    location = geolocator.reverse((lat, lon), exactly_one=True, language='en')
    # Get the address components
    address = location.raw['address']
    country = address.get('country', None)

    if city:
        city = address.get('city', None)
        return country  # , city, address

    return country


# Check if img is there:
def remeber_img(new_img):
    with open("loaded_imgs.txt", "a") as file:
        file.write(new_img + "\n")


def append_csv(data, file):
    with open(file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(data)

#get_country_from_latlon(55.676098, 12.568337)

def save_imgs_from_shards(shard_path, save_path, n=100, testing = False):
    count = 0
    loaded_imgs = set()


    # Open the file and read it line by line
    with open('loaded_imgs.txt', 'r') as file:
        for line in file:
            # Strip any leading/trailing whitespace or newline characters
            loaded_imgs.add(line.strip())

    failed_imgs = []
    data_all_shards = []
    for shard in os.listdir(shard_path):
        count2 = 0
        images = []
        count += 1
        if count == 5: break
        with open(path + shard, "rb") as infile:
            for i, record in tqdm(enumerate(msgpack.Unpacker(infile, raw=False)), total=n):
                count2 += 1
                try:
                    if record['id'] in loaded_imgs: continue
                    lat, lon = record["latitude"], record["longitude"]
                    country = get_country(lat, lon)

                    image = Image.open(BytesIO(record["image"]))
                    images.append(image)

                    # saving the images in their respective folders:
                    name = record['id'].split("/")[-1]
                    folders = [country.upper()[0], country.split("/")[0].strip()]
                    mkpath = "\\".join(folders)

                    Path(save_path, mkpath).mkdir(parents=True, exist_ok=True)
                    image.save(Path(save_path, mkpath, name))
                    remeber_img(record['id']), append_csv([record['id'], lat, lon, country, str(Path(save_path, mkpath, name))], "csv's/first_test.csv")


                except:
                    failed_imgs.append([record['id'], (record['latitude'], record['latitude'])])
                if testing:
                    if count2 > n:
                        break


    with open('failed_imgs', "wb") as file:
        pickle.dump(failed_imgs, file)

    print("Amount of failed imgs:", len(failed_imgs))
    print("Done")
    #print(len(data_all_shards), print(len(data_all_shards[0])))

    return data_all_shards

save_imgs_from_shards(path, save_path, testing=True)

#testing
#print(data_all_shards)

# Create csv

#check list
# testing = False
#break count = 5
#path = DTU_BLACKHOLE