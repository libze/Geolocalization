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
import json


path = ".\\Data_test2\\"
DTU_BLACKHOLE = "/dtu/blackhole/05/146725/shards/"

save_path = DTU_BLACKHOLE + "/mp16/" #Path where images are stored.


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
        return country, city, address

    return country


# Check if img is there:
def remeber_img(new_img):
    with open("loaded_imgs.txt", "a") as file:
        file.write(new_img + "\n")


#get_country_from_latlon(55.676098, 12.568337)

def save_imgs_from_shards(shard_path, save_path, n=5):
    count = 0
    new_loaded_imgs = set()
    loaded_imgs = set()

    # Open the file and read it line by line
    with open('loaded_imgs.txt', 'r') as file:
        for line in file:
            # Strip any leading/trailing whitespace or newline characters
            loaded_imgs.add(line.strip())

    failed_imgs = []
    data_all_shards = []
    for shard in os.listdir(path):
        count2 = 0
        images = []
        data_per_shard = []
        count += 1
        if count == 5: break
        with open(path + shard, "rb") as infile:
            for i, record in tqdm(enumerate(msgpack.Unpacker(infile, raw=False)), total = n):
                count2 += 1
                try:
                    if record['id'] in loaded_imgs: continue
                    lat, lon = record["latitude"], record["longitude"]
                    country, city, loc_dict = get_country_from_latlon(lat, lon, city=True)
                    image = Image.open(BytesIO(record["image"]))
                    images.append(image)

                    # saving the images in their respective folders:
                    name = record['id'].split("/")[-1]
                    folders = [country.upper()[0], country.split("/")[0].strip()]
                    mkpath = "\\".join(folders)

                    Path(save_path, mkpath).mkdir(parents=True, exist_ok=True)
                    image.save(Path(save_path, mkpath, name))
                    data_per_shard.append([record['id'], lat, lon, country, city, loc_dict, str(Path(save_path, mkpath, name))])
                    remeber_img(record['id'])

                except:
                    failed_imgs.append([record['id'], (record['latitude'], record['latitude'])])

                if count2 > n:
                    break

            data_all_shards.append(data_per_shard)

    with open('failed_imgs', "wb") as file:
        pickle.dump(failed_imgs, file)

    print("Amount of failed imgs:", len(failed_imgs))
    print("Done")
    print(len(data_all_shards), print(len(data_all_shards[0])))

    return data_all_shards

data_all_shards = save_imgs_from_shards(path, save_path)

#testing
print(data_all_shards)

# Create csv
def create_csv(data, path, name):
    os.makedirs(path, exist_ok=True)
    with open(path + "/" + name, mode = "w", newline="") as file:
        writer = csv.writer(file)

        for item in data:
            writer.writerow([json.dumps(item)])


    print("CSV file created")

create_csv(data_all_shards, "csv's", "first_test.csv")
