import os
import pandas as pd
from geopy.geocoders import Nominatim
import geocoder
import pickle
from tqdm import tqdm
# initialize Nominatim API
#geolocator = Nominatim(user_agent="geoapiExercises")




file = "mp16_places365.csv"
df = pd.read_csv(file)


img_names = os.listdir("Images/")

from geopy.geocoders import Nominatim


def get_country_from_latlon(lat, lon):
    # Initialize Nominatim API
    geolocator = Nominatim(user_agent="reading_files_IM2GPS")

    # Reverse geocode (get location details from lat/lon)
    location = geolocator.reverse((lat, lon), exactly_one=True)

    # Get the address components
    address = location.raw['address']
    country = address.get('country', None)

    return country


    

counties = []
count = 0
# Extract data and connect to image
for img in tqdm(img_names):
    img = img.replace('_', '/')
    row = df.loc[df['IMG_ID'] == img]
    lat, lon = float(row['LAT']), float(row['LON'])

    try:
        c = get_country_from_latlon(lat, lon)
    except:
        count += 1
        continue
    counties.append([img, c, lat, lon])


with open("countries", "wb") as file:
    pickle.dump(counties, file)
