import requests
import piexif
import pandas as pd
import os
from tqdm import tqdm


imgs_file = "C:/Users/August/PycharmProjects/GeoEstimation/resources/mp16_urls.csv"

#imgs_file = "C:/Users/August/PycharmProjects/GeoEstimation/resources/yfcc25600_urls.csv"
df = pd.read_csv(imgs_file)

os.makedirs('Images', exist_ok=True)

def get_img(image_url, img_name):
    img_name = img_name.replace('/', '_')
    img_data = requests.get(image_url).content
    with open(f'Images/{img_name}', 'wb') as handler:
        handler.write(img_data)

failed_indexes = []

df.columns = ['Name', 'img']

i = 0
for (img, name) in tqdm(zip(df['img'], df['Name']), total= 50000): # len(df['img'])):
    if i == 50000:
        break

    try:
        get_img(img, name)
        i += 1
    except:
        failed_indexes.append(name.replace('/', '_'))




