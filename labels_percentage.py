import pandas as pd
import numpy as np

path = "AHB_download_img/csv's/first_test.csv"
data = pd.read_csv(path, names=['id', 'lat', 'lon', 'country', 'city', 'loc_dict', 'path'], header=None, encoding='utf-8')
countries = data["country"]

countries_count = dict()
for country in countries:
    if country in countries_count:
        countries_count[country] += 1  # Increment count if country is already in the dict
    else:
        countries_count[country] = 1

full_sum = sum(countries_count.values())
sorted_items = sorted(countries_count.items(), key=lambda x: x[1], reverse=True)

# Separate keys and values
keys = [item[0] for item in sorted_items]
values = [item[1] for item in sorted_items]

for i in range(5):
    print(keys[i], values[i]/full_sum)