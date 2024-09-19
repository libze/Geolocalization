import matplotlib.pyplot as plt
import pickle
from collections import Counter
from tqdm import tqdm
import pandas as pd
import plotly.express as px


with open('countries', 'rb') as file:
    countries = pickle.load(file)

countries_count = {}

for name, country, lat, lon in tqdm(countries, total = len(countries)):
    if country in countries_count:
        countries_count[country] += 1  # Increment count if country is already in the dict
    else:
        countries_count[country] = 1


# Extract countries and their counts
countries = list(countries_count.keys())
counts = list(countries_count.values())
none_indices = [index for index, item in enumerate(countries) if item is None]
countries = [item for idx, item in enumerate(countries) if idx not in none_indices]
counts = [item for idx, item in enumerate(counts) if idx not in none_indices]
# Plot the bar chart

# Convert dictionary to a DataFrame
country_data = pd.DataFrame(list(countries_count.items()), columns=['country', 'count'])

# Plotly Express does not require a pre-built map; it uses the 'country' column to match country names
fig = px.choropleth(
    country_data,
    locations='country',
    locationmode='country names',
    color='count',
    color_continuous_scale='OrRd',
    title='Country Counts on World Map',
    labels={'count': 'Count'}
)

# Show the plot
fig.show()