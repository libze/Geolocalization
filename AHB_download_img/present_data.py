import folium
from folium.plugins import HeatMap
import pandas as pd


def create_interactive_map(csv):
    # Step 1: Load your data
    # Assuming you have a CSV file or pandas DataFrame with latitude and longitude
    df = pd.read_csv(csv, names=['id', 'lat', 'lon', 'country', 'city', 'loc_dict', 'path'], header=None, encoding='utf-8')

    # Step 2: Create a base map centered around a central point (e.g., world center)
    # You can adjust the zoom level (starting at 2 to show the world, for example)
    m = folium.Map(location=[20, 0], zoom_start=3, tiles="CartoDB Positron")

    # Step 3: Prepare the data for HeatMap (list of [lat, lon])
    heat_data = [[row['lat'], row['lon']] for index, row in df.iterrows()]

    # Step 4: Add the heatmap to the map
    HeatMap(heat_data).add_to(m)

    # Step 5: Save the map as an HTML file to view in a browser
    m.save("full_heatmap.html")



create_interactive_map("C:/Users/Libz/Documents/meta_data_csv.csv")
