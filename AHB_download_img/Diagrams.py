import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

# Load the CSV file
csv = "csv's/meta_data_csv.csv"
df = pd.read_csv(
    csv,
    names=['id', 'lat', 'lon', 'country', 'city', 'loc_dict', 'path'],
    header=None,
    encoding='utf-8'
)

# Initialize a dictionary to count occurrences of each country
countries_count = {}

# Count occurrences of each country
for country in df["country"]:
    if pd.notna(country):  # Skip NaN values
        country = country.strip()  # Remove leading/trailing whitespace
        countries_count[country] = countries_count.get(country, 0) + 1

# Replace "United States of America" with "United States" if it exists
if 'United States of America' in countries_count:
    countries_count['United States'] = countries_count.pop('United States of America')

# Convert dictionary to a DataFrame
country_data = pd.DataFrame(list(countries_count.items()), columns=['country', 'count'])

# Clean up country names
country_data['country'] = country_data['country'].str.strip()

# Debugging and manual adjustments (optional)
#if 'United States' in country_data['country'].values:
    #country_data.loc[country_data['country'] == 'United States', 'count'] += 1200000  # Example manual adjustment

# Ensure 'count' column is numeric
country_data['count'] = pd.to_numeric(country_data['count'], errors='coerce').fillna(0)

# Sort the data for better visualization
country_data_sorted = country_data.sort_values("count", ascending=False)

# Plotly Express choropleth map
fig = px.choropleth(
    country_data,
    locations='country',
    locationmode='country names',
    color='count',
    color_continuous_scale=[
        [0, 'lightblue'],   # Lowest value: dark blue
        [0.5, 'yellow'],   # Middle value: yellow
        [1, 'red']         # Maximum value: red
    ],
    range_color=[country_data['count'].min(), country_data['count'].max()],
    title='Country Counts on World Map',
    labels={'count': 'Count'}
)
fig.show()

# Matplotlib bar chart
plt.bar(country_data_sorted["country"][:7], country_data_sorted["count"])
plt.xlabel('Country')
plt.ylabel('Count')
plt.title('Country Counts')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to fit all labels
plt.show()

# Display summary statistics for counts
print(country_data['count'].describe())
