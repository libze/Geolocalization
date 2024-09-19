import plotly.express as px
import pandas as pd

# Sample dictionary with countries and their counts
country_dict = {
    'Denmark': 3,
    'USA': 5,
    'Germany': 2,
    'India': 8,
    'Brazil': 4,
    'China': 7
}

# Convert dictionary to a DataFrame
country_data = pd.DataFrame(list(country_dict.items()), columns=['country', 'count'])

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