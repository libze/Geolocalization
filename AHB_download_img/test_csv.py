import csv
import json

# Specify the CSV file name
filename = "csv's/first_test.csv"

# Read dictionaries back from CSV
with open(filename, mode='r', newline='') as file:
    reader = csv.reader(file)

    # Load each row as a dictionary (by parsing the JSON string)
    loaded_data = []
    for row in reader:
        # Convert the JSON string back to a dictionary
        loaded_data.append(json.loads(row[0]))  # row[0] is the string representation of the dictionary

# Print the loaded data
print(loaded_data)

print(loaded_data[0][0][5]['road'])