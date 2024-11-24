from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Specify the dataset
dataset_owner = "habedi"  # Replace with the dataset owner's username
dataset_name = "large-dataset-of-geotagged-images"  # Replace with the dataset name

# List all files in the dataset
print("Fetching dataset file list...")
files = api.dataset_list_files(f"{dataset_owner}/{dataset_name}").files
print(files)
# Count and display files
print(f"Total files in dataset: {len(files)}")
count = 0
for file in files[10]:  # Display the first 10 files as a preview
    print(file.name)
    count += 1

print(count)