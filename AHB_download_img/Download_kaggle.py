from tqdm import tqdm
import os
import kaggle

from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

# Define target directory
target_dir = "/dtu/blackhole/05/146725/shards"
target_dir = "Data_test2"
os.makedirs(target_dir, exist_ok=True)  # Create directory if it doesn't exist

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Dataset information
dataset_owner = "habedi"
dataset_name = "large-dataset-of-geotagged-images"

# List all files in the dataset
print("Fetching dataset file list...")
files = api.dataset_list_files(f"{dataset_owner}/{dataset_name}").files
print(files, len(files))

# Download each shard
def download_from_kaggle(target_dir):
    for file in files:

        if file.name.split("/")[-1] in os.listdir(target_dir):
            continue
        file_name = file.name

        api.dataset_download_file(
            dataset=f"{dataset_owner}/{dataset_name}",
            file_name=file_name,
            path=target_dir
        )

        with zipfile.ZipFile(os.path.join(target_dir, file.name.split("/")[-1] + ".zip"), 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        os.remove(os.path.join(target_dir, file.name.split("/")[-1] + ".zip"))
        print("Download continues")


download_from_kaggle(target_dir)
print('Done')
