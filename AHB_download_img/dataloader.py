import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import ast  # To safely evaluate string representations of dictionaries
from torchvision import transforms as trn
import numpy as np
import json


class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=True):
        """
        Args:
            csv_file (str): Path to the CSV file.
            transform (callable, optional): Optional transforms to be applied to images.
        """
        #self.data = pd.read_csv(csv_file, names=['id', 'lat', 'lon', 'country', 'path'], header=None)

        # Read the CSV into a pandas DataFrame
        self.data = pd.read_csv(csv_file, names=['id', 'lat', 'lon', 'country', 'path'], header=None, encoding='utf-8')

        # Convert the JSON string in the fourth column to a dictionary
        #self.data['loc_dict'] = self.data['loc_dict'].apply(json.loads)

        # Convert the appropriate columns back to their original types (float, etc.)

        self.data['lat'] = self.data['lat'].astype(float)  # Convert column 2 to float
        self.data['lon'] = self.data['lon'].astype(float)  # Convert column 3 to float
        #self.data.drop(['lat', 'lon'], axis=1, inplace = True)
        self.data['country_int'], unique_ids_country = pd.factorize(self.data['country'])
        self.data['id'], unique_id = pd.factorize(self.data['id'])
        self.int_to_country_dict = dict(enumerate(unique_ids_country))
        self.int_to_id_dict = dict(enumerate(unique_id))
        self.data.drop('country', axis=1, inplace=True)
        self.transform = transform
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Extract data from the CSV
        row = self.data.iloc[idx]
        image_path = row['path']

        lat = row['lat']
        lon = row['lon']
        country_int = row['country_int']
        #city = row['city']
        # info_dict = ast.literal_eval(row['loc_dict'])  # Convert string to dictionary

        # Load the image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            self.transform = trn.Compose([
                trn.Resize((256, 256)),
                trn.CenterCrop(224),
                trn.ToTensor(),
                trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            image = self.transform(image)

        # Pack data into a dictionary
        sample = {
            'id': row['id'],
            'lat': lat,
            'lon': lon,
            'country_int': country_int,
            #'city': city,
            #'loc_dict': loc_dict,
            'image': image,
        }
        return sample


if __name__ == '__main__':
    print('Running dataloader file')
    centre_crop = trn.Compose([
            trn.Resize((256, 256)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    dataset = CustomDataset(csv_file="csv's/first_test.csv", transform=centre_crop)

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # Example usage
    for batch in dataloader:
        print(batch['id'])



