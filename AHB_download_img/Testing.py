import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import ast  # To safely evaluate string representations of dictionaries
from torchvision import transforms as trn
import numpy as np
import json
from dataloader import CustomDataset



if __name__ == '__main__':
    dataset = CustomDataset(csv_file="csv's/first_test.csv")
    print(list(dataset.int_to_country_dict.keys())[0], type(list(dataset.int_to_country_dict.keys())[0]))
    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    # Example usage
    for batch in dataloader:       # Access IDs
        for id in batch['country_int']:
            print(dataset.int_to_country_dict[id.item()])
        #print(batch['image'])    # Access images
        #print(batch['info_dict'])