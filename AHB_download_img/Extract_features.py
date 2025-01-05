import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataloader import CustomDataset
import os
import torchvision.models as models
import pickle
from pathlib import Path
import pandas as pd
from itertools import islice
import time

start_time = time.time()
print('Starting')



def create_dir(path, folder_name):
    Path(path, folder_name).mkdir(parents=False, exist_ok=True)

# the architecture to use
arch = 'resnet50'
path = "../AHB_model"
# load the pre-trained weights
model_file = f'{path}/{arch}_places365.pth.tar'
if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url)

n_classes = 365

model = models.__dict__[arch](num_classes=n_classes)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage, weights_only=True)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()


# Remove or bypass the final layer (e.g., `fc`) to extract features
feature_extractor = nn.Sequential(*list(model.children())[:-1])  # Removes the last layer

# Alternatively, if you know the specific layer, you can forward pass until that layer
# For example, for ResNet:
# feature_extractor = nn.Sequential(model.conv1, model.bn1, ..., model.layer4)

# Data loader for input data
dataset = CustomDataset("csv's/first_test.csv")
data_loader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=4)  # Define your DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to device
feature_extractor.to(device)

# Store extracted features

#print(os.listdir('Features/'))

if __name__ == '__main__':
    feature_vectors = []
    labels = []
    Path("Features").mkdir(exist_ok=True)

    if len(os.listdir('Features')):
        amount_to_skip = max(int(filename.replace('features', '').replace('.pth', '')) for filename in os.listdir('Features'))
    else: amount_to_skip = 0

    with open('../int_to_country_dict', "wb") as file:
        pickle.dump(dataset.int_to_country_dict, file)
    with open('int_to_id_dict', "wb") as file:
        pickle.dump(dataset.int_to_id_dict, file)

    batch_num = amount_to_skip
    test_tqdm = len(pd.read_csv("csv's/first_test.csv"))
    with torch.no_grad():
        for batch in islice(data_loader, amount_to_skip, None):
            inputs, targets = batch['image'], batch['country_int']
            inputs = inputs.to(device)

            # Pass data through the feature extractor
            features = feature_extractor(inputs)

            # Flatten features if needed (e.g., for ResNet where `avgpool` outputs a 4D tensor)
            features = torch.flatten(features, start_dim=1)

            feature_vectors.append(features.cpu())  # Store feature vectors
            labels.append(targets.cpu())  # Store labels for reference

            # Concatenate all feature vectors and labels and save
            if batch_num % 5 == 0: # First save will be with 1 batch
                feature_vectors = torch.cat(feature_vectors, dim=0)
                labels = torch.cat(labels, dim=0)

                # Save feature vectors and labels
                torch.save({'features': feature_vectors, 'labels': labels, 'ids': batch['id']}, f"Features/features{batch_num}.pth")
                feature_vectors = []
                labels = []
            batch_num += 1


    # End measuring the total time after the loop finishes
    end_time = time.time()

    # Print the total elapsed time
    print(f"Total time taken: {end_time - start_time:.4f} seconds")