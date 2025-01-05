from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_curve, auc
import csv
from torch.utils.data import DataLoader, random_split, TensorDataset
import random



dtu_path = "/dtu/blackhole/05/146725/shards/shards/"
path = "AHB_download_img/Features"
server = False
amount_path = 100000003

if server:
    path = dtu_path + "Features/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('Running on GPU')
else:
    print('Running on CPU')
class SingleFCModel(nn.Module):
    def __init__(self, input_size, num_classes, temp_size=1012, dropout_rate=0.5):
        super(SingleFCModel, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)  # First fully connected layer
        self.dropout = nn.Dropout(dropout_rate)    # Dropout layer to reduce overfitting
        self.fc2 = nn.Linear(input_size + 500, input_size - 200)
        self.fc3 = nn.Linear(input_size - 200, temp_size)
        self.fc4 = nn.Linear(temp_size, num_classes)



    def forward(self, x):
        x = self.fc(x)            # Pass input through the first FC layer
        return x

def append_csv(data, file):
    with open(file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(data)


# List all .pth files in the directory
pth_files = [f for f in os.listdir(path) if f.endswith('.pth')]
# Number of files to select
num_files_to_select = amount_path
# Randomly select the specified number of files
selected_files = random.sample(pth_files, min(num_files_to_select, len(pth_files)))

# File to save the selected filenames
output_file = "selected_files.txt"

# Write the selected filenames to the text file
with open(output_file, 'w') as file:
    for filename in selected_files:
        file.write(f"{filename}\n")


features = []
labels = []
for i, pth in enumerate(os.listdir(path)):
    pth_data = torch.load(path + "/" + pth, weights_only='False')
    features.append(pth_data['features'])
    labels.append(pth_data['labels'])

    if not server and i > 5:
        print("I stopped loading data")
        break


features_test = []
labels_test = []

# Flatten feature blocks
for feature_block in features:
    for feature in feature_block:
        features_test.append(feature.numpy() if isinstance(feature, torch.Tensor) else feature)

# Flatten label blocks
for label_block in labels:
    for label in label_block:
        labels_test.append(label.numpy() if isinstance(label, torch.Tensor) else label)

# Convert to tensors
features = torch.tensor(np.array(features_test), dtype=torch.float32)  # Convert features to float tensor
labels = torch.tensor(np.array(labels_test), dtype=torch.int64)

del features_test
del labels_test

num_classes = 221
class_counts = torch.bincount(labels, minlength=num_classes)  # Ensure counts for all classes
torch.save(class_counts, "class_count.pt")
class_weights = 1.0 / (class_counts.float() + 1e-6)  # Avoid division by zero

torch.save(class_weights, "class_weights_pre_norm.pt")

class_weights /= class_weights.sum()  # Normalize weights (optional)

# Move weights to the device and set up the loss function
class_weights = class_weights.to(device)

torch.save(class_weights, "class_weights.pt")
