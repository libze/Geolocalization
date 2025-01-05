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
from torch.utils.data import WeightedRandomSampler



dtu_path = "x½shards/shards/"
path = "AHB_download_img/Features"
server = True
amount_path = 2000

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
        self.fc = nn.Linear(input_size, temp_size)  # First fully connected layer
        self.dropout = nn.Dropout(dropout_rate)    # Dropout layer to reduce overfitting
        self.fc2 = nn.Linear(temp_size, num_classes)  # Second fully connected layer (output layer)

    def forward(self, x):
        x = self.fc(x)       # Pass through the second FC layer
        return x

def append_csv(data, file):
    with open(file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(data)


features = []
labels = []
for i, pth in enumerate(os.listdir(path)):
    pth_data = torch.load(path + "/" + pth, weights_only='False')
    features.append(pth_data['features'])
    labels.append(pth_data['labels'])

    if i == amount_path:
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
class_weights = 1.0 / (class_counts.float() + 1e-6)  # Avoid division by zero
class_weights /= class_weights.sum()  # Normalize weights (optional)

# Move weights to the device and set up the loss function
class_weights = class_weights.to(device)

# Example dimensions
input_size = 2048

model = SingleFCModel(input_size, num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=0.001)



  # Convert labels to int tensor
dataset = TensorDataset(features, labels)
num_epochs = 200  # Set the number of epochs for training
batch_size = 100  # Define the batch size

# Create a simple DataLoader if you have a custom dataset
# Example: Assuming you have images and labels as tensors
train_size = int(0.8 * len(features))
val_size = len(features) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

del train_dataset
del val_dataset
del dataset

best_loss = np.inf
training_info = []
val_info = []
print('Starting training')
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    correct_predictions_train = 0
    running_acc_train = 0
    inputs_in_epoch = 0

    for inputs, targets in train_loader:
        # Move inputs and targets to the GPU
        inputs, targets = inputs.to(device), targets.to(device)
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Update the parameters
        optimizer.step()

        __, predicted_train = torch.max(outputs, 1)  # Get the index of the highest logit
        correct_predictions_train = (predicted_train == targets).sum().item()

        running_acc_train += correct_predictions_train
        inputs_in_epoch += targets.size(0)
    #print(running_acc_train, inputs_in_epoch)
    training_info.append([epoch + 1, running_acc_train/inputs_in_epoch * 100])

    with torch.no_grad():
        running_loss = 0.0
        correct_predictions_val = 0
        total_samples_val = 0
        running_acc_val = 0
        for inputs_val, targets_val in val_loader:
            # Move inputs and targets to the GPU
            inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
            # Forward pass
            outputs_val = model(inputs_val)
            loss_val = criterion(outputs_val, targets_val)
            running_loss += loss_val.item()
            _, predicted = torch.max(outputs_val, 1)  # Get the index of the highest logit
            correct_predictions_val = (predicted == targets_val).sum().item()
            total_samples_val += targets_val.size(0)

            running_acc_val += correct_predictions_val

        if best_loss > running_loss:
            best_loss = running_loss
            best_model = model
            best_epoch = epoch + 1
            best_acc = correct_predictions_val/total_samples_val * 100
    # save to csv
    #append_csv(, "csv's/training_data")
        val_info.append([epoch + 1, running_acc_val / total_samples_val * 100])

print("Best epoch:", best_epoch)
print("Best acc:", best_acc)
print("Best loss:", best_loss)

print(training_info)
print(val_info)
x_train = [training_info_element[0] for training_info_element in training_info]
y_train = [training_info_element[1] for training_info_element in training_info]
x_val = [val_info_element[0] for val_info_element in val_info]
y_val = [val_info_element[1] for val_info_element in val_info]

plt.plot(x_train, y_train)
plt.plot(x_val, y_val)
plt.legend(['training', 'validation'])
plt.savefig(f'trainVsVal_{amount_path * 10000}_testing_dists.png')
plt.show()


