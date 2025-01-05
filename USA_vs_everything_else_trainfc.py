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
from collections import defaultdict
import pickle
import torch.nn.init as init


seed = 3
torch.manual_seed(seed)


dtu_path = "/dtu/blackhole/05/146725/shards/shards/"
path = "AHB_download_img/Features"
server = True
amount_path = 10000000033
bar_chart_amount = 212
verbose = False
num_classes = 2

num_epochs = 40  # Set the number of epochs for training

if server:
    path = dtu_path + "Features/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('Running on GPU')
else:
    print('Running on CPU')
class SingleFCModel(nn.Module):
    def __init__(self, input_size, num_classes, temp_size=1012):
        super(SingleFCModel, self).__init__()
        self.fc = nn.Linear(input_size, input_size + 500)  # First fully connected layer
        self.fc2 = nn.Linear(input_size + 500, input_size - 200)
        self.fc3 = nn.Linear(input_size - 200, input_size - 500)
        self.fc4 = nn.Linear(input_size - 500, temp_size - 300)
        self.fc5 = nn.Linear(temp_size - 300, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize each layer with Xavier initialization
        for layer in [self.fc, self.fc2, self.fc3, self.fc4, self.fc5]:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)  # Xavier initialization
                if layer.bias is not None:
                    init.zeros_(layer.bias)
    def forward(self, x):
        x = self.fc(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        x = torch.relu(x)
        x = self.fc5(x)
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

with open('int_to_country_dict', 'rb') as file:
    int_to_country = pickle.load(file)

features = []
labels = []
ids = []
for i, pth in enumerate(os.listdir(path)):
    pth_data = torch.load(path + "/" + pth, weights_only='False')

    features.append(pth_data['features'])
    labels.append(pth_data['labels'])
    ids.append(pth_data['ids'])


    if not server and i > 5:
        print("I stopped loading data")
        break


features_test = []
labels_test = []
id_test = []

# Flatten feature blocks
for feature_block in features:
    for feature in feature_block:
        features_test.append(feature.numpy() if isinstance(feature, torch.Tensor) else feature)

# Flatten label blocks
for label_block in labels:
    for label in label_block:
        labels_test.append(label.numpy() if isinstance(label, torch.Tensor) else label)

bin_label = []
for label in labels_test:
    if label == 9:
        bin_label.append(1)
    else:
        bin_label.append(0)


for id_block in ids:
    for id in id_block:
        id_test.append(id.numpy() if isinstance(id, torch.Tensor) else id)


# Convert to tensors
features = torch.tensor(np.array(features_test), dtype=torch.float32)
labels = torch.tensor(np.array(bin_label), dtype=torch.int64)
ids = torch.tensor(range(len(features)), dtype=torch.int64)


del features_test
del labels_test

features = torch.tensor(np.array(features), dtype=torch.float32)
labels = torch.tensor(np.array(labels), dtype=torch.int64)
ids = torch.tensor(range(len(features)), dtype=torch.int64)



class_counts = torch.bincount(labels, minlength=num_classes)
class_weights = 1.0 / (class_counts.float() + 1e-6)  # Avoid division by zero
class_weights /= class_weights.sum()

class_weights = class_weights.to(device)

input_size = 2048

model = SingleFCModel(input_size, num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
criterion_reduction_none = nn.CrossEntropyLoss(weight=class_weights.to(device), reduction='none')
lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)


dataset = TensorDataset(features, labels, ids)

batch_size = 300  # Define the batch size


train_size = int(0.8 * len(features))
val_size = len(features) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

del train_dataset
del val_dataset
del dataset

best_acc = 0
training_info = []
val_info = []
val_info_top5 = []
training_info_top5 = []
training_info_loss = []
val_info_loss = []

id_loss = defaultdict(int)
country_loss = defaultdict(float)
country_correct = defaultdict(int)
country_samples = defaultdict(int)

print('Starting training')

for epoch in range(num_epochs):
    model.train()
    correct_predictions_train = 0
    running_acc_train = 0
    inputs_in_epoch = 0
    sum_loss = 0
    running_acc_top5 = 0

    for inputs, targets, ids in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, targets)
        loss_for_dict = criterion_reduction_none(outputs, targets)
        for id_, loss_ in zip(ids, loss_for_dict):
            id_loss[id_.item()] += loss_.item()

        # Update per-country loss
        for id_, target, loss_ in zip(ids, targets, loss_for_dict):
            country_loss[target.item()] += loss_.item()
            country_samples[target.item()] += 1

        # Backpropagation and optimizer step
        loss.backward()
        optimizer.step()

        # Update per-country accuracy
        __, predicted_train = torch.max(outputs, 1)
        for target, prediction in zip(targets, predicted_train):
            if target.item() == prediction.item():
                country_correct[target.item()] += 1

        # Get top-5 accuracy
        #top5_values, top5_indices = torch.topk(outputs, 5, dim=1)
        #correct_in_top5 = torch.any(top5_indices == targets.unsqueeze(1), dim=1).sum().item()
        #running_acc_top5 += correct_in_top5

        # Update epoch statistics
        running_acc_train += (predicted_train == targets).sum().item()
        inputs_in_epoch += targets.size(0)
        sum_loss += loss.item()

    # Epoch statistics
    training_info.append([epoch + 1, running_acc_train / inputs_in_epoch * 100])
    #training_info_top5.append([epoch + 1, running_acc_top5 / inputs_in_epoch * 100])
    training_info_loss.append([epoch + 1, sum_loss/ inputs_in_epoch])

    with torch.no_grad():
        running_loss = 0.0
        total_correct_top1 = 0
        total_correct_top5 = 0
        total_samples_val = 0

        for inputs_val, targets_val, ids in val_loader:
            inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)

            outputs_val = model(inputs_val)
            loss_val = criterion(outputs_val, targets_val)
            running_loss += loss_val.item()

            # Top-1 accuracy
            _, predicted = torch.max(outputs_val, 1)  # Get the index of the highest logit
            total_correct_top1 += (predicted == targets_val).sum().item()

            # Top-5 accuracy
            #top5_values, top5_indices = torch.topk(outputs_val, 5, dim=1)
            #total_correct_top5 += torch.any(top5_indices == targets_val.unsqueeze(1), dim=1).sum().item()

            # Count total samples
            total_samples_val += targets_val.size(0)

        # Compute metrics
        running_acc_val = total_correct_top1 / total_samples_val * 100
        #running_acc_top5_val = total_correct_top5 / total_samples_val * 100
        running_loss = running_loss / total_samples_val

        # Track the best model
        if best_acc < running_acc_val:
            best_loss = running_loss
            best_model = model
            best_epoch = epoch + 1
            best_acc = running_acc_val

        # Log results
        val_info.append([epoch + 1, running_acc_val])
        #val_info_top5.append([epoch + 1, running_acc_top5_val])
        val_info_loss.append([epoch + 1, running_loss])

torch.save(best_model.state_dict(), f"model_weights_all_data_learningrate_{0.00001}_{seed}_{num_epochs}.pth")
print("Best epoch:", best_epoch)
print("Best acc:", best_acc)
print("Best loss:", best_loss)


average_loss_per_country = {k: country_loss[k] / country_samples[k] for k in country_loss}
accuracy_per_country = {k: country_correct[k] / country_samples[k] * 100 for k in country_correct}

# Convert integer keys to country names
average_loss_per_country_named = {int_to_country[k]: v for k, v in average_loss_per_country.items()}
accuracy_per_country_named = {int_to_country[k]: v for k, v in accuracy_per_country.items()}

# Sort and get top 10 and lowest 10 countries
top_10_loss = sorted(average_loss_per_country_named.items(), key=lambda x: x[1])[:10]
lowest_10_loss = sorted(average_loss_per_country_named.items(), key=lambda x: x[1], reverse=True)[:10]

top_10_accuracy = sorted(accuracy_per_country_named.items(), key=lambda x: x[1], reverse=True)[:10]
lowest_10_accuracy = sorted(accuracy_per_country_named.items(), key=lambda x: x[1])[:10]


x_train = [training_info_element[0] for training_info_element in training_info]
y_train = [training_info_element[1] for training_info_element in training_info]
x_val = [val_info_element[0] for val_info_element in val_info]
y_val = [val_info_element[1] for val_info_element in val_info]


x_train_loss = [training_info_element[0] for training_info_element in training_info_loss]
y_train_loss = [training_info_element[1] for training_info_element in training_info_loss]
x_val_loss = [val_info_element[0] for val_info_element in val_info_loss]
y_val_loss = [val_info_element[1] for val_info_element in val_info_loss]


# Function to plot line graphs
def plot_graph(x_values, y_values_list, labels, xlabel, ylabel, title, filename=None, server=False):
    plt.figure(figsize=(10, 6))
    for y_values, label in zip(y_values_list, labels):
        plt.plot(x_values, y_values, label=label)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    if server:
        if filename:
            plt.savefig(filename)
    if not server:
        plt.show()
    plt.clf()


# Function to plot bar charts
def plot_bar_chart(x_values, y_values, xlabel, ylabel, title, filename=None, server=False, verbose=False):
    plt.figure(figsize=(15, 6))
    plt.bar(x_values, y_values, color='skyblue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    if server:
        if filename:
            plt.savefig(filename)
    if verbose:
        plt.show()
    plt.clf()

def plot_bar_chart_custom(data, title, xlabel, ylabel, filename=None, server=False, verbose=False):
    countries, values = zip(*data)
    plt.figure(figsize=(10, 6))
    plt.bar(countries, values, color='skyblue')
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if server:
        if filename:
            plt.savefig(filename)
    if verbose:
        plt.show()
    plt.clf()

# Plot Training vs Validation Accuracy
plot_graph(
    x_values=x_train,
    y_values_list=[y_train, y_val],
    labels=['Training', 'Validation'],
    xlabel="Epochs",
    ylabel="Accuracy (%)",
    title="Training vs Validation Accuracy",
    filename=f'trainVsVal_acc_{amount_path * 10000}_learningrate_{lr}_{seed}_{num_epochs}_Xavierini.png' if server else None,
    server=server
)

# Plot Training vs Validation Loss
plot_graph(
    x_values=x_train_loss,
    y_values_list=[y_train_loss, y_val_loss],
    labels=['Training', 'Validation'],
    xlabel="Epochs",
    ylabel="Loss (mean)",
    title="Training vs Validation Loss",
    filename=f'trainVsVal_loss_{amount_path * 10000}_learningrate_{lr}_{seed}_{num_epochs}_Xavierini.png' if server else None,
    server=server
)
"""
# Plot Training vs Validation Top-5 Accuracy
y_train_top5 = [train_info_element[1] for train_info_element in training_info_top5]
y_val_top5 = [val_info_element[1] for val_info_element in val_info_top5]

plot_graph(
    x_values=x_train,
    y_values_list=[y_train_top5, y_val_top5],
    labels=['Training', 'Validation'],
    xlabel="Epochs",
    ylabel="Accuracy in Top 5% (%)",
    title="Training vs Validation Top-5 Accuracy",
    filename=f'trainVsVal_{amount_path * 10000}_learningrate_{lr}_{seed}_{num_epochs}_top5_Xavierini.png' if server else None,
    server=server
)


# Plot Bar Chart for Top Images by Loss
top_10 = dict(sorted(id_loss.items(), key=lambda item: item[1], reverse=True)[:bar_chart_amount])

plot_bar_chart(
    x_values=list(range(1, bar_chart_amount + 1)),
    y_values=list(top_10.values()),
    xlabel="Individual image",
    ylabel="Loss",
    title=f"Top {bar_chart_amount} Images by Loss",
    filename=f'barchart_{bar_chart_amount}_{amount_path}_learningrate_{lr}_{seed}_{num_epochs}_Xavierini.png' if server else None,
    server=server,
    verbose=not server and verbose
)

# Plot top 10 and lowest 10 countries by loss
plot_bar_chart_custom(
    data=top_10_loss,
    title="Top 10 Countries by Lowest Loss",
    xlabel="Country",
    ylabel="Average Loss",
    filename=f'top10_lowest_loss_{amount_path}.png' if server else None,
    server=server,
    verbose=not server and verbose
)
plot_bar_chart_custom(
    data=lowest_10_loss,
    title="Top 10 Countries by Highest Loss",
    xlabel="Country",
    ylabel="Average Loss",
    filename=f'top10_highest_loss_{amount_path}.png' if server else None,
    server=server,
    verbose=not server and verbose
)

plot_bar_chart_custom(
    data=lowest_10_accuracy,
    title="Top 10 Countries by Lowest Accuracy",
    xlabel="Country",
    ylabel="Accuracy (%)",
    filename=f'top10_lowest_accuracy_{amount_path}.png' if server else None,
    server=server,
    verbose=not server and verbose
)
plot_bar_chart_custom(
    data=top_10_accuracy,
    title="Top 10 Countries by Highest Accuracy",
    xlabel="Country",
    ylabel="Accuracy (%)",
    filename=f'top10_highest_accuracy_{amount_path}.png' if server else None,
    server=server,
    verbose=not server and verbose
)


# Save Loss Dictionary
if server:
    with open(f'Loss_dict_{amount_path}', 'wb') as file:
        pickle.dump(id_loss, file)

# Step 1: Extract top 10 most frequent classes (or top classes by accuracy)
top_10_classes = sorted(accuracy_per_country_named.items(), key=lambda x: x[1], reverse=True)[:10]
top_10_class_indices = [k for k, v in top_10_classes]

# Step 2: Collect true labels and predictions for all classes (1 pass through the data)
all_true_labels = []
all_predicted_labels = []

with torch.no_grad():
    for inputs_val, targets_val, _ in val_loader:
        inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)

        # Get predictions
        outputs_val = model(inputs_val)
        _, predicted = torch.max(outputs_val, 1)

        # Collect all true labels and predictions
        all_true_labels.extend(targets_val.tolist())
        all_predicted_labels.extend(predicted.tolist())

# Step 3: Generate confusion matrix for all classes
cm_all = confusion_matrix(all_true_labels, all_predicted_labels)

# Step 4: Filter out true labels and predicted labels for top 10 frequent classes
true_labels_filtered = []
predicted_labels_filtered = []

for i, label in enumerate(all_true_labels):
    if label in top_10_class_indices:
        true_labels_filtered.append(all_true_labels[i])
        predicted_labels_filtered.append(all_predicted_labels[i])

# Generate confusion matrix for Top 10 frequent classes
cm_frequent_classes = confusion_matrix(true_labels_filtered, predicted_labels_filtered, labels=top_10_class_indices)

# Step 5: Identify top-K misclassified classes from the full confusion matrix
misclassification_rate = []
for idx in range(len(top_10_class_indices)):
    misclassified = np.sum(cm_all[idx]) - cm_all[idx, idx]  # Misclassified samples for this class
    total = np.sum(cm_all[idx])  # Total samples of this class
    # Avoid division by zero
    if total > 0:
        misclassification_rate.append((top_10_class_indices[idx], misclassified / total))
    else:
        misclassification_rate.append((top_10_class_indices[idx], 0))

    # Sort by misclassification rate
top_misclassified_classes = sorted(misclassification_rate, key=lambda x: x[1], reverse=True)[:10]
top_misclassified_class_indices = [x[0] for x in top_misclassified_classes]

# Step 6: Filter out true labels and predicted labels for top misclassified classes
true_labels_filtered_misclassified = []
predicted_labels_filtered_misclassified = []

for i, label in enumerate(all_true_labels):
    if label in top_misclassified_class_indices:
        true_labels_filtered_misclassified.append(all_true_labels[i])
        predicted_labels_filtered_misclassified.append(all_predicted_labels[i])

# Generate confusion matrix for Top misclassified classes
cm_misclassified = confusion_matrix(true_labels_filtered_misclassified, predicted_labels_filtered_misclassified, labels=top_misclassified_class_indices)

# Convert indices to class names for readability
class_names_top_10 = top_10_class_indices
class_names_misclassified = top_misclassified_class_indices

# Step 7: Plot confusion submatrix for Top-K classes and misclassified classes
# Plot for Top-K most frequent classes
plt.figure(figsize=(10, 8))
sns.heatmap(cm_frequent_classes, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_top_10, yticklabels=class_names_top_10)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Submatrix for Top-K Frequent Classes")
plt.tight_layout()
if server:
    plt.savefig("confusion_matrix_top_k_frequent_classes.png")
if not server:
    plt.show()

# Plot for Top-K misclassified classes
plt.figure(figsize=(10, 8))
sns.heatmap(cm_misclassified, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_misclassified, yticklabels=class_names_misclassified)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Submatrix for Top-K Misclassified Classes")
plt.tight_layout()
if server:
    plt.savefig("confusion_matrix_top_k_misclassified_classes.png")
if not server:
    plt.show()

print("Confusion matrices saved as 'confusion_matrix_top_k_frequent_classes.png' and 'confusion_matrix_top_k_misclassified_classes.png'")


# Print top 10 losses for verification
if not server:
    print("Top 10 Losses:", top_10)


"""
