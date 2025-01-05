from scipy.stats import norm
import torch
import random
import numpy as np


counts = torch.load('class_count.pt', weights_only=True)
props = counts / sum(counts)
print(sum(props**2))


num_classes = len(props)
num_samples = 10000

# Simulate true labels based on the dataset distribution
true_labels = np.random.choice(num_classes, size=num_samples, p=props)

# Simulate random predictions using the same weights
random_predictions = np.random.choice(num_classes, size=num_samples, p=props)

# Check if predictions are correct
correct_predictions = random_predictions == true_labels

# Calculate simulated accuracy
simulated_accuracy = correct_predictions.mean()
print(simulated_accuracy)