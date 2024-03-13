import torch
import torchbearer
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchbearer import Trial
import matplotlib.pyplot as plt
import json
import os

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
else:
    print ("MPS device not found.")

seed = 7
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(seed)

# flatten 28*28 images to a 784 vector for each image
transform = transforms.Compose([
    transforms.ToTensor(),  # convert to tensor
    transforms.Lambda(lambda x: x.view(-1))  # flatten into vector
])

# load data
trainset = MNIST(".", train=True, download=True, transform=transform)
testset = MNIST(".", train=False, download=True, transform=transform)

# create data loaders
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
testloader = DataLoader(testset, batch_size=128, shuffle=True)

# define baseline model
class BaselineModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(BaselineModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        if not self.training:
            out = F.softmax(out, dim=1)
        return out

layer_sizes = [1000, 10000, 50000, 100000, 300000, 500000]
# layer_sizes = [1000, 5000]

NUM_EPOCHS = 100
all_train_accuracies = []
all_test_accuracies = []
all_train_losses = []
all_test_losses = []

for size in layer_sizes:
    model = BaselineModel(784, size, 10).to(mps_device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    train_accuracies = []
    test_accuracies = []
    train_losses = []
    test_losses = []

    for epoch in range(NUM_EPOCHS):
        print(f'Layer Size: {size}, Epoch: {epoch +1}')
        trial = Trial(model, optimizer, loss_function, metrics=['accuracy', 'loss']).to(mps_device)
        trial.with_generators(trainloader, test_generator=testloader)
        trial.run(epochs=1)
        
        train_results = trial.evaluate(data_key=torchbearer.TRAIN_DATA)
        test_results = trial.evaluate(data_key=torchbearer.TEST_DATA)
        
        train_accuracies.append(train_results['train_acc'])
        test_accuracies.append(test_results['test_acc'])
        train_losses.append(train_results['train_loss'])
        test_losses.append(test_results['test_loss'])
    
    all_train_accuracies.append(train_accuracies)
    all_test_accuracies.append(test_accuracies)
    all_train_losses.append(train_losses)
    all_test_losses.append(test_losses)

# Creating necessary directories
folders = ["individual", "train", "test", "joint"]
for folder in folders:
    os.makedirs(f"results/{folder}", exist_ok=True)

def save_combined_plot(train_data, test_data, title, folder, filename, label1='Train', label2='Test'):
    plt.figure()
    plt.plot(train_data, label=label1)
    plt.plot(test_data, label=label2)
    plt.title(title)
    plt.legend()
    plt.savefig(f"results/{folder}/{filename}.png")
    plt.close()

# Function to save joint plots for all layer sizes
def save_joint_plot(data, title, folder, filename, layer_sizes):
    plt.figure()
    for i, size in enumerate(layer_sizes):
        plt.plot(data[i], label=f'Layer Size {size}')
    plt.title(title)
    plt.legend()
    plt.savefig(f"results/{folder}/{filename}.png")
    plt.close()

# Saving the individual, train, test, and joint plots
for i, size in enumerate(layer_sizes):
    # Save individual combined plots
    save_combined_plot(all_train_accuracies[i], all_test_accuracies[i], f"Accuracy for Layer Size {size}", "individual", f"{size}_accuracy")
    save_combined_plot(all_train_losses[i], all_test_losses[i], f"Loss for Layer Size {size}", "individual", f"{size}_loss")

# Save aggregated training and testing plots
save_joint_plot(all_train_accuracies, "Training Accuracy for All Layer Sizes", "train", "all_accuracy", layer_sizes)
save_joint_plot(all_train_losses, "Training Loss for All Layer Sizes", "train", "all_loss", layer_sizes)
save_joint_plot(all_test_accuracies, "Testing Accuracy for All Layer Sizes", "test", "all_accuracy", layer_sizes)
save_joint_plot(all_test_losses, "Testing Loss for All Layer Sizes", "test", "all_loss", layer_sizes)

# Save joint plots for accuracy and loss
save_joint_plot(all_train_accuracies + all_test_accuracies, "Accuracy for All Layer Sizes", "joint", "all_accuracy", layer_sizes + layer_sizes)
save_joint_plot(all_train_losses + all_test_losses, "Loss for All Layer Sizes", "joint", "all_loss", layer_sizes + layer_sizes)

# Organize the accuracy and loss data for each epoch
results_data = {
    "train_accuracies": {str(size): accuracies for size, accuracies in zip(layer_sizes, all_train_accuracies)},
    "test_accuracies": {str(size): accuracies for size, accuracies in zip(layer_sizes, all_test_accuracies)},
    "train_losses": {str(size): losses for size, losses in zip(layer_sizes, all_train_losses)},
    "test_losses": {str(size): losses for size, losses in zip(layer_sizes, all_test_losses)}
}

# Save the accuracy and loss data to a JSON file
with open('results/results_data.json', 'w') as json_file:
    json.dump(results_data, json_file, indent=4)

print("Accuracy and loss data saved to results/results_data.json")
