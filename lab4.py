import argparse
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
import numpy as np

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

def save_combined_plot(train_data, test_data, title, folder, filename, label1='Train', label2='Test'):
    plt.figure()
    plt.plot(train_data, label=label1)
    plt.plot(test_data, label=label2)
    plt.title(title)
    plt.legend()
    plt.savefig(f"results/{folder}/{filename}.png")
    plt.close()

def save_joint_plot(data, title, folder, filename, layer_sizes):
    plt.figure()
    for i, size in enumerate(layer_sizes):
        plt.plot(data[i], label=f'Layer Size {size}')
    plt.title(title)
    plt.legend()
    plt.savefig(f"results/{folder}/{filename}.png")
    plt.close()
    
def save_combined_joint_plot(train_data, val_data, title, folder, filename, layer_sizes):
    plt.figure()
    num_layers = len(layer_sizes)

    for i in range(num_layers):
        plt.plot(train_data[i], label=f'Train Layer {layer_sizes[i]}', linestyle='-', marker='o')

    for i in range(num_layers):
        plt.plot(val_data[i], label=f'Val Layer {layer_sizes[i]}', linestyle='--', marker='x')

    plt.title(title)
    plt.legend()
    plt.savefig(f"results/{folder}/{filename}.png")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--device', default='cuda', type=str, help='Device on which to run: "cuda" or "cpu"')
    parser.add_argument('--epochs', default=100, type=int, help='total epochs to run')
    parser.add_argument('--layer-sizes', nargs='+', type=int, default=[1000, 10000, 50000, 100000, 300000], help='List of layer sizes')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    folders = ["individual", "train", "test", "joint"]
    for folder in folders:
        os.makedirs(f"results/{folder}", exist_ok=True)

    seed = 7
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    trainset = MNIST(".", train=True, download=True, transform=transform)
    testset = MNIST(".", train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True)

    all_train_acc = []
    all_val_acc = []
    all_train_loss = []
    all_val_loss = []

    for size in args.layer_sizes:
        model = BaselineModel(784, size, 10).to(device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())

        trial = Trial(model, optimizer, loss_function, metrics=['accuracy', 'loss']).to(device)
        trial.with_generators(trainloader, val_generator=testloader)
        
        history = trial.run(epochs=args.epochs)
        results = trial.evaluate(data_key=torchbearer.TEST_DATA)
        
        
        train_loss = [epoch['loss'] for epoch in history]
        val_loss = [epoch['val_loss'] for epoch in history]
        
        train_acc = [epoch['acc'] for epoch in history]
        val_acc = [epoch['val_acc'] for epoch in history]
            
        torch.save(model.state_dict(), f'results/model_weights_{size}.pth')

        all_train_acc.append(train_acc)
        all_val_acc.append(val_acc)
        all_train_loss.append(train_loss)
        all_val_loss.append(val_loss)

    for i, size in enumerate(args.layer_sizes):
        save_combined_plot(all_train_acc[i], all_val_acc[i], f"Train vs Validation Accuracy for Layer Size {size}", "individual", f"{size}_accuracy")
        save_combined_plot(all_train_loss[i], all_val_loss[i], f"Train vs Validation Loss for Layer Size {size}", "individual", f"{size}_loss")

    save_joint_plot(all_train_acc, "Training Accuracy for All Layer Sizes", "train", "all_train_accuracy", args.layer_sizes)
    save_joint_plot(all_train_loss, "Training Loss for All Layer Sizes", "train", "all_train_loss", args.layer_sizes)
    save_joint_plot(all_val_acc, "Validation Accuracy for All Layer Sizes", "test", "all_val_accuracy", args.layer_sizes)
    save_joint_plot(all_val_loss, "Validation Loss for All Layer Sizes", "test", "all_val_loss", args.layer_sizes)
    
    save_combined_joint_plot(all_train_acc, all_val_acc, "Train and Validation Accuracy for All Layer Sizes", "joint", "all_accuracy", args.layer_sizes)
    save_combined_joint_plot(all_train_loss, all_val_loss, "Train and Validation Loss for All Layer Sizes", "joint", "all_loss", args.layer_sizes)
    
    results_data = {
        "all_train_acc": {str(size): accuracies for size, accuracies in zip(args.layer_sizes, all_train_acc)},
        "all_val_acc": {str(size): accuracies for size, accuracies in zip(args.layer_sizes, all_val_acc)},
        "all_train_loss": {str(size): losses for size, losses in zip(args.layer_sizes, all_train_loss)},
        "all_val_loss": {str(size): losses for size, losses in zip(args.layer_sizes, all_val_loss)}
    }

    with open('results/results_data.json', 'w') as json_file:
        json.dump(results_data, json_file, indent=4)

    print("Accuracy and loss data saved to results/results_data.json")
