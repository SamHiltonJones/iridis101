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

if __name__ == '__main__':
    # Setup argument parser
    parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--device', default='cpu', type=str, help='Device on which to run: "cuda" or "cpu"')
    parser.add_argument('--epochs', default=100, type=int, help='total epochs to run')
    parser.add_argument('--layer-sizes', nargs='+', type=int, default=[1000, 10000, 50000, 100000, 300000, 500000], help='List of layer sizes')
    args = parser.parse_args()

    # Use the specified device
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

    all_train_accuracies = []
    all_test_accuracies = []
    all_train_losses = []
    all_test_losses = []

    for size in args.layer_sizes:
        model = BaselineModel(784, size, 10).to(device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())

        train_accuracies = []
        test_accuracies = []
        train_losses = []
        test_losses = []

        for epoch in range(args.epochs):
            print(f'Layer Size: {size}, Epoch: {epoch +1}')
            trial = Trial(model, optimizer, loss_function, metrics=['accuracy', 'loss']).to(device)
            trial.with_generators(trainloader, test_generator=testloader)
            trial.run(epochs=1)
            
            train_results = trial.evaluate(data_key=torchbearer.TRAIN_DATA)
            test_results = trial.evaluate(data_key=torchbearer.TEST_DATA)
            
            train_accuracies.append(train_results['train_acc'])
            test_accuracies.append(test_results['test_acc'])
            train_losses.append(train_results['train_loss'])
            test_losses.append(test_results['test_loss'])
            
        torch.save(model.state_dict(), f'results/model_weights_{size}.pth')

        all_train_accuracies.append(train_accuracies)
        all_test_accuracies.append(test_accuracies)
        all_train_losses.append(train_losses)
        all_test_losses.append(test_losses)

    for i, size in enumerate(args.layer_sizes):
        save_combined_plot(all_train_accuracies[i], all_test_accuracies[i], f"Accuracy for Layer Size {size}", "individual", f"{size}_accuracy")
        save_combined_plot(all_train_losses[i], all_test_losses[i], f"Loss for Layer Size {size}", "individual", f"{size}_loss")

    save_joint_plot(all_train_accuracies, "Training Accuracy for All Layer Sizes", "train", "all_accuracy", args.layer_sizes)
    save_joint_plot(all_train_losses, "Training Loss for All Layer Sizes", "train", "all_loss", args.layer_sizes)
    save_joint_plot(all_test_accuracies, "Testing Accuracy for All Layer Sizes", "test", "all_accuracy", args.layer_sizes)
    save_joint_plot(all_test_losses, "Testing Loss for All Layer Sizes", "test", "all_loss", args.layer_sizes)

    save_joint_plot(all_train_accuracies + all_test_accuracies, "Accuracy for All Layer Sizes", "joint", "all_accuracy", args.layer_sizes + args.layer_sizes)
    save_joint_plot(all_train_losses + all_test_losses, "Loss for All Layer Sizes", "joint", "all_loss", args.layer_sizes + args.layer_sizes)

    results_data = {
        "train_accuracies": {str(size): accuracies for size, accuracies in zip(args.layer_sizes, all_train_accuracies)},
        "test_accuracies": {str(size): accuracies for size, accuracies in zip(args.layer_sizes, all_test_accuracies)},
        "train_losses": {str(size): losses for size, losses in zip(args.layer_sizes, all_train_losses)},
        "test_losses": {str(size): losses for size, losses in zip(args.layer_sizes, all_test_losses)}
    }

    with open('results/results_data.json', 'w') as json_file:
        json.dump(results_data, json_file, indent=4)

    print("Accuracy and loss data saved to results/results_data.json")
