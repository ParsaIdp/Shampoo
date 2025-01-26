import os
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Create results directory
os.makedirs('results', exist_ok=True)

from optimizer.shampoo import Shampoo


# Logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(input_dim, 1))
        self.bias = nn.Parameter(torch.randn(1))
        
    def forward(self, x):
        return torch.matmul(x, self.weights) + self.bias

def train_model(model, optimizer, train_loader, epochs=100, log_interval=10):
    criterion = nn.BCEWithLogitsLoss()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch.view(-1, 28*28)).squeeze()
            loss = criterion(outputs, y_batch)
            
            reg_loss = 0.01*(torch.norm(model.weights)**2 + torch.norm(model.bias)**2)
            loss += reg_loss
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        if epoch % log_interval == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    
    return losses


# Analysis functions
def save_analysis(losses, name):
    # Save plots
    plt.figure()
    plt.plot(losses)
    plt.title(f'{name} Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f'results/{name}_loss.png')
    plt.close()
    
    # Log-scale plot
    plt.figure()
    plt.semilogy(losses)
    plt.title(f'{name} Training Loss (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f'results/{name}_log_loss.png')
    plt.close()
    
    # Calculate slope
    valid_losses = [l for l in losses if not np.isinf(l)]
    x = np.arange(len(valid_losses))
    slope = np.polyfit(x, np.log(valid_losses), 1)[0]
    
    with open(f'results/{name}_slope.txt', 'w') as f:
        f.write(f"Convergence slope: {slope:.6f}")

# Main execution
if __name__ == '__main__':
    # Load and prepare dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((28, 28))
    ])
    
    full_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Filter 0s and 1s
    idx = (full_train.targets == 0) | (full_train.targets == 1)
    X = full_train.data[idx].float() / 255.0
    y = full_train.targets[idx].float()

    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=10000, shuffle=True)
    
    # Train with different optimizers
    optimizers = {
        'Shampoo': Shampoo,
        'SGD': torch.optim.SGD,
        'SGD_Momentum': lambda params: torch.optim.SGD(params, momentum=0.9, nesterov=True, lr=0.001)
    }
    
    results = {}
    
    for opt_name, opt_class in optimizers.items():
        print(f"\nTraining with {opt_name}...")
        
        model = LogisticRegression(28*28)
        if opt_name == 'Shampoo':
            optimizer = opt_class(model.parameters(), lr=0.00001, momentum=0.9)
        elif opt_name == 'SGD':
            optimizer = opt_class(model.parameters(), lr=0.001)
        else:
            optimizer = opt_class(model.parameters())
        
        losses = train_model(model, optimizer, train_loader= train_loader, epochs=100)
        
        start_time = time.time()
        training_time = time.time() - start_time
        
        results[opt_name] = {
            'losses': losses,
            'time': training_time
        }
        
        save_analysis(losses, opt_name)
        print(f"{opt_name} training completed in {training_time:.2f} seconds")
    
    # Generate comparison plot
    plt.figure(figsize=(10, 6))
    for name, data in results.items():
        plt.semilogy(data['losses'], label=name)
    
    plt.title('Optimizer Comparison (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('results/optimizer_comparison.png')
    plt.close()