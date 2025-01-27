import os
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from optimizer.shampoo import Shampoo  # Assuming Shampoo optimizer is available

# Create results directory
os.makedirs('results', exist_ok=True)

# Define MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Training function
def train_model(model, optimizer, train_loader, epochs=100, log_interval=10):
    criterion = nn.CrossEntropyLoss()
    losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
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

# Main execution
if __name__ == '__main__':
    # Load and prepare dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    
    # Define model parameters
    input_dim = 28 * 28
    hidden_dim = 128
    num_classes = 10
    epochs = 20

    # Optimizers to compare
    optimizers = {
        'SGD': lambda params: torch.optim.SGD(params, lr=0.01),
        'Adam': lambda params: torch.optim.Adam(params, lr=0.001),
        'Shampoo': lambda params: Shampoo(params, lr=0.01, momentum=0.9)
    }
    
    results = {}
    
    for opt_name, opt_fn in optimizers.items():
        print(f"\nTraining with {opt_name}...")
        
        # Create model and optimizer
        model = MLP(input_dim, hidden_dim, num_classes)
        optimizer = opt_fn(model.parameters())
        
        # Train model
        start_time = time.time()
        losses = train_model(model, optimizer, train_loader, epochs=epochs)
        training_time = time.time() - start_time
        
        # Save results
        results[opt_name] = {
            'losses': losses,
            'time': training_time
        }
        
        save_analysis(losses, opt_name)
        print(f"{opt_name} training completed in {training_time:.2f} seconds")
    
    # Generate comparison plot
    plt.figure(figsize=(10, 6))
    for name, data in results.items():
        plt.plot(data['losses'], label=name)
    
    plt.title('Optimizer Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('results/optimizer_comparison.png')
    plt.close()
