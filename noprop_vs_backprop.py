import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import numpy as np
import gc
import os
import argparse
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.text import Text
from utils import (
    console, dataset_config, CNN, DenoisingMLP, TraditionalModel,
    load_data, get_gpu_memory, plot_metrics, evaluate, print_metrics,
    evaluate_during_training
)

# Check if visualization libraries are available
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Initialize rich console
console = Console()

# Create results directory if not exists
if not os.path.exists("results"):
    os.makedirs("results")

# Command-line arguments for dataset selection
parser = argparse.ArgumentParser(description='NoProp vs Traditional Backpropagation with different datasets')
parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10', 'cifar100'],
                    help='Dataset to use (mnist, cifar10, or cifar100)')
parser.add_argument('--epochs', type=int, default=None, help='Number of epochs to train')
parser.add_argument('--batch-size', type=int, default=None, help='Batch size for training')
parser.add_argument('--lr', type=float, default=None, help='Learning rate')

# Parse arguments
args = parser.parse_args()

# Set hyperparameters based on selected dataset
dataset_name = args.dataset
config = dataset_config[dataset_name]

# Use command line args if provided, otherwise use dataset defaults
epochs = args.epochs if args.epochs is not None else config['default_epochs']
batch_size = args.batch_size if args.batch_size is not None else config['default_batch_size']
lr = args.lr if args.lr is not None else config['default_lr']

# Hyperparameters
T = 10  # Diffusion steps
embed_dim = config['num_classes']  # Label embedding dimension (No. of Classes)

# Noise schedule (linear)
alpha = torch.linspace(1.0, 0.1, T)  # α_t from 1.0 → 0.1

# Train NoProp model
def train_noprop(train_loader, device, epochs):
    console.print(Panel.fit(
        f"[bold blue]Training NoProp Model on {config['name']}[/]",
        title="NoProp Training",
        subtitle="Each layer trained independently"
    ))
    
    # Initialize models
    cnn = CNN(config['in_channels'], config['feature_dim']).to(device)
    mlps = nn.ModuleList([DenoisingMLP(config['mlp_hidden_dim'], embed_dim).to(device) for _ in range(T)])
    optimizer_cnn = optim.Adam(cnn.parameters(), lr=lr, weight_decay=1e-5)  # Added weight decay
    optimizers = [optim.Adam(mlp.parameters(), lr=lr, weight_decay=1e-5) for mlp in mlps]
    
    # Learning rate scheduler for better convergence
    scheduler_cnn = optim.lr_scheduler.ReduceLROnPlateau(optimizer_cnn, 'min', patience=2, factor=0.5)
    schedulers = [optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=2, factor=0.5) for opt in optimizers]
    
    # Metrics tracking
    losses = []
    times = []
    gpu_mem = []
    accuracy = []
    
    # Create a small validation set for tracking accuracy during training
    # We'll use a subset of the training data for simplicity
    val_size = min(len(train_loader.dataset) // 10, 1000)  # 10% or max 1000 samples
    val_indices = torch.randperm(len(train_loader.dataset))[:val_size]
    val_dataset = torch.utils.data.Subset(train_loader.dataset, val_indices)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Progress tracking
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        train_task = progress.add_task("[green]Training...", total=epochs)
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            start_time = time.time()
            
            batch_task = progress.add_task(f"[cyan]Epoch {epoch+1}/{epochs}", total=len(train_loader))
            
            # Set to training mode
            cnn.train()
            for mlp in mlps:
                mlp.train()
            
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                current_batch_size = x.shape[0]
                u_y = torch.zeros(current_batch_size, embed_dim, device=device).scatter_(1, y.unsqueeze(1), 1)

                # Forward diffusion (Adding Noise for each 'T')
                z = [u_y]
                for t in range(1, T + 1):
                    eps = torch.randn_like(u_y)
                    z_t = torch.sqrt(alpha[t-1]) * z[-1] + torch.sqrt(1 - alpha[t-1]) * eps
                    z.append(z_t)

                # First, train the CNN
                optimizer_cnn.zero_grad()
                x_features = cnn(x)
                
                # Compute CNN loss - FIXED to ensure gradient computation works properly
                cnn_loss = 0
                for t in range(T):
                    # Detach the MLP parameters but allow gradients to flow through x_features
                    for param in mlps[t].parameters():
                        param.requires_grad = False
                        
                    # Forward pass with gradients tracked for CNN
                    u_hat = mlps[t](x_features, z[t+1].detach())
                    loss = torch.mean((u_hat - u_y) ** 2)
                    cnn_loss += loss
                    
                    # Re-enable gradients for MLP parameters
                    for param in mlps[t].parameters():
                        param.requires_grad = True
                
                # Update CNN
                cnn_loss.backward()  # Backprop affects only CNN since MLP params were disabled
                optimizer_cnn.step()
                
                # Now train each MLP independently - THIS IS THE KEY NOPROP CONCEPT
                total_loss = 0
                for t in range(T):
                    # Extract features without gradients
                    with torch.no_grad():
                        x_features = cnn(x)
                    
                    # Forward for this specific MLP 
                    optimizers[t].zero_grad()
                    u_hat = mlps[t](x_features, z[t+1].detach())
                    loss = torch.mean((u_hat - u_y) ** 2)
                    
                    # Each MLP is trained independently - no cross-MLP gradient flow
                    loss.backward()
                    optimizers[t].step()
                    
                    total_loss += loss.item()

                epoch_loss += total_loss
                batch_count += 1
                progress.update(batch_task, advance=1)

            # Collect metrics
            epoch_time = time.time() - start_time
            avg_loss = epoch_loss / batch_count
            losses.append(avg_loss)
            times.append(epoch_time)
            gpu_mem.append(get_gpu_memory())
            
            # Update learning rate schedulers
            scheduler_cnn.step(avg_loss)
            for scheduler in schedulers:
                scheduler.step(avg_loss)
            
            # Evaluate accuracy on validation set during training
            # Test function for current NoProp state
            def current_predict(x):
                # Set models to eval mode
                cnn.eval()
                for mlp in mlps:
                    mlp.eval()
                
                x = x.to(device)
                z_t = torch.randn(x.size(0), embed_dim, device=device)  # Start from noise
                with torch.no_grad():
                    x_features = cnn(x)
                    for t in reversed(range(T)):
                        u_hat = mlps[t](x_features, z_t)
                        z_t = torch.sqrt(alpha[t]) * u_hat + torch.sqrt(1 - alpha[t]) * torch.randn_like(u_hat)
                return torch.argmax(z_t, dim=1)  # Final prediction
            
            # Evaluate on validation set
            val_acc = evaluate_during_training(current_predict, val_loader, device, silent=True)
            accuracy.append(val_acc)
            
            progress.remove_task(batch_task)
            progress.update(train_task, advance=1)
            
            # Display epoch summary
            current_lr = optimizer_cnn.param_groups[0]['lr']
            console.print(f"[bold]Epoch {epoch+1}/{epochs}[/] | Loss: [blue]{avg_loss:.4f}[/] | Val Acc: [green]{val_acc:.4f}[/] | "
                        f"Time: [yellow]{epoch_time:.2f}s[/] | GPU: [magenta]{get_gpu_memory():.2f}MB[/] | "
                        f"LR: [cyan]{current_lr:.6f}[/]")
    
    # Final test function for NoProp
    def predict(x):
        # Set to evaluation mode
        cnn.eval()
        for mlp in mlps:
            mlp.eval()
            
        x = x.to(device)
        z_t = torch.randn(x.size(0), embed_dim, device=device)  # Start from noise
        with torch.no_grad():
            x_features = cnn(x)
            for t in reversed(range(T)):
                u_hat = mlps[t](x_features, z_t)
                z_t = torch.sqrt(alpha[t]) * u_hat + torch.sqrt(1 - alpha[t]) * torch.randn_like(u_hat)
        return torch.argmax(z_t, dim=1)  # Final prediction
    
    return predict, {'loss': losses, 'time': times, 'gpu_memory': gpu_mem, 'accuracy': accuracy}

# Train Traditional model (with backpropagation)
def train_traditional(train_loader, device, epochs):
    console.print(Panel.fit(
        f"[bold red]Training Traditional Model on {config['name']}[/]",
        title="Traditional Backpropagation",
        subtitle="End-to-end gradient flow"
    ))
    
    # Initialize model
    model = TraditionalModel(config['in_channels'], config['feature_dim'], 
                             config['mlp_hidden_dim'], embed_dim, T, alpha).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # Added weight decay
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    # Metrics tracking
    losses = []
    times = []
    gpu_mem = []
    accuracy = []
    
    # Create a small validation set for tracking accuracy during training
    # We'll use a subset of the training data for simplicity
    val_size = min(len(train_loader.dataset) // 10, 1000)  # 10% or max 1000 samples
    val_indices = torch.randperm(len(train_loader.dataset))[:val_size]
    val_dataset = torch.utils.data.Subset(train_loader.dataset, val_indices)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Progress tracking
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        train_task = progress.add_task("[green]Training...", total=epochs)
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            start_time = time.time()
            
            batch_task = progress.add_task(f"[cyan]Epoch {epoch+1}/{epochs}", total=len(train_loader))
            
            # Set to training mode
            model.train()
            
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                current_batch_size = x.shape[0]
                u_y = torch.zeros(current_batch_size, embed_dim, device=device).scatter_(1, y.unsqueeze(1), 1)
                
                # Forward pass
                predictions, noisy_labels = model(x, u_y, train=True)
                
                # Compute loss - similar to NoProp for fair comparison
                batch_losses = []
                for t in range(T):
                    loss = torch.mean((predictions[t] - u_y) ** 2)
                    batch_losses.append(loss)
                
                total_loss = sum(batch_losses)
                
                # Backward pass and optimization - this is end-to-end backprop
                optimizer.zero_grad()
                total_loss.backward()  # Gradients flow through the entire network
                optimizer.step()
                
                epoch_loss += total_loss.item()
                batch_count += 1
                progress.update(batch_task, advance=1)
            
            # Collect metrics
            epoch_time = time.time() - start_time
            avg_loss = epoch_loss / batch_count
            losses.append(avg_loss)
            times.append(epoch_time)
            gpu_mem.append(get_gpu_memory())
            
            # Update learning rate scheduler
            scheduler.step(avg_loss)
            
            # Evaluate accuracy on validation set during training
            # Test function for current traditional model state
            def current_predict(x):
                model.eval()
                x = x.to(device)
                with torch.no_grad():
                    pred = model(x, u_y, train=False)
                return torch.argmax(pred, dim=1)
            
            # Evaluate on validation set
            val_acc = evaluate_during_training(current_predict, val_loader, device, silent=True)
            accuracy.append(val_acc)
            
            progress.remove_task(batch_task)
            progress.update(train_task, advance=1)
            
            # Display epoch summary
            current_lr = optimizer.param_groups[0]['lr']
            console.print(f"[bold]Epoch {epoch+1}/{epochs}[/] | Loss: [blue]{avg_loss:.4f}[/] | Val Acc: [green]{val_acc:.4f}[/] | "
                        f"Time: [yellow]{epoch_time:.2f}s[/] | GPU: [magenta]{get_gpu_memory():.2f}MB[/] | "
                        f"LR: [cyan]{current_lr:.6f}[/]")
    
    # Test function
    def predict(x):
        # Set to evaluation mode
        model.eval()
        
        x = x.to(device)
        with torch.no_grad():
            pred = model(x, u_y, train=False)
        return torch.argmax(pred, dim=1)
    
    return predict, {'loss': losses, 'time': times, 'gpu_memory': gpu_mem, 'accuracy': accuracy}

# Main function
def main():
    # Display the title
    title = Text(f"NoProp vs. Traditional Backpropagation Comparison on {config['name']}", style="bold blue")
    console.print(Panel(title, expand=False))
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f"Using device: [bold]{device}[/]")
    
    # Display dataset info
    console.print(f"[bold cyan]Dataset:[/] {config['name']}")
    console.print(f"[bold cyan]Number of classes:[/] {config['num_classes']}")
    console.print(f"[bold cyan]Image size:[/] {config['image_size']}x{config['image_size']} with {config['in_channels']} channels")
    console.print(f"[bold cyan]Training parameters:[/] epochs={epochs}, batch_size={batch_size}, learning_rate={lr}")
    console.print(f"[bold cyan]Model size:[/] CNN feature dimension={config['feature_dim']}, MLP hidden dimension={config['mlp_hidden_dim']}")
    
    # Load data
    train_loader, test_loader = load_data(dataset_name, batch_size, config)
    
    # Train NoProp model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    noprop_predict, noprop_metrics = train_noprop(train_loader, device, epochs)
    
    # Evaluate NoProp
    noprop_accuracy = evaluate(noprop_predict, test_loader, device, "NoProp", config)
    
    # Train traditional model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    trad_predict, trad_metrics = train_traditional(train_loader, device, epochs)
    
    # Evaluate traditional
    trad_accuracy = evaluate(trad_predict, test_loader, device, "Traditional BP", config)
    
    # Print summary
    console.print(f"\n[bold green]--- Performance Summary for {config['name']} ---[/]")
    console.print(f"NoProp - Final Loss: [bold]{noprop_metrics['loss'][-1]:.4f}[/], "
                 f"Total Time: [bold]{sum(noprop_metrics['time']):.2f}s[/], "
                 f"Accuracy: [bold]{noprop_accuracy:.4f}[/]")
    console.print(f"Traditional BP - Final Loss: [bold]{trad_metrics['loss'][-1]:.4f}[/], "
                 f"Total Time: [bold]{sum(trad_metrics['time']):.2f}s[/], "
                 f"Accuracy: [bold]{trad_accuracy:.4f}[/]")
    
    # Print detailed metrics
    print_metrics(noprop_metrics, trad_metrics, noprop_accuracy, trad_accuracy, config)
    
    # Create plots with enhanced visualization
    plot_metrics(noprop_metrics, trad_metrics, dataset_name, config, noprop_accuracy, trad_accuracy)

if __name__ == "__main__":
    main() 