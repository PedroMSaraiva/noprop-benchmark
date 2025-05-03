import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import numpy as np
import gc
import os
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.text import Text

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

# Hyperparameters
T = 10  # Diffusion steps
embed_dim = 10  # Label embedding dimension (No. of Classes)
batch_size = 64  # Reduced batch size for faster testing
lr = 0.001
epochs = 3  # Reduced for quicker comparison

# Noise schedule (linear)
alpha = torch.linspace(1.0, 0.1, T)  # α_t from 1.0 → 0.1

# CNN for image features - shared between both approaches
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1600, 128)  # MNIST: 28x28 → 1600-dim
        )

    def forward(self, x):
        return self.features(x)

# MLP for denoising - used in NoProp approach
class DenoisingMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(128 + embed_dim, 256),  # Input: image features + noisy label
            nn.ReLU(),
            nn.Linear(256, embed_dim)         # Output: denoised label
        )

    def forward(self, x_features, z_t):
        combined = torch.cat([x_features, z_t], dim=1)
        return self.mlp(combined)

# Traditional model with end-to-end backpropagation
class TraditionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN feature extractor
        self.cnn = CNN()
        
        # Multiple denoising layers - equivalent to the NoProp approach
        self.denoisers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128 + embed_dim, 256),
                nn.ReLU(),
                nn.Linear(256, embed_dim)
            ) for _ in range(T)
        ])

    def forward(self, x, u_y=None, train=True):
        # Extract image features
        x_features = self.cnn(x)
        
        if not train:
            # Inference mode - start from noise and denoise step by step
            z_t = torch.randn(x.size(0), embed_dim, device=x.device)
            for t in reversed(range(T)):
                combined = torch.cat([x_features, z_t], dim=1)
                u_hat = self.denoisers[t](combined)
                z_t = torch.sqrt(alpha[t]) * u_hat + torch.sqrt(1 - alpha[t]) * torch.randn_like(u_hat)
            return z_t  # Final denoised prediction
        else:
            # Training mode - forward diffusion then denoising
            # Forward diffusion
            z = [u_y]  # Start with clean labels
            for t in range(1, T + 1):
                eps = torch.randn_like(u_y)
                z_t = torch.sqrt(alpha[t-1]) * z[-1] + torch.sqrt(1 - alpha[t-1]) * eps
                z.append(z_t)
            
            # Denoising predictions
            predictions = []
            for t in range(T):
                combined = torch.cat([x_features, z[t+1]], dim=1)
                predictions.append(self.denoisers[t](combined))
            
            return predictions, z[1:]  # Return predictions and noisy labels

# Load MNIST
def load_data():
    transform = transforms.ToTensor()
    with console.status("[bold green]Loading datasets..."):
        train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    console.print(f"[bold green]✓[/] Loaded {len(train_data)} training and {len(test_data)} test examples")
    return train_loader, test_loader

# Function to measure GPU memory usage
def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2  # Convert to MB
    return 0

# Function to plot metrics
def plot_metrics(noprop_metrics, trad_metrics, save_dir="results"):
    if not VISUALIZATION_AVAILABLE:
        console.print("[bold red]Matplotlib and seaborn are not available for plotting.[/]")
        return
    
    # Set up the style
    sns.set(style="whitegrid")
    
    # Prepare data for DataFrame format (easier for seaborn)
    epochs_range = range(1, len(noprop_metrics['loss']) + 1)
    
    # Create figure and subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))
    
    # Plot 1: Loss comparison
    ax = axs[0]
    sns.lineplot(x=epochs_range, y=noprop_metrics['loss'], label="NoProp", marker='o', ax=ax)
    sns.lineplot(x=epochs_range, y=trad_metrics['loss'], label="Traditional BP", marker='x', ax=ax)
    ax.set_title('Training Loss Comparison', fontsize=16)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True)
    
    # Plot 2: Training time per epoch
    ax = axs[1]
    sns.barplot(x=list(epochs_range), y=noprop_metrics['time'], label="NoProp", color='blue', alpha=0.7, ax=ax)
    sns.barplot(x=list(epochs_range), y=trad_metrics['time'], label="Traditional BP", color='orange', alpha=0.7, ax=ax)
    ax.set_title('Training Time per Epoch', fontsize=16)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    # Create a custom legend
    ax.legend(handles=[
        plt.Rectangle((0,0),1,1, color='blue', alpha=0.7),
        plt.Rectangle((0,0),1,1, color='orange', alpha=0.7)
    ], labels=['NoProp', 'Traditional BP'], fontsize=12)
    ax.grid(True)
    
    # Plot 3: GPU Memory Usage
    ax = axs[2]
    sns.lineplot(x=epochs_range, y=noprop_metrics['gpu_memory'], label="NoProp", marker='o', ax=ax)
    sns.lineplot(x=epochs_range, y=trad_metrics['gpu_memory'], label="Traditional BP", marker='x', ax=ax)
    ax.set_title('GPU Memory Usage', fontsize=16)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Memory (MB)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/noprop_vs_backprop_comparison.png", dpi=300, bbox_inches='tight')
    console.print(f"[bold green]Plots saved to {save_dir}/noprop_vs_backprop_comparison.png")

# Train NoProp model
def train_noprop(train_loader, device, epochs):
    console.print(Panel.fit(
        "[bold blue]Training NoProp Model[/]",
        title="NoProp Training",
        subtitle="Each layer trained independently"
    ))
    
    # Initialize models
    cnn = CNN().to(device)
    mlps = nn.ModuleList([DenoisingMLP().to(device) for _ in range(T)])
    optimizer_cnn = optim.Adam(cnn.parameters(), lr=lr)
    optimizers = [optim.Adam(mlp.parameters(), lr=lr) for mlp in mlps]
    
    # Metrics tracking
    losses = []
    times = []
    gpu_mem = []
    
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
            
            progress.remove_task(batch_task)
            progress.update(train_task, advance=1)
            
            # Display epoch summary
            console.print(f"[bold]Epoch {epoch+1}/{epochs}[/] | Loss: [blue]{avg_loss:.4f}[/] | "
                        f"Time: [yellow]{epoch_time:.2f}s[/] | GPU: [magenta]{get_gpu_memory():.2f}MB[/]")
    
    # Test function for NoProp
    def predict(x):
        x = x.to(device)
        z_t = torch.randn(x.size(0), embed_dim, device=device)  # Start from noise
        with torch.no_grad():
            x_features = cnn(x)
            for t in reversed(range(T)):
                u_hat = mlps[t](x_features, z_t)
                z_t = torch.sqrt(alpha[t]) * u_hat + torch.sqrt(1 - alpha[t]) * torch.randn_like(u_hat)
        return torch.argmax(z_t, dim=1)  # Final prediction
    
    return predict, {'loss': losses, 'time': times, 'gpu_memory': gpu_mem}

# Train Traditional model (with backpropagation)
def train_traditional(train_loader, device, epochs):
    console.print(Panel.fit(
        "[bold red]Training Traditional Model[/]",
        title="Traditional Backpropagation",
        subtitle="End-to-end gradient flow"
    ))
    
    # Initialize model
    model = TraditionalModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Metrics tracking
    losses = []
    times = []
    gpu_mem = []
    
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
            
            progress.remove_task(batch_task)
            progress.update(train_task, advance=1)
            
            # Display epoch summary
            console.print(f"[bold]Epoch {epoch+1}/{epochs}[/] | Loss: [blue]{avg_loss:.4f}[/] | "
                        f"Time: [yellow]{epoch_time:.2f}s[/] | GPU: [magenta]{get_gpu_memory():.2f}MB[/]")
    
    # Test function
    def predict(x):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x, train=False)
        return torch.argmax(pred, dim=1)
    
    return predict, {'loss': losses, 'time': times, 'gpu_memory': gpu_mem}

# Evaluate accuracy
def evaluate(predict_fn, test_loader, device, name):
    console.print(f"\n[bold yellow]Evaluating {name} model...[/]")
    
    correct = 0
    total = 0
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        eval_task = progress.add_task("[cyan]Evaluating...", total=len(test_loader))
        
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = predict_fn(x)
            correct += (pred == y).sum().item()
            total += y.size(0)
            progress.update(eval_task, advance=1)
    
    accuracy = correct / total
    console.print(f"[bold green]{name} Accuracy:[/] {accuracy:.4f}")
    return accuracy

# Print comparison metrics in a rich table
def print_metrics(noprop_metrics, trad_metrics, noprop_accuracy, trad_accuracy):
    console.print("\n[bold blue]Metrics Comparison[/]")
    
    # Create table for loss
    table = Table(title="Loss by Epoch")
    table.add_column("Epoch", justify="center", style="cyan")
    table.add_column("NoProp", justify="right", style="green")
    table.add_column("Traditional BP", justify="right", style="red")
    
    for i in range(len(noprop_metrics['loss'])):
        table.add_row(
            f"{i+1}",
            f"{noprop_metrics['loss'][i]:.4f}",
            f"{trad_metrics['loss'][i]:.4f}"
        )
    
    console.print(table)
    
    # Create table for time
    table = Table(title="Training Time (seconds)")
    table.add_column("Epoch", justify="center", style="cyan")
    table.add_column("NoProp", justify="right", style="green")
    table.add_column("Traditional BP", justify="right", style="red")
    table.add_column("NoProp Speedup", justify="right", style="yellow")
    
    for i in range(len(noprop_metrics['time'])):
        speedup = trad_metrics['time'][i] / noprop_metrics['time'][i] if noprop_metrics['time'][i] > 0 else 0
        table.add_row(
            f"{i+1}",
            f"{noprop_metrics['time'][i]:.2f}s",
            f"{trad_metrics['time'][i]:.2f}s",
            f"{speedup:.2f}x"
        )
    
    # Add total time row
    total_noprop = sum(noprop_metrics['time'])
    total_trad = sum(trad_metrics['time'])
    total_speedup = total_trad / total_noprop if total_noprop > 0 else 0
    table.add_row(
        "Total",
        f"{total_noprop:.2f}s",
        f"{total_trad:.2f}s",
        f"{total_speedup:.2f}x",
        style="bold"
    )
    
    console.print(table)
    
    # Create table for memory
    table = Table(title="GPU Memory Usage (MB)")
    table.add_column("Epoch", justify="center", style="cyan")
    table.add_column("NoProp", justify="right", style="green")
    table.add_column("Traditional BP", justify="right", style="red")
    table.add_column("Memory Savings", justify="right", style="yellow")
    
    for i in range(len(noprop_metrics['gpu_memory'])):
        mem_diff = trad_metrics['gpu_memory'][i] - noprop_metrics['gpu_memory'][i]
        mem_savings = f"{mem_diff:.2f}MB ({mem_diff/trad_metrics['gpu_memory'][i]*100:.1f}%)" if trad_metrics['gpu_memory'][i] > 0 else "N/A"
        table.add_row(
            f"{i+1}",
            f"{noprop_metrics['gpu_memory'][i]:.2f}MB",
            f"{trad_metrics['gpu_memory'][i]:.2f}MB",
            mem_savings
        )
    
    console.print(table)
    
    # Final accuracy table
    table = Table(title="Final Accuracy")
    table.add_column("Model", justify="left", style="cyan")
    table.add_column("Accuracy", justify="right", style="green")
    
    table.add_row("NoProp", f"{noprop_accuracy:.4f}")
    table.add_row("Traditional BP", f"{trad_accuracy:.4f}")
    
    console.print(table)

# Main function
def main():
    # Display the title
    title = Text("NoProp vs. Traditional Backpropagation Comparison", style="bold blue")
    console.print(Panel(title, expand=False))
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f"Using device: [bold]{device}[/]")
    
    # Load data
    train_loader, test_loader = load_data()
    
    # Train NoProp model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    noprop_predict, noprop_metrics = train_noprop(train_loader, device, epochs)
    
    # Evaluate NoProp
    noprop_accuracy = evaluate(noprop_predict, test_loader, device, "NoProp")
    
    # Train traditional model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    trad_predict, trad_metrics = train_traditional(train_loader, device, epochs)
    
    # Evaluate traditional
    trad_accuracy = evaluate(trad_predict, test_loader, device, "Traditional BP")
    
    # Print summary
    console.print("\n[bold green]--- Performance Summary ---[/]")
    console.print(f"NoProp - Final Loss: [bold]{noprop_metrics['loss'][-1]:.4f}[/], "
                 f"Total Time: [bold]{sum(noprop_metrics['time']):.2f}s[/], "
                 f"Accuracy: [bold]{noprop_accuracy:.4f}[/]")
    console.print(f"Traditional BP - Final Loss: [bold]{trad_metrics['loss'][-1]:.4f}[/], "
                 f"Total Time: [bold]{sum(trad_metrics['time']):.2f}s[/], "
                 f"Accuracy: [bold]{trad_accuracy:.4f}[/]")
    
    # Print detailed metrics
    print_metrics(noprop_metrics, trad_metrics, noprop_accuracy, trad_accuracy)
    
    # Create plots if visualization is available
    if VISUALIZATION_AVAILABLE:
        plot_metrics(noprop_metrics, trad_metrics)
    else:
        console.print("[bold yellow]Visualization libraries (matplotlib, seaborn) not available. Skipping plots.[/]")

if __name__ == "__main__":
    main() 