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

# Dataset specific configurations
dataset_config = {
    'mnist': {
        'name': 'MNIST',
        'in_channels': 1,
        'image_size': 28,
        'num_classes': 10,
        'feature_dim': 12544,  # 7×7×256 after pooling and third conv
        'default_epochs': 3,
        'default_batch_size': 64,
        'default_lr': 0.001,
        'mlp_hidden_dim': 256
    },
    'cifar10': {
        'name': 'CIFAR10',
        'in_channels': 3,
        'image_size': 32,
        'num_classes': 10,
        'feature_dim': 16384,  # 8×8×256 after pooling and third conv
        'default_epochs': 15,  # More epochs for complex dataset
        'default_batch_size': 128,
        'default_lr': 0.0005,  # Slightly lower learning rate
        'mlp_hidden_dim': 512  # Larger hidden layer for more capacity
    },
    'cifar100': {
        'name': 'CIFAR100',
        'in_channels': 3,
        'image_size': 32,
        'num_classes': 100,
        'feature_dim': 16384,  # 8×8×256 after pooling and third conv
        'default_epochs': 20,  # Even more epochs for 100 classes
        'default_batch_size': 128,
        'default_lr': 0.0005,
        'mlp_hidden_dim': 1024  # Much larger for 100 classes
    }
}

# Enhanced CNN for image features - shared between both approaches
class CNN(nn.Module):
    def __init__(self, in_channels, feature_dim):
        super().__init__()
        # Calculate feature dimensions based on input size and channel count
        self.feature_dim = feature_dim
        
        # Enhanced architecture with more channels and batch normalization
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(in_channels, 64, 3, 1, padding=1),  # Increased from 32 to 64 channels
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Halves spatial dimensions
            
            # Second conv block
            nn.Conv2d(64, 128, 3, 1, padding=1),  # Increased from 64 to 128 channels
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Halves spatial dimensions again
            
            # Third conv block for more complex datasets
            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Flatten(),
            nn.Dropout(0.3),  # Regularization
            nn.Linear(feature_dim, 256),  # Adjusted for the calculated feature dimension
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        return self.features(x)

# Enhanced MLP for denoising - used in NoProp approach
class DenoisingMLP(nn.Module):
    def __init__(self, hidden_dim, embed_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(128 + embed_dim, hidden_dim),  # Input: image features + noisy label
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embed_dim)         # Output: denoised label
        )

    def forward(self, x_features, z_t):
        combined = torch.cat([x_features, z_t], dim=1)
        return self.mlp(combined)

# Enhanced Traditional model with end-to-end backpropagation
class TraditionalModel(nn.Module):
    def __init__(self, in_channels, feature_dim, hidden_dim, embed_dim, T, alpha):
        super().__init__()
        # CNN feature extractor
        self.cnn = CNN(in_channels, feature_dim)
        self.T = T
        self.alpha = alpha
        
        # Multiple denoising layers - equivalent to the NoProp approach
        self.denoisers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128 + embed_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, embed_dim)
            ) for _ in range(T)
        ])

    def forward(self, x, u_y=None, train=True):
        # Extract image features
        x_features = self.cnn(x)
        
        if not train:
            # Inference mode - start from noise and denoise step by step
            z_t = torch.randn(x.size(0), u_y.size(1), device=x.device)
            for t in reversed(range(self.T)):
                combined = torch.cat([x_features, z_t], dim=1)
                u_hat = self.denoisers[t](combined)
                z_t = torch.sqrt(self.alpha[t]) * u_hat + torch.sqrt(1 - self.alpha[t]) * torch.randn_like(u_hat)
            return z_t  # Final denoised prediction
        else:
            # Training mode - forward diffusion then denoising
            # Forward diffusion
            z = [u_y]  # Start with clean labels
            for t in range(1, self.T + 1):
                eps = torch.randn_like(u_y)
                z_t = torch.sqrt(self.alpha[t-1]) * z[-1] + torch.sqrt(1 - self.alpha[t-1]) * eps
                z.append(z_t)
            
            # Denoising predictions
            predictions = []
            for t in range(self.T):
                combined = torch.cat([x_features, z[t+1]], dim=1)
                predictions.append(self.denoisers[t](combined))
            
            return predictions, z[1:]  # Return predictions and noisy labels

# Load dataset based on selection
def load_data(dataset_name, batch_size, config):
    if dataset_name == 'mnist':
        transform = transforms.ToTensor()
        train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    else:  # CIFAR10 or CIFAR100
        # Enhanced transformations for CIFAR datasets
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # Data augmentation
            transforms.RandomHorizontalFlip(),     # Data augmentation
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        if dataset_name == 'cifar10':
            train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
            test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        else:  # cifar100
            train_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
            test_data = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    
    with console.status(f"[bold green]Loading {config['name']} dataset..."):
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    console.print(f"[bold green]✓[/] Loaded {len(train_data)} training and {len(test_data)} test examples from {config['name']}")
    return train_loader, test_loader

# Function to measure GPU memory usage
def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2  # Convert to MB
    return 0

# Function to plot metrics
def plot_metrics(noprop_metrics, trad_metrics, dataset_name, config, noprop_accuracy=None, trad_accuracy=None, save_dir="results"):
    if not VISUALIZATION_AVAILABLE:
        console.print("[bold red]Matplotlib and seaborn are not available for plotting.[/]")
        return
    
    # Set up the style
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Prepare data for DataFrame format (easier for seaborn)
    epochs_range = range(1, len(noprop_metrics['loss']) + 1)
    
    # Create a comprehensive dashboard with multiple plots
    fig = plt.figure(figsize=(16, 20))
    gs = fig.add_gridspec(5, 2, hspace=0.4, wspace=0.3)
    
    # 1. Loss comparison (larger plot spanning two columns)
    ax1 = fig.add_subplot(gs[0, :])
    sns.lineplot(x=list(epochs_range), y=noprop_metrics['loss'], label="NoProp", marker='o', ax=ax1, linewidth=2)
    sns.lineplot(x=list(epochs_range), y=trad_metrics['loss'], label="Traditional BP", marker='x', ax=ax1, linewidth=2)
    ax1.set_title(f'{config["name"]} - Training Loss Comparison', fontsize=16)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=12, loc='upper right')
    ax1.grid(True)
    
    # 2. Training time per epoch
    ax2 = fig.add_subplot(gs[1, 0])
    bars1 = ax2.bar(np.array(list(epochs_range)) - 0.2, noprop_metrics['time'], width=0.4, label="NoProp", color='blue', alpha=0.7)
    bars2 = ax2.bar(np.array(list(epochs_range)) + 0.2, trad_metrics['time'], width=0.4, label="Traditional BP", color='orange', alpha=0.7)
    ax2.set_title(f'Training Time per Epoch', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Add speedup as text on the bars
    for i, (t1, t2) in enumerate(zip(noprop_metrics['time'], trad_metrics['time'])):
        speedup = t2 / t1 if t1 > 0 else 0
        if speedup > 1:  # Only show speedup if NoProp is faster
            ax2.text(i+1, max(t1, t2) + 0.1, f"{speedup:.1f}x", 
                     ha='center', va='bottom', fontsize=10, rotation=0, 
                     bbox=dict(boxstyle="round,pad=0.3", fc='yellow', alpha=0.3))
    
    # 3. GPU Memory Usage
    ax3 = fig.add_subplot(gs[1, 1])
    sns.lineplot(x=list(epochs_range), y=noprop_metrics['gpu_memory'], label="NoProp", marker='o', ax=ax3)
    sns.lineplot(x=list(epochs_range), y=trad_metrics['gpu_memory'], label="Traditional BP", marker='x', ax=ax3)
    ax3.set_title(f'GPU Memory Usage', fontsize=14)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Memory (MB)', fontsize=12)
    ax3.legend(fontsize=11)
    ax3.grid(True)
    
    # 4. Memory savings percentage
    ax4 = fig.add_subplot(gs[2, 0])
    memory_savings = [(trad - noprop) / trad * 100 if trad > 0 else 0 
                      for noprop, trad in zip(noprop_metrics['gpu_memory'], trad_metrics['gpu_memory'])]
    ax4.bar(list(epochs_range), memory_savings, color='green', alpha=0.6)
    ax4.set_title(f'Memory Savings (%)', fontsize=14)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Memory Saved (%)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    # Add percentage text on bars
    for i, savings in enumerate(memory_savings):
        if savings > 0:
            ax4.text(i+1, savings + 1, f"{savings:.1f}%", ha='center', va='bottom', fontsize=10)
    
    # 5. Cumulative training time
    ax5 = fig.add_subplot(gs[2, 1])
    cum_noprop = np.cumsum(noprop_metrics['time'])
    cum_trad = np.cumsum(trad_metrics['time'])
    sns.lineplot(x=list(epochs_range), y=cum_noprop, label="NoProp", marker='o', ax=ax5)
    sns.lineplot(x=list(epochs_range), y=cum_trad, label="Traditional BP", marker='x', ax=ax5)
    ax5.set_title(f'Cumulative Training Time', fontsize=14)
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('Total Time (seconds)', fontsize=12)
    ax5.legend(fontsize=11)
    ax5.grid(True)
    
    # 6. Accuracy over time if available
    if 'accuracy' in noprop_metrics and 'accuracy' in trad_metrics:
        ax6 = fig.add_subplot(gs[3, 0])
        sns.lineplot(x=list(epochs_range), y=noprop_metrics['accuracy'], 
                    label="NoProp", marker='o', ax=ax6)
        sns.lineplot(x=list(epochs_range), y=trad_metrics['accuracy'], 
                    label="Traditional BP", marker='x', ax=ax6)
        ax6.set_title(f'Training Accuracy', fontsize=14)
        ax6.set_xlabel('Epoch', fontsize=12)
        ax6.set_ylabel('Accuracy', fontsize=12)
        ax6.legend(fontsize=11)
        ax6.grid(True)
    
    # 7. Final accuracy comparison (bar chart)
    if noprop_accuracy is not None and trad_accuracy is not None:
        ax7 = fig.add_subplot(gs[3, 1])
        models = ['NoProp', 'Traditional BP']
        accuracies = [noprop_accuracy, trad_accuracy]
        bars = ax7.bar(models, accuracies, color=['blue', 'orange'], alpha=0.7)
        ax7.set_title(f'Final Test Accuracy', fontsize=14)
        ax7.set_ylabel('Accuracy', fontsize=12)
        ax7.set_ylim(0, 1.0)  # Assuming accuracy is between 0 and 1
        # Add accuracy values on bars
        for bar in bars:
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=12)
        ax7.grid(True, alpha=0.3)
    
    # 8. Performance efficiency - Accuracy per time spent (bubble chart)
    if noprop_accuracy is not None and trad_accuracy is not None:
        ax8 = fig.add_subplot(gs[4, 0])
        total_time_noprop = sum(noprop_metrics['time'])
        total_time_trad = sum(trad_metrics['time'])
        efficiency_noprop = noprop_accuracy / total_time_noprop if total_time_noprop > 0 else 0
        efficiency_trad = trad_accuracy / total_time_trad if total_time_trad > 0 else 0
        
        # Create a bubble chart
        models = ['NoProp', 'Traditional BP']
        accuracies = [noprop_accuracy, trad_accuracy]
        times = [total_time_noprop, total_time_trad]
        efficiencies = [efficiency_noprop, efficiency_trad]
        colors = ['blue', 'orange']
        
        # Calculate relative bubble sizes (normalized by the max value)
        max_efficiency = max(efficiencies)
        sizes = [1000 * (e / max_efficiency) for e in efficiencies]
        
        # Plot bubbles
        scatter = ax8.scatter(times, accuracies, s=sizes, c=colors, alpha=0.6)
        
        # Add labels
        for i, model in enumerate(models):
            ax8.annotate(f"{model}\n({efficiencies[i]:.5f} acc/sec)", 
                        (times[i], accuracies[i]),
                        xytext=(10, 0), textcoords='offset points',
                        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))
        
        ax8.set_title('Efficiency: Accuracy vs. Training Time', fontsize=14)
        ax8.set_xlabel('Total Training Time (seconds)', fontsize=12)
        ax8.set_ylabel('Accuracy', fontsize=12)
        ax8.grid(True)
    
    # 9. Accuracy vs Memory tradeoff (if accuracy available)
    if noprop_accuracy is not None and trad_accuracy is not None:
        ax9 = fig.add_subplot(gs[4, 1])
        # Average memory usage
        avg_mem_noprop = sum(noprop_metrics['gpu_memory']) / len(noprop_metrics['gpu_memory'])
        avg_mem_trad = sum(trad_metrics['gpu_memory']) / len(trad_metrics['gpu_memory'])
        
        models = ['NoProp', 'Traditional BP']
        accuracies = [noprop_accuracy, trad_accuracy]
        memories = [avg_mem_noprop, avg_mem_trad]
        colors = ['blue', 'orange']
        
        # Plot scatter points
        scatter = ax9.scatter(memories, accuracies, c=colors, s=100, alpha=0.7)
        
        # Add labels
        for i, model in enumerate(models):
            ax9.annotate(model, (memories[i], accuracies[i]),
                        xytext=(10, 0), textcoords='offset points',
                        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))
        
        ax9.set_title('Accuracy vs. Memory Usage', fontsize=14)
        ax9.set_xlabel('Average Memory Usage (MB)', fontsize=12)
        ax9.set_ylabel('Accuracy', fontsize=12)
        ax9.grid(True)
    
    # Add overall title
    fig.suptitle(f'Comprehensive Performance Analysis - {config["name"]} Dataset', fontsize=20, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save the figure
    filename = f"{save_dir}/noprop_vs_backprop_{dataset_name}_dashboard.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    console.print(f"[bold green]Comprehensive dashboard saved to {filename}")
    
    # Create a separate accuracy trend plot if available (for tracking during training)
    if 'accuracy' in noprop_metrics and 'accuracy' in trad_metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, noprop_metrics['accuracy'], 'b-o', label='NoProp')
        plt.plot(epochs_range, trad_metrics['accuracy'], 'r-x', label='Traditional BP')
        plt.title(f'{config["name"]} - Training Accuracy', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.grid(True)
        plt.legend(fontsize=12)
        acc_filename = f"{save_dir}/noprop_vs_backprop_{dataset_name}_accuracy.png"
        plt.savefig(acc_filename, dpi=300)
        console.print(f"[bold green]Accuracy plot saved to {acc_filename}")

# Evaluate accuracy
def evaluate(predict_fn, test_loader, device, name, config):
    console.print(f"\n[bold yellow]Evaluating {name} model on {config['name']}...[/]")
    
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
    console.print(f"[bold green]{name} Accuracy on {config['name']}:[/] {accuracy:.4f}")
    return accuracy

# Print comparison metrics in a rich table
def print_metrics(noprop_metrics, trad_metrics, noprop_accuracy, trad_accuracy, config):
    console.print(f"\n[bold blue]Metrics Comparison for {config['name']}[/]")
    
    # Create table for loss
    table = Table(title=f"Loss by Epoch ({config['name']})")
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
    table = Table(title=f"Training Time (seconds) ({config['name']})")
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
    table = Table(title=f"GPU Memory Usage (MB) ({config['name']})")
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
    table = Table(title=f"Final Accuracy ({config['name']})")
    table.add_column("Model", justify="left", style="cyan")
    table.add_column("Accuracy", justify="right", style="green")
    
    table.add_row("NoProp", f"{noprop_accuracy:.4f}")
    table.add_row("Traditional BP", f"{trad_accuracy:.4f}")
    
    console.print(table)

# Update the evaluate function to collect accuracy over epochs
def evaluate_during_training(predict_fn, loader, device, name=None, silent=False):
    """Evaluate the model during training and return accuracy"""
    correct = 0
    total = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = predict_fn(x)
        correct += (pred == y).sum().item()
        total += y.size(0)
    
    accuracy = correct / total
    if not silent and name:
        console.print(f"[bold green]{name} Accuracy:[/] {accuracy:.4f}")
    return accuracy

# Modified train function to collect accuracy metrics
def collect_accuracy_metrics(predict_fn_noprop, predict_fn_trad, loader, device, epoch_metrics_noprop, epoch_metrics_trad):
    """Helper function to collect accuracy metrics for both models"""
    # Evaluate NoProp
    noprop_acc = evaluate_during_training(predict_fn_noprop, loader, device, silent=True)
    epoch_metrics_noprop.setdefault('accuracy', []).append(noprop_acc)
    
    # Evaluate Traditional BP
    trad_acc = evaluate_during_training(predict_fn_trad, loader, device, silent=True)
    epoch_metrics_trad.setdefault('accuracy', []).append(trad_acc)
    
    return noprop_acc, trad_acc 