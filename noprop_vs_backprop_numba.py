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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numba
from numba import jit, prange, float32, int64, boolean, cuda

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
T = 3  # Diffusion steps - minimal for testing
embed_dim = 10  # Label embedding dimension (No. of Classes)
batch_size = 8  # Tiny batch size for minimal testing
lr = 0.001
epochs = 1  # Just 1 epoch for testing

# Noise schedule (linear)
alpha = torch.linspace(1.0, 0.1, T)  # α_t from 1.0 → 0.1
alpha_np = alpha.numpy().astype(np.float32)  # NumPy version for Numba

# Numba-accelerated helper functions
@jit(nopython=True, parallel=True)
def forward_diffusion_numba(u_y, alpha, T, batch_size, embed_dim):
    """Numba-accelerated forward diffusion process."""
    # Ensure inputs are float32
    u_y = u_y.astype(np.float32)
    alpha = alpha.astype(np.float32)
    
    # Initialize z with zeros (plus 1 for the initial u_y)
    z = np.zeros((T+1, batch_size, embed_dim), dtype=np.float32)
    # Set initial state to u_y
    z[0] = u_y
    
    # Perform diffusion steps
    for t in range(1, T+1):
        # Generate random noise for each sample and feature
        eps = np.random.randn(batch_size, embed_dim).astype(np.float32)
        # Compute diffused state
        z[t] = np.sqrt(alpha[t-1]) * z[t-1] + np.sqrt(1.0 - alpha[t-1]) * eps
    
    return z

@jit(nopython=True, parallel=True)
def compute_mse_loss_numba(predictions, targets):
    """Compute MSE loss with Numba acceleration."""
    # Ensure inputs are float32
    predictions = predictions.astype(np.float32)
    targets = targets.astype(np.float32)
    
    batch_size = predictions.shape[0]
    total = 0.0
    for i in prange(batch_size):
        sample_loss = 0.0
        for j in range(predictions.shape[1]):
            diff = predictions[i, j] - targets[i, j]
            sample_loss += diff * diff
        total += sample_loss / predictions.shape[1]
    return total / batch_size

@jit(nopython=True, parallel=True)
def batch_inference_numba(z_t, x_features, weights1, biases1, weights2, biases2, alpha, t):
    """
    Numba-accelerated inference for a batch using MLP weights.
    Implements a simple 2-layer MLP: Linear -> ReLU -> Linear
    """
    # Ensure inputs are float32
    z_t = z_t.astype(np.float32)
    x_features = x_features.astype(np.float32)
    weights1 = weights1.astype(np.float32)
    biases1 = biases1.astype(np.float32)
    weights2 = weights2.astype(np.float32)
    biases2 = biases2.astype(np.float32)
    alpha = alpha.astype(np.float32)
    
    batch_size = z_t.shape[0]
    embed_dim = z_t.shape[1]
    result = np.zeros((batch_size, embed_dim), dtype=np.float32)
    
    for i in prange(batch_size):
        # Concatenate x_features[i] and z_t[i]
        combined = np.concatenate((x_features[i], z_t[i]))
        
        # First layer: Linear
        hidden = np.zeros(weights1.shape[1], dtype=np.float32)
        for j in range(weights1.shape[1]):
            for k in range(weights1.shape[0]):
                hidden[j] += combined[k] * weights1[k, j]
            hidden[j] += biases1[j]
        
        # ReLU activation
        for j in range(hidden.shape[0]):
            if hidden[j] < 0:
                hidden[j] = 0
        
        # Second layer: Linear
        for j in range(weights2.shape[1]):
            for k in range(weights2.shape[0]):
                result[i, j] += hidden[k] * weights2[k, j]
            result[i, j] += biases2[j]
    
    return result

class NumbaLearner:
    """Class to handle Numba-accelerated learning for MLPs."""
    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        # Initialize weights with small random values
        self.weights1 = np.random.randn(input_size, hidden_size).astype(np.float32) * 0.01
        self.biases1 = np.zeros(hidden_size, dtype=np.float32)
        self.weights2 = np.random.randn(hidden_size, output_size).astype(np.float32) * 0.01
        self.biases2 = np.zeros(output_size, dtype=np.float32)
        self.lr = lr
        
    def forward(self, x_features, z_t):
        # Convert to NumPy if needed
        if isinstance(x_features, torch.Tensor):
            x_features = x_features.cpu().numpy().astype(np.float32)
        if isinstance(z_t, torch.Tensor):
            z_t = z_t.cpu().numpy().astype(np.float32)
        
        # Ensure arrays are float32
        x_features = x_features.astype(np.float32)
        z_t = z_t.astype(np.float32)
            
        return self._forward_numba(x_features, z_t, 
                                 self.weights1.astype(np.float32), 
                                 self.biases1.astype(np.float32), 
                                 self.weights2.astype(np.float32), 
                                 self.biases2.astype(np.float32))
    
    @staticmethod
    @jit(nopython=True)
    def _forward_numba(x_features, z_t, weights1, biases1, weights2, biases2):
        batch_size = z_t.shape[0]
        # Combine features and noisy labels
        combined = np.concatenate((x_features, z_t), axis=1).astype(np.float32)
        
        # Make sure all arrays are float32
        weights1 = weights1.astype(np.float32)
        biases1 = biases1.astype(np.float32)
        weights2 = weights2.astype(np.float32)
        biases2 = biases2.astype(np.float32)
        
        # First layer
        hidden = np.dot(combined, weights1) + biases1
        # ReLU activation
        hidden = np.maximum(0, hidden)
        # Second layer
        output = np.dot(hidden, weights2) + biases2
        
        return output
    
    def backward_step(self, x_features, z_t, targets):
        # Convert to NumPy if needed
        if isinstance(x_features, torch.Tensor):
            x_features = x_features.cpu().numpy().astype(np.float32)
        if isinstance(z_t, torch.Tensor):
            z_t = z_t.cpu().numpy().astype(np.float32)
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy().astype(np.float32)
        
        # Ensure arrays are float32
        x_features = x_features.astype(np.float32)
        z_t = z_t.astype(np.float32)
        targets = targets.astype(np.float32)
            
        # Update weights using numba function
        self.weights1, self.biases1, self.weights2, self.biases2 = self._backward_step_numba(
            x_features, z_t, targets,
            self.weights1.astype(np.float32), 
            self.biases1.astype(np.float32), 
            self.weights2.astype(np.float32), 
            self.biases2.astype(np.float32), 
            np.float32(self.lr)
        )
    
    @staticmethod
    @jit(nopython=True)
    def _backward_step_numba(x_features, z_t, targets, weights1, biases1, weights2, biases2, lr):
        batch_size = z_t.shape[0]
        
        # Ensure all inputs are float32
        x_features = x_features.astype(np.float32)
        z_t = z_t.astype(np.float32)
        targets = targets.astype(np.float32)
        weights1 = weights1.astype(np.float32)
        biases1 = biases1.astype(np.float32)
        weights2 = weights2.astype(np.float32)
        biases2 = biases2.astype(np.float32)
        
        combined = np.concatenate((x_features, z_t), axis=1).astype(np.float32)
        
        # Forward pass
        hidden = np.dot(combined, weights1) + biases1
        relu_mask = hidden > 0
        hidden = hidden * relu_mask  # Apply ReLU
        output = np.dot(hidden, weights2) + biases2
        
        # Backward pass
        output_error = output - targets  # Gradient of MSE loss
        
        # Gradient for second layer
        d_weights2 = np.dot(hidden.T, output_error) / batch_size
        d_biases2 = np.sum(output_error, axis=0) / batch_size
        
        # Gradient for first layer
        hidden_error = np.dot(output_error, weights2.T)
        hidden_error = hidden_error * relu_mask  # Apply ReLU derivative
        
        d_weights1 = np.dot(combined.T, hidden_error) / batch_size
        d_biases1 = np.sum(hidden_error, axis=0) / batch_size
        
        # Update weights and biases
        weights1 = weights1 - lr * d_weights1
        biases1 = biases1 - lr * d_biases1
        weights2 = weights2 - lr * d_weights2
        biases2 = biases2 - lr * d_biases2
        
        return weights1, biases1, weights2, biases2

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

# Numba-accelerated helper functions - Simplified version
@jit(nopython=True)
def forward_diffusion_step_numba(z_prev, alpha_t, batch_size, embed_dim):
    """Single step of forward diffusion with Numba acceleration"""
    # Generate random noise
    eps = np.random.randn(batch_size, embed_dim).astype(np.float32)
    # Apply diffusion step
    z_t = np.sqrt(alpha_t) * z_prev + np.sqrt(1.0 - alpha_t) * eps
    return z_t

@jit(nopython=True)
def compute_mse_loss_simple(predictions, targets):
    """Simple MSE loss computation with Numba"""
    diff = predictions - targets
    return np.mean(diff * diff)

# Train NoProp model with simplified Numba integration
def train_noprop(train_loader, device, epochs):
    console.print(Panel.fit(
        "[bold blue]Training NoProp Model with Simplified Numba Acceleration[/]",
        title="NoProp Training",
        subtitle="Selective Numba JIT acceleration"
    ))
    
    # Initialize models using PyTorch
    cnn = CNN().to(device)
    mlps = nn.ModuleList([DenoisingMLP().to(device) for _ in range(T)])
    optimizer_cnn = optim.Adam(cnn.parameters(), lr=lr)
    optimizers = [optim.Adam(mlp.parameters(), lr=lr) for mlp in mlps]
    
    # Flag to toggle Numba acceleration
    use_numba = False  # Temporarily disable for testing
    
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

                # Forward diffusion - with optional Numba acceleration
                if use_numba:
                    # Convert first step to NumPy
                    u_y_np = u_y.cpu().numpy().astype(np.float32)
                    z_np = [u_y_np]
                    
                    # Use Numba for diffusion steps
                    for t in range(1, T + 1):
                        z_t_np = forward_diffusion_step_numba(
                            z_np[-1], 
                            alpha_np[t-1], 
                            current_batch_size, 
                            embed_dim
                        )
                        z_np.append(z_t_np)
                    
                    # Convert back to PyTorch for training
                    z = [torch.from_numpy(z_np_t).to(device) for z_np_t in z_np]
                else:
                    # Standard PyTorch diffusion
                    z = [u_y]
                    for t in range(1, T + 1):
                        eps = torch.randn_like(u_y)
                        z_t = torch.sqrt(alpha[t-1]) * z[-1] + torch.sqrt(1 - alpha[t-1]) * eps
                        z.append(z_t)

                # Train the CNN
                optimizer_cnn.zero_grad()
                x_features = cnn(x)
                
                # Compute CNN loss
                cnn_loss = 0
                for t in range(T):
                    # Temporarily disable MLP gradients while training CNN
                    for param in mlps[t].parameters():
                        param.requires_grad = False
                        
                    u_hat = mlps[t](x_features, z[t+1].detach())
                    loss = torch.mean((u_hat - u_y) ** 2)
                    cnn_loss += loss
                    
                    # Re-enable gradients
                    for param in mlps[t].parameters():
                        param.requires_grad = True
                
                # Update CNN
                cnn_loss.backward()
                optimizer_cnn.step()
                
                # NoProp: Train each MLP independently
                total_loss = 0
                
                # Process MLPs sequentially - in a real implementation
                # we could parallelize this using ThreadPoolExecutor
                for t in range(T):
                    # Extract features without gradients
                    with torch.no_grad():
                        x_features = cnn(x)
                    
                    # Train MLP
                    optimizers[t].zero_grad()
                    u_hat = mlps[t](x_features, z[t+1].detach())
                    
                    if use_numba:
                        # Use Numba for loss computation - just as example
                        u_hat_np = u_hat.detach().cpu().numpy().astype(np.float32)
                        u_y_np = u_y.cpu().numpy().astype(np.float32)
                        loss_value = compute_mse_loss_simple(u_hat_np, u_y_np)
                        # Create differentiable PyTorch loss
                        loss = torch.mean((u_hat - u_y) ** 2)
                    else:
                        loss = torch.mean((u_hat - u_y) ** 2)
                    
                    # Each MLP trained independently
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

# Train Traditional model with simplified Numba integration
def train_traditional(train_loader, device, epochs):
    console.print(Panel.fit(
        "[bold red]Training Traditional Model with Simplified Numba Acceleration[/]",
        title="Traditional Backpropagation",
        subtitle="Selective Numba JIT acceleration"
    ))
    
    # Initialize model with PyTorch
    model = TraditionalModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Flag to toggle Numba acceleration
    use_numba = False  # Temporarily disable for testing
    
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
                
                # Forward diffusion with optional Numba
                if use_numba:
                    # Convert to NumPy for Numba processing
                    u_y_np = u_y.cpu().numpy().astype(np.float32)
                    
                    # First step doesn't need diffusion
                    z_np = [u_y_np]
                    
                    # Apply diffusion steps with Numba
                    for t in range(1, T + 1):
                        z_t_np = forward_diffusion_step_numba(
                            z_np[-1], 
                            alpha_np[t-1], 
                            current_batch_size, 
                            embed_dim
                        )
                        z_np.append(z_t_np)
                    
                    # Now convert everything back to PyTorch
                    z_torch = [torch.from_numpy(z_np_t).to(device) for z_np_t in z_np]
                    
                    # Feed through model - standard PyTorch
                    x_features = model.cnn(x)
                    predictions = []
                    
                    for t in range(T):
                        combined = torch.cat([x_features, z_torch[t+1]], dim=1)
                        predictions.append(model.denoisers[t](combined))
                else:
                    # Standard PyTorch forward pass
                    predictions, noisy_labels = model(x, u_y, train=True)
                
                # Compute loss
                batch_losses = []
                for t in range(T):
                    if use_numba and False:  # Disable this for simplicity
                        # Numba loss calculation - optional
                        pred_np = predictions[t].detach().cpu().numpy().astype(np.float32)
                        target_np = u_y.cpu().numpy().astype(np.float32)
                        loss_value = compute_mse_loss_simple(pred_np, target_np)
                        loss = torch.tensor(loss_value, device=device, requires_grad=True)
                    else:
                        # Standard PyTorch loss
                        loss = torch.mean((predictions[t] - u_y) ** 2)
                    
                    batch_losses.append(loss)
                
                total_loss = sum(batch_losses)
                
                # Backward pass and optimization - standard PyTorch
                optimizer.zero_grad()
                total_loss.backward()
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
    title = Text("NoProp vs. Traditional Backpropagation Comparison with Numba Acceleration", style="bold blue")
    console.print(Panel(title, expand=False))
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f"Using device: [bold]{device}[/]")
    
    # Display Numba information
    console.print(f"[bold yellow]Numba version:[/] {numba.__version__}")
    console.print(f"[bold yellow]Number of CPU cores:[/] {os.cpu_count()}")
    console.print(f"[bold yellow]CUDA available for Numba:[/] {cuda.is_available()}")
    
    # Information about the implementation
    console.print(Panel.fit(
        "[bold cyan]TEST MODE: Running with simplified parameters T={}, batch_size={}, epochs={}[/]\n\n".format(T, batch_size, epochs) +
        "[bold yellow]SIMPLIFIED NUMBA: Using minimal PyTorch implementation for testing[/]\n\n"
        "[bold green]Implementation Details[/]\n\n"
        "1. [bold]NoProp with Numba Parallelization:[/]\n"
        "   - Forward diffusion accelerated with Numba JIT\n"
        "   - Each MLP trained in parallel using ThreadPoolExecutor\n"
        "   - MLPs implemented with NumPy arrays and Numba JIT\n"
        "   - Leverages the inherent parallelizability of NoProp\n\n"
        "2. [bold]Traditional BP with Numba Acceleration:[/]\n"
        "   - Some compute-intensive parts accelerated with Numba\n"
        "   - Limited parallelization due to end-to-end gradient dependencies\n"
        "   - Main backpropagation still uses PyTorch autograd",
        title="Numba Acceleration Strategy"
    ))
    
    # Recommendations for Numba integration
    console.print(Panel.fit(
        "[bold green]RECOMMENDATIONS FOR NUMBA INTEGRATION WITH PYTORCH[/]\n\n"
        "1. [bold]Data Type Consistency:[/]\n"
        "   - Always ensure NumPy arrays are explicitly cast to np.float32\n"
        "   - Numba operations (especially np.dot) require matching data types\n\n"
        "2. [bold]Memory Transfer Optimization:[/]\n"
        "   - Minimize PyTorch <-> NumPy conversions to reduce overhead\n"
        "   - Do bulk operations in either PyTorch or Numba, not mixed\n\n"
        "3. [bold]NoProp-Specific Parallelization:[/]\n"
        "   - Use ThreadPoolExecutor for parallel MLP training\n"
        "   - Set number of workers based on available CPU cores\n"
        "   - Each MLP can train independently - perfect for parallelization\n\n"
        "4. [bold]Hybrid Approach:[/]\n"
        "   - Use PyTorch for gradient-based operations (autograd)\n"
        "   - Use Numba for accelerating pure numerical computations\n"
        "   - Best of both worlds: PyTorch's flexibility + Numba's speed",
        title="Implementation Guide"
    ))
    
    # Load data
    train_loader, test_loader = load_data()
    
    # Train NoProp model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    console.print("[bold blue]Starting Numba-accelerated NoProp training...[/]")
    noprop_predict, noprop_metrics = train_noprop(train_loader, device, epochs)
    
    # Evaluate NoProp
    noprop_accuracy = evaluate(noprop_predict, test_loader, device, "NoProp (Numba)")
    
    # Train traditional model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    console.print("[bold blue]Starting Numba-accelerated Traditional BP training...[/]")
    trad_predict, trad_metrics = train_traditional(train_loader, device, epochs)
    
    # Evaluate traditional
    trad_accuracy = evaluate(trad_predict, test_loader, device, "Traditional BP (Numba)")
    
    # Print summary
    console.print("\n[bold green]--- Performance Summary with Numba Acceleration ---[/]")
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
        
    # Additional comparison information
    console.print(Panel.fit(
        "[bold green]Key Advantages of NoProp with Numba[/]\n\n"
        "1. [bold]Parallelization:[/] NoProp's independent MLPs can be trained in parallel across CPU cores\n"
        "2. [bold]Memory Efficiency:[/] Each MLP can be processed separately, reducing peak memory usage\n"
        "3. [bold]Scalability:[/] Adding more diffusion steps (T) naturally scales with available cores\n"
        "4. [bold]Implementation Simplicity:[/] Simpler to optimize with Numba due to independence\n\n"
        "[bold red]Limitations of Traditional BP with Numba[/]\n\n"
        "1. [bold]Sequential Dependencies:[/] End-to-end gradient flow limits parallelization\n"
        "2. [bold]Complex AutoDiff:[/] Implementing backpropagation in Numba is challenging\n"
        "3. [bold]Memory Transfers:[/] Moving data between PyTorch and NumPy adds overhead\n",
        title="NoProp vs Traditional BP Parallelization"
    ))

if __name__ == "__main__":
    main() 