# NoProp vs. Traditional Backpropagation Comparison

This project compares two neural network training approaches:

1. **NoProp** - A novel approach where layers are trained independently without backpropagating gradients across the entire network.
2. **Traditional Backpropagation** - The conventional end-to-end gradient-based training.

## Background

NoProp (No Backpropagation) is a training paradigm that avoids end-to-end backpropagation through a neural network. Instead, it uses a diffusion-based approach where each layer is trained independently to denoise progressively noisy inputs.

The key advantages of NoProp include:

- **Parallelizable training** - Each layer can be trained independently, enabling parallel computation
- **Reduced memory usage** - No need to store the entire computational graph
- **Mitigated vanishing/exploding gradients** - Gradients don't flow through the entire network
- **Potential for distributed training** - Layers can be trained on separate machines

## How NoProp Works

NoProp's training process involves:

1. **Forward diffusion** - Adding progressively more noise to clean labels
2. **Independent layer training** - Each layer learns to predict clean labels from noisy ones
3. **Inference by denoising** - At test time, starting from pure noise and progressively denoising

The process in more detail:

1. Convert one-hot labels to progressively noisier versions (z₁, z₂, ..., zₜ)
2. Train each denoising MLP to predict clean labels from its corresponding noise level
3. Crucially, during backpropagation, gradients don't flow between MLPs

At inference time, we start with random noise and sequentially apply the denoising MLPs in reverse order.

## Implementation Details

The comparison script (`noprop_vs_backprop.py`) performs the following:

1. Defines equivalent neural network architectures for both approaches
2. Trains both models on the MNIST dataset
3. Collects metrics on:
   - Training time
   - GPU memory usage
   - Loss convergence
   - Final accuracy

### Model Architecture

Both implementations use:
- A CNN for feature extraction from MNIST images
- Multiple denoising MLPs for classification 
- A diffusion-based approach with gradually noised labels

### Key Differences

The key difference between the two approaches is in the gradient flow:

**NoProp Implementation:**
```python
# Each MLP is trained independently
for t in range(T):
    # Disable gradients for MLP parameters during CNN training
    for param in mlps[t].parameters():
        param.requires_grad = False
        
    # Forward pass and compute loss
    u_hat = mlps[t](x_features, z[t+1].detach())
    loss = torch.mean((u_hat - u_y) ** 2)
    cnn_loss += loss
    
    # Re-enable gradients
    for param in mlps[t].parameters():
        param.requires_grad = True

# Train MLPs separately
for t in range(T):
    with torch.no_grad():
        x_features = cnn(x)  # Features without gradients
    
    # Train this MLP only
    u_hat = mlps[t](x_features, z[t+1].detach())
    loss = torch.mean((u_hat - u_y) ** 2)
    loss.backward()  # Gradients affect only this MLP
```

**Traditional Implementation:**
```python
# End-to-end training
predictions, noisy_labels = model(x, u_y, train=True)
total_loss = sum(batch_losses)
total_loss.backward()  # Gradients flow through the entire model
```

## Running the Comparison

To run the comparison:

```bash
python noprop_vs_backprop.py
```

Or using UV:

```bash
uv run noprop_vs_backprop.py
```

## Results

The script will output metrics for both approaches, including:
- Training loss per epoch
- Training time per epoch
- GPU memory usage
- Final test accuracy

Example output:

```
--- Performance Summary ---
NoProp - Final Loss: 0.xxxx, Total Time: xx.xx, Accuracy: 0.xxxx
Traditional BP - Final Loss: 0.xxxx, Total Time: xx.xx, Accuracy: 0.xxxx
```

## Requirements

- PyTorch
- torchvision
- NumPy
- Rich (for prettier console output)
- Seaborn (optional, for visualization)

If visualization libraries are not available, the script will still run and output text-based metrics.
