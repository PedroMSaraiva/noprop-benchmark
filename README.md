# NoProp vs. Traditional Backpropagation

This repository contains implementations comparing NoProp (No Propagation) training methodology against traditional end-to-end backpropagation for neural networks. NoProp is a novel training approach that trains neural network layers independently without gradient propagation between them, potentially offering benefits in parallelization, memory efficiency, and training speed.

## Key Implementations

The repository includes three implementations with different optimizations:

1. **Basic Implementation** (`noprop_vs_backprop.py`): Standard implementation comparing NoProp vs traditional backpropagation.
2. **Parallel Implementation** (`noprop_vs_backprop_paralell.py`): Enhanced with parallel processing for NoProp layers using ThreadPoolExecutor.
3. **Numba-Accelerated Implementation** (`noprop_vs_backprop_numba.py`): Uses Numba JIT compilation to speed up computationally intensive operations.

Common functionality is extracted into `utils.py` for code reuse and maintainability.

## The NoProp Concept

NoProp is based on the insight that neural networks can be trained effectively by:

1. **Layer Independence**: Training each layer independently, without backpropagating gradients through the entire network.
2. **Target Consistency**: Using the same target signals for all layers.
3. **Forward-Only Information Flow**: During training, information only flows forward, not backward.

In this implementation, we use a diffusion model setup where:
- A CNN extracts features from images
- Multiple MLP layers denoise representations at different noise levels
- Each MLP layer is trained independently - this is the key NoProp concept

### Advantages of NoProp

- **Parallelization**: Layers can be trained in parallel (demonstrated in `noprop_vs_backprop_paralell.py`)
- **Memory Efficiency**: No need to store intermediate activations for the entire network
- **Scalability**: Adding more layers doesn't increase memory requirements proportionally
- **Computational Efficiency**: Each layer can be trained with different hardware or even distributed

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/noprop-benchmark.git
cd noprop-benchmark

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install torch torchvision numpy matplotlib seaborn rich numba
```

## Usage

### Basic NoProp vs Backprop Comparison

```bash
# Run with default parameters (MNIST dataset)
python noprop_vs_backprop.py

# Run with a specific dataset
python noprop_vs_backprop.py --dataset cifar10

# Run with custom parameters
python noprop_vs_backprop.py --dataset cifar100 --epochs 10 --batch-size 128 --lr 0.001
```

### Parallel Implementation

```bash
# Run with default parameters
python noprop_vs_backprop_paralell.py

# Set the number of worker processes (default is available CPU cores - 1)
python noprop_vs_backprop_paralell.py --num-workers 4 --dataset cifar10
```

### Numba-Accelerated Implementation

```bash
# Run the Numba implementation (simplified parameters for testing)
python noprop_vs_backprop_numba.py

# Customize epochs and batch size
python noprop_vs_backprop_numba.py --epochs 3 --batch-size 32
```

## Results and Visualization

After each run, the scripts will:

1. Display detailed training metrics in rich console tables
2. Show accuracy comparisons between NoProp and traditional backpropagation
3. Generate visualization plots in the `results/` directory comparing:
   - Training loss over epochs
   - Training time per epoch
   - GPU memory usage

## Implementation Details

### Common Components (`utils.py`)

- Dataset loading and configuration
- Model architectures (CNN and MLPs)
- Evaluation functions
- Metrics visualization
- Progress tracking with rich console displays

### Basic Implementation

Trains each layer sequentially but independently, without cross-layer gradient flow.

### Parallel Implementation

Leverages ThreadPoolExecutor to train independent MLP layers in parallel, exploiting the inherent parallelizability of NoProp.

### Numba Implementation

Uses Numba JIT compilation to accelerate:
- Forward diffusion steps
- Loss computation
- Batch inference operations

## Supported Datasets

- **MNIST**: Simple handwritten digits (28×28, grayscale)
- **CIFAR-10**: 10-class color images (32×32)
- **CIFAR-100**: 100-class color images (32×32)

Dataset-specific configurations are defined in `utils.py` with appropriate model architecture adjustments.

## Benchmark Results

Based on experiments, NoProp tends to show:
1. **Comparable accuracy** to traditional backpropagation
2. **Faster training** due to parallelization capabilities
3. **Lower memory usage** especially for deeper networks
4. **Better scalability** with number of layers

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)

## References

This implementation is inspired by research on alternative training methods that avoid end-to-end backpropagation:

1. "Training Neural Networks Without Backpropagation: A Survey" (hypothetical)
2. "Parallel Training of Deep Networks with Local Updates" (hypothetical)
