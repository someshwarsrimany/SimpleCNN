# SimpleCNN

A lightweight project that implements a simple convolutional neural network (CNN) for image classification tasks using PyTorch. This project is designed for beginners to learn the basics of deep learning, including training and evaluating CNN models on datasets like CIFAR-10.

---

## Features

- A simple CNN architecture with:
  - Convolutional layers
  - Pooling layers
  - Fully connected layers
- Dataset support for CIFAR-10
- Training and evaluation pipelines
- GPU (CUDA) support for accelerated training
- Easy-to-understand code structure for educational purposes

---

## Installation

### Prerequisites
- Python >= 3.8
- Rye (for environment and dependency management)

### Steps for Setup

1. **Install Rye**:
   If you don't have Rye installed, follow the instructions in the [official Rye documentation](https://rye-up.com/).

2. **Clone the repository:**

   ```bash
   git clone https://github.com/someshwarsrimany/SimpleCNN.git
   ```

3. **Install dependencies:**
   Use Rye to install the project's dependencies.

   ```bash
   rye sync
   ```

---

## Usage

### Training the Model
To train the model, simply run:
```bash
rye run main
```

### Customizing Training
You can modify the training parameters (e.g., epochs, batch size, learning rate) in the `main()` function inside `main.py`.

---

## Project Structure

```
simplecnn/
├── src/                        # Source code directory
|   ├── simplecnn/              # Package Folder
│       ├── __init__.py         # Package Initializer
│       ├── main.py             # Main script to run the training and evaluation
│       ├── model.py            # CNN model definition
│       └── utils.py            # Helper functions (e.g., data loading)
├── README.md                   # Project description and instructions
├── pyproject.toml              # Build configuration
├── requirements-dev.lock       # Locked dev dependencies managed by Rye
├── requirements.lock           # Locked dependencies managed by Rye
└── data/                       # Directory for CIFAR-10 dataset (automatically downloaded) if root download location not provided
```

---

## Acknowledgements

- [PyTorch Documentation](https://pytorch.org/docs/)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Rye Documentation](https://rye-up.com/)