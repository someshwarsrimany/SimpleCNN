import torch
import torch.nn as nn
import torch.optim as optim
from .utils import evaluate_model, load_data, train_model
from .model import SimpleCNN


def main():
    """Main function to load data, train, and evaluate the model."""
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_loader, test_loader = load_data(batch_size=32)

    # Initialize model, loss, and optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=100)

    # Evaluate the model
    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    main()
