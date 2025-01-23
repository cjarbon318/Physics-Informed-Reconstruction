
import torch.optim as optim
import torch.nn as nn
import mlflow
import mlflow.pytorch
from model import RNNTransformerModel

def train_model(model, data, target, device, num_epochs=200, learning_rate=0.001):
    """
    Train the RNN-Transformer model on synthetic data.
    """
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    return model

def log_model_with_mlflow(model):
    """
    Logs the trained model to MLflow.
    """
    mlflow.pytorch.log_model(model, "model")
