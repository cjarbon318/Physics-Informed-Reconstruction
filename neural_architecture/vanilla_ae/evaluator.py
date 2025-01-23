import numpy as np
import torch

# Evaluate the model on the test set
def evaluate_model(model, test_loader, scaler, device):
    model.eval()
    reconstructed = []
    with torch.no_grad():
        for x in test_loader:
            x = x.to(device).float()
            outputs = model(x)
            reconstructed.extend(outputs.cpu().numpy())

    reconstructed = np.array(reconstructed).reshape(-1, 1)
    reconstruction_error = np.mean((scaler.inverse_transform(reconstructed) - scaler.inverse_transform(reconstructed)) ** 2, axis=1)
    return reconstructed, reconstruction_error