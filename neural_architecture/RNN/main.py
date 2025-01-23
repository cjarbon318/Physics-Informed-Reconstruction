import torch
import mlflow
import mlflow.pytorch
import numpy as np
from training import train_model, log_model_with_mlflow
from model import RNNTransformerModel
from visualizer import RNNvisualizer

def main():
    # Start an MLflow
    mlflow.set_experiment("RNN_Transformer_Anomaly_Detection")
    with mlflow.start_run():
        # Hyperparameters
        batch_size = 100
        seq_len = 50
        input_dim = 1
        num_epochs = 200
        learning_rate = 0.001

        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = RNNTransformerModel().to(device)

        # Synthetic data (for testing purposes, replace with real data if available)
        data = torch.rand(batch_size, seq_len, input_dim).to(device)
        target = torch.rand(batch_size, seq_len, input_dim).to(device)

        # Train the model
        trained_model = train_model(model, data, target, device, num_epochs, learning_rate)

        # Log parameters
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("seq_len", seq_len)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("learning_rate", learning_rate)

        # Log the trained model
        log_model_with_mlflow(trained_model)

        # Anomaly detection
        model.eval()
        with torch.no_grad():
            reconstructed_data = trained_model(data) 
            reconstruction_error = torch.abs(data - reconstructed_data) 

        # Convert tensors to numpy for visualization 
        reconstructed_data = reconstructed_data.cpu().numpy()
        reconstruction_error = reconstruction_error.cpu().numpy()
        data = data.cpu().numpy()

        # Set a threshold for anomaly detection
        threshold = 0.1  
        train_size = int(batch_size * 0.7) 
        
        # Visualize the results
        visualizer = RNNvisualizer(data, reconstructed_data, reconstruction_error, threshold, train_size)
        visualizer.visualize_data()
    
if __name__ == "__main__":
    main()
