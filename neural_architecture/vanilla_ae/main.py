import mlflow
import mlflow.pytorch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from data_loader import load_data, split_data, create_dataloaders, preprocess_data
from model import DeeperVanillaNN
from trainer import train_model
from evaluator import evaluate_model
from visualizer import plot_results

def main():
    mlflow.set_experiment("Stress_Anomaly_Detection")
    with mlflow.start_run():
        # Load and preprocess data
        data_path = '/Users/carliarbon/dataproject/stress.csv'
        stress_data = load_data(data_path)
        stress_data_scaled, scaler = preprocess_data(stress_data)

        # Split data and create DataLoaders
        train_data_set, test_data_set = split_data(stress_data_scaled)
        train_loader, test_loader = create_dataloaders(train_data_set, test_data_set, batch_size=16)

        # Define and train model
        model = DeeperVanillaNN(input_dim=1, latent_dim=8)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.2)
        device = torch.device("cpu")

        mlflow.log_param("input_dim", 1)
        mlflow.log_param("latent_dim", 8)
        mlflow.log_param("learning_rate", 0.01)
        mlflow.log_param("batch_size", 16)

        train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs=1)

        # Evaluate and plot
        reconstructed, reconstruction_error = evaluate_model(model, test_loader, scaler, device)
        threshold = np.percentile(reconstruction_error, 95)
        plot_results(stress_data, reconstructed, reconstruction_error, threshold, len(train_data_set))
        mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    main()