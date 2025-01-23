import numpy as np
import matplotlib.pyplot as plt
import mlflow


class RNNvisualizer:
    def __init__(self, original_data, reconstructed_data, reconstruction_error, threshold, train_size):
        """
        Initialize the visualizer with required data and configurations.

        Args:
            original_data (np.ndarray): The original input data.
            reconstructed_data (np.ndarray): The data reconstructed by the model.
            reconstruction_error (np.ndarray): The error between original and reconstructed data.
            threshold (float): The threshold for detecting anomalies.
            train_size (int): Number of samples used for training.
        """
        self.original_data = original_data
        self.reconstructed_data = reconstructed_data
        self.reconstruction_error = reconstruction_error
        self.threshold = threshold
        self.train_size = train_size

    def visualize_data(self):
        """
        Visualizes the original data, reconstructed data, reconstruction error, and detected anomalies.
        """
        # Flatten data along batch and sequence dimensions for visualization
        flattened_original = self.original_data.reshape(-1)
        flattened_reconstructed = self.reconstructed_data.reshape(-1)
        flattened_error = self.reconstruction_error.reshape(-1)

        # Adjust train size for flattened data
        flattened_train_size = self.train_size * self.original_data.shape[1]

        # Detect anomalies in the test set
        test_error = flattened_error[flattened_train_size:]
        test_original = flattened_original[flattened_train_size:]
        anomalies = test_error > self.threshold
        anomaly_indices = np.where(anomalies)[0]
        anomaly_values = test_original[anomaly_indices]

        # Create the plots
        plt.figure(figsize=(15, 10))

        # Plot 1: Original vs Reconstructed Data
        plt.subplot(2, 1, 1)
        plt.plot(flattened_original, label="Original Data", color="blue", alpha=0.6)
        plt.plot(flattened_reconstructed, label="Reconstructed Data", color="orange", alpha=0.6)
        plt.axvline(x=flattened_train_size, color="green", linestyle="--", label="Training/Test Split")
        plt.title("Original vs Reconstructed Data")
        plt.xlabel("Time Steps (Flattened)")
        plt.ylabel("Values")
        plt.legend()

        # Plot 2: Reconstruction Error and Anomalies
        plt.subplot(2, 1, 2)
        plt.plot(test_error, label="Reconstruction Error", color="purple", alpha=0.6)
        plt.axhline(y=self.threshold, color="red", linestyle="--", label="Anomaly Threshold")
        plt.scatter(anomaly_indices, anomaly_values, color="red", label="Anomalies", alpha=0.8)
        plt.title("Reconstruction Error and Detected Anomalies")
        plt.xlabel("Time Steps (Flattened Test Data)")
        plt.ylabel("Error")
        plt.legend()

        # Save the plots and log with MLflow
        plot_filename = "model_visualization.png"
        plt.tight_layout()
        plt.savefig(plot_filename)
        mlflow.log_artifact(plot_filename)

        plt.show()
