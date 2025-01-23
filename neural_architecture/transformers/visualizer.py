import matplotlib.pyplot as plt
import numpy as np
import mlflow
import mlflow.pytorch

class TransformerVisualization:
    def __init__(self, threshold=0.1):
        """Initialize the class with a threshold for anomaly detection."""
        self.threshold = threshold

    def plot_results(self, stress_data, reconstructed, reconstruction_error, threshold, train_size):
        """Plot the results of the Transformer model: original vs. reconstructed data, anomalies, and reconstruction error."""
        plt.figure(figsize=(12, 6))
        plt.plot(stress_data, label='Original Stress Data', color='blue')
        plt.plot(np.arange(train_size, len(stress_data)), reconstructed, label='Reconstructed Stress Data', color='orange')
        plt.scatter(np.arange(train_size, len(stress_data))[reconstruction_error > threshold],
                    stress_data[train_size:][reconstruction_error > threshold], color='red', label='Anomalies')
        plt.axhline(y=threshold, color='red', linestyle='--', label='Anomaly Threshold')
        plt.title('Transformer Model: Stress Data and Anomaly Detection')
        plt.xlabel('Time')
        plt.ylabel('Stress (Pa)')
        plt.legend()
        plt.savefig("transformer_reconstructed_results.png")
        mlflow.log_artifact("transformer_reconstructed_results.png")
        plt.show()

    def plot_reconstruction_error(self, reconstruction_error, threshold):
        """Plot the reconstruction error and highlight the anomalies."""
        plt.figure(figsize=(12, 6))
        plt.plot(reconstruction_error, label='Reconstruction Error', color='green')
        plt.axhline(y=threshold, color='red', linestyle='--', label='Anomaly Threshold')
        plt.title('Reconstruction Error and Anomaly Detection')
        plt.xlabel('Time')
        plt.ylabel('Reconstruction Error')
        plt.legend()
        plt.savefig("reconstruction_error_plot.png")
        mlflow.log_artifact("reconstruction_error_plot.png")
        plt.show()

    def plot_multiple_sequences(self, original_sequences, reconstructed_sequences, num_samples=5):
        """Plot a few sequences of the original vs reconstructed data."""
        plt.figure(figsize=(12, 6))
        for i in range(num_samples):
            plt.plot(original_sequences[i], label=f'Original Sequence {i+1}')
            plt.plot(reconstructed_sequences[i], label=f'Reconstructed Sequence {i+1}', linestyle='dashed')
        plt.title(f'Original vs Reconstructed Sequences ({num_samples} samples)')
        plt.xlabel('Time Step')
        plt.ylabel('Scaled Stress Value')
        plt.legend()
        plt.savefig("original_vs_reconstructed_sequences.png")
        mlflow.log_artifact("original_vs_reconstructed_sequences.png")
        plt.show()

    def log_results_to_mlflow(self, model, stress_data, reconstructed, reconstruction_error, threshold, train_size):
        """Log results such as the reconstructed plot and reconstruction error to MLflow."""
        # Log the artifact (plot)
        self.plot_results(stress_data, reconstructed, reconstruction_error, threshold, train_size)

        # Log the model
        mlflow.pytorch.log_model(model, "transformer_model")

        # Log reconstruction error plot
        self.plot_reconstruction_error(reconstruction_error, threshold)
