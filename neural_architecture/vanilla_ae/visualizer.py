import matplotlib.pyplot as plt
import numpy as np
import mlflow
import mlflow.pytorch

def plot_results(stress_data, reconstructed, reconstruction_error, threshold, train_size):
    plt.figure(figsize=(12, 6))
    plt.plot(stress_data, label='Original Stress Data', color='blue')
    plt.plot(np.arange(train_size, len(stress_data)), reconstructed, label='Reconstructed Stress Data', color='orange')
    plt.scatter(np.arange(train_size, len(stress_data))[reconstruction_error > threshold],
                stress_data[train_size:][reconstruction_error > threshold], color='red', label='Anomalies')
    plt.axhline(y=threshold, color='red', linestyle='--', label='Anomaly Threshold')
    plt.title('Stress Data and Anomaly Detection')
    plt.xlabel('Time')
    plt.ylabel('Stress (Pa)')
    plt.legend()
    plt.savefig("reconstructed_results.png")
    mlflow.log_artifact("reconstructed_results.png")
    plt.show()