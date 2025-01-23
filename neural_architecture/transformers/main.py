import mlflow
import mlflow.pytorch
from training import train_transformer_autoencoder
from model import TransformerAutoencoder
from data_prep import prepare_data

def main():
    mlflow.set_experiment("Stress_Anomaly_Detection")
    with mlflow.start_run():
        # Parameters
        data_path = '/Users/carliarbon/dataproject/stress.csv'
        input_dim = 1
        d_model = 64
        nhead = 8
        num_encoder_layers = 3
        num_decoder_layers = 3
        dim_feedforward = 128
        max_len = 5000
        batch_size = 100
        num_epochs = 10

        # Prepare data
        train_loader, test_loader = prepare_data(data_path, batch_size=batch_size)

        # Define model
        model = TransformerAutoencoder(input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_len)

        # Train model
        trained_model = train_transformer_autoencoder(model, train_loader, num_epochs=num_epochs)

        # Log model

        # Log model and parameters
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.pytorch.log_model(trained_model, "model")

if __name__ == "__main__":
    main()
