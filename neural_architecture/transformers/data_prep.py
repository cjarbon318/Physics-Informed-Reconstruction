import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

def load_data(data_path):
    """Load data from a CSV file."""
    print(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path)
    print(f"Dataset shape: {data.shape}")
    print(f"First few rows:\n{data.head()}")
    return data

def preprocess_data(data):
    """Preprocess the data by extracting and scaling the 'stress_value (Pa)' column."""
    if 'stress_value (Pa)' not in data.columns:
        raise ValueError("Column 'stress_value (Pa)' not found in the dataset.")

    stress_data = data['stress_value (Pa)'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    stress_data_scaled = scaler.fit_transform(stress_data)
    
    return stress_data_scaled

def create_sequences(stress_data_scaled, sequence_length=100):
    """Create sequences of data for training."""
    sequences = []
    for i in range(len(stress_data_scaled) - sequence_length):
        sequences.append(stress_data_scaled[i:i + sequence_length])
    return np.array(sequences)

def split_data(sequences, test_size=0.1):
    """Split the sequences into training and testing datasets."""
    train_idx, test_idx = train_test_split(range(len(sequences)), test_size=test_size, shuffle=True)
    train_data_set = Subset(sequences, train_idx)
    test_data_set = Subset(sequences, test_idx)
    return train_data_set, test_data_set

def create_data_loaders(train_data_set, test_data_set, batch_size=100):
    """Create DataLoader objects for training and testing."""
    train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, test_loader

def prepare_data(data_path, batch_size=100, test_size=0.1, sequence_length=100):
    """Orchestrator function to load, preprocess, create sequences, and create DataLoaders."""
    # Load and preprocess data
    data = load_data(data_path)
    stress_data_scaled = preprocess_data(data)
    
    # Create sequences from the data
    sequences = create_sequences(stress_data_scaled, sequence_length)
    print(f"Total sequences: {len(sequences)}")
    
    # Split data into training and testing datasets
    train_data_set, test_data_set = split_data(sequences, test_size)
    
    # Create DataLoader objects
    train_loader, test_loader = create_data_loaders(train_data_set, test_data_set, batch_size)
    
    print(f"Number of batches in train_loader: {len(train_loader)}")
    return train_loader, test_loader
