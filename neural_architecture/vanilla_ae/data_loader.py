import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def load_data(data_path):
    data = pd.read_csv(data_path)
    stress_data = data['stress_value (Pa)'].values.reshape(-1, 1)
    return stress_data

#preprocess the data 
def preprocess_data(stress_data):
    scaler = MinMaxScaler()
    stress_data_scaled = scaler.fit_transform(stress_data)
    return stress_data_scaled, scaler

#split the data into train and test sets
def split_data(data, test_size=0.1):
    train_indx, test_indx = train_test_split(range(len(data)), shuffle=True, test_size=test_size)
    train_data_set = Subset(data, train_indx)
    test_data_set = Subset(data, test_indx)
    return train_data_set, test_data_set

#create dataloaders
def create_dataloaders(train_data_set, test_data_set, batch_size):
    small_batch_size = 16
    train_data_loader = DataLoader(train_data_set, batch_size=small_batch_size, shuffle=True, drop_last=True, num_workers=0)
    test_data_loader = DataLoader(test_data_set, batch_size=small_batch_size, shuffle=False, drop_last=True, num_workers=0)
    return train_data_loader, test_data_loader


