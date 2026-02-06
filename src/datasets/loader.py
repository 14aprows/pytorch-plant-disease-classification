import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from .dataset import PlantDataset

def load_data(file_path, test_size=0.2, batch_size=32, random_state=42):
    data = pd.read_csv(file_path)
    
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    train_dataset = PlantDataset(X_train, y_train)
    test_dataset = PlantDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

