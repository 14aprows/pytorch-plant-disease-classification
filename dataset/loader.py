from torch.utils.data import DataLoader
from dataset.dataset import PlantDataset

def create_loaders(X_train, y_train, X_val, y_val, batch_size=32):
    train_loader = DataLoader(
        PlantDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        PlantDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader
