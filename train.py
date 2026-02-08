import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dataset.loader import create_loaders
from models.mlp import MLP
from trainer.trainer import Trainer

def main():
    df = pd.read_csv("data/plant_disease_dataset.csv")

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    scaler = StandardScaler()
    X = scaler.fit_transform()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_loader, val_loader = create_loaders(
        X_train, y_train, X_val, y_val, batch_size=32
    )

    model = MLP(input_dim = X.shape[1])

    trainer = Trainer(model, train_loader, val_loader, lr=1e-3)
    trainer.fit(epochs=20)

main()