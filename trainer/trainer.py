import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

class Trainer:
    def __init__(self, model, train_loader, val_loader, lr=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.best_acc = 0.0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        loop = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for x, y in loop:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        preds_all, targets_all = [], []

        loop = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
        with torch.no_grad():
            for x, y in loop:
                x, y = x.to(self.device), y.to(self.device)

                outputs = self.model(x)
                loss = self.criterion(outputs, y)

                preds = torch.argmax(outputs, dim=1)

                total_loss += loss.item()
                preds_all.extend(preds.cpu().numpy())
                targets_all.extend(y.cpu().numpy())
                loop.set_postfix(loss=loss.item())

        acc = (torch.tensor(preds_all) == torch.tensor(targets_all)).float().mean().item() * 100
        cm = confusion_matrix(targets_all, preds_all)
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss, acc, cm
    
    def fit(self, epochs=20):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_acc, cm = self.validate(epoch)

            print(
                f"Epoch {epoch} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}%"
            )

            print("Confusion Matrix:\n", cm)

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                torch.save(self.model.state_dict(), "checkpoints/best_model.pth")
                print("Best model saved with accuracy: {:.2f}%".format(self.best_acc))
