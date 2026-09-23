import torch
import torch.nn as nn
import torch.optim as optim

from dataloader_setup import get_dataloaders
from model import MultiTaskResNet

def train_model(data_path, epochs=10, batch_size=32, lr=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, fruit_classes, freshness_classes = get_dataloaders(
        data_path=data_path,
        batch_size=batch_size
    )

    model = MultiTaskResNet(num_fruits=len(fruit_classes), num_freshness=len(freshness_classes))
    model.to(device)

    fruit_criterion = nn.CrossEntropyLoss()
    freshness_criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, fruit_labels, freshness_labels in train_loader:
            images = images.to(device)
            fruit_labels = fruit_labels.to(device)
            freshness_labels = freshness_labels.to(device)

            optimizer.zero_grad()

            fruit_output, freshness_output = model(images)

            loss1 = fruit_criterion(fruit_output, fruit_labels)
            loss2 = freshness_criterion(freshness_output, freshness_labels)

            loss = loss1 + loss2
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        val_fruit_correct = 0
        val_fresh_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, fruit_labels, freshness_labels in val_loader:
                images = images.to(device)
                fruit_labels = fruit_labels.to(device)
                freshness_labels = freshness_labels.to(device)

                fruit_output, freshness_output = model(images)

                _, fruit_pred = torch.max(fruit_output, 1)
                _, fresh_pred = torch.max(freshness_output, 1)

                val_fruit_correct += (fruit_pred == fruit_labels).sum().item()
                val_fresh_correct += (fresh_pred == freshness_labels).sum().item()
                val_total += fruit_labels.size(0)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Training Loss: {total_loss:.4f}")
        print(f"Fruit Accuracy: {val_fruit_correct / val_total * 100:.2f}%")
        print(f"Freshness Accuracy: {val_fresh_correct / val_total * 100:.2f}%")
        print("--------------------------------------------------")

    torch.save(model.state_dict(), "../models/multitask_resnet18.pth")
    print("Model saved to: models/multitask_resnet18.pth")

if __name__ == "__main__":
    train_model("../data/Fruit Freshness Dataset", epochs=10)
