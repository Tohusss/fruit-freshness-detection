from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from dataset import FruitFreshnessDataset

def get_dataloaders(data_path, batch_size=32, img_size=224, train_split=0.8):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    dataset = FruitFreshnessDataset(root_dir=data_path, transform=transform)

    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, dataset.fruit_classes, dataset.freshness_classes
