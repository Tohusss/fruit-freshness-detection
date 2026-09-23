import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FruitFreshnessDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = []
        self.fruit_labels = []
        self.freshness_labels = []

        self.fruit_classes = ["Apple", "Banana", "Strawberry"]
        self.freshness_classes = ["Fresh", "Rotten"]

        self._load_dataset()

    def _load_dataset(self):
        for fruit in self.fruit_classes:
            for freshness in self.freshness_classes:
                folder = os.path.join(self.root_dir, fruit, freshness)

                if not os.path.exists(folder):
                    continue

                fruit_label = self.fruit_classes.index(fruit)
                freshness_label = self.freshness_classes.index(freshness)

                for filename in os.listdir(folder):
                    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".jfif", ".webp")):
                        filepath = os.path.join(folder, filename)
                        self.image_paths.append(filepath)
                        self.fruit_labels.append(fruit_label)
                        self.freshness_labels.append(freshness_label)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.fruit_labels[idx], self.freshness_labels[idx]

    def __len__(self):
        return len(self.image_paths)
