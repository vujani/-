import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import argparse


# Функции для добавления артефактов
def add_noise(pil_image, noise_level=0.15):
    image = np.array(pil_image).astype(np.float32) / 255.0
    noise = np.random.normal(0, noise_level, image.shape)
    noisy = image + noise
    noisy = np.clip(noisy, 0, 1)
    noisy = (noisy * 255).astype(np.uint8)
    return Image.fromarray(noisy)


def add_blur(pil_image, kernel_size=5):
    image = np.array(pil_image)
    blurred = Image.fromarray(
        cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    )
    return blurred


def add_combined(pil_image):
    # сначала шум, затем размытие
    return add_blur(add_noise(pil_image), kernel_size=5)


# Если cv2 не импортирован – импортируем
try:
    import cv2
except ImportError:
    raise ImportError("Требуется opencv-python для применения размытости.")

# Определяем классы артефактов
# 0: clean, 1: noisy, 2: blurred, 3: combined
ARTIFACT_CLASSES = ["clean", "noisy", "blurred", "combined"]


# Кастомный датасет для классификации артефактов
class ArtifactDataset(Dataset):
    def __init__(self, root, transform=None, artifact_prob=0.5):
        """
        root – путь к папке с чистыми изображениями (например, из DIV2K/train)
        artifact_prob – вероятность применения артефакта к изображению (иначе изображение остается чистым)
        """
        self.root = root
        self.transform = transform
        self.image_files = [os.path.join(root, f) for f in os.listdir(root)
                            if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        self.artifact_prob = artifact_prob

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert("RGB")
        label = 0  # по умолчанию "clean"
        # Случайное применение артефакта с вероятностью artifact_prob
        if random.random() < self.artifact_prob:
            artifact_choice = random.choice([1, 2, 3])  # исключаем 0
            label = artifact_choice
            if artifact_choice == 1:
                image = add_noise(image)
            elif artifact_choice == 2:
                image = add_blur(image)
            elif artifact_choice == 3:
                image = add_combined(image)
        # Применяем transform
        if self.transform:
            image = self.transform(image)
        return image, label


# Простейшая CNN для классификации артефактов
class ArtifactClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(ArtifactClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 128x128 если вход 256x256
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64x64
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 32x32
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_artifact_classifier():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    dataset = ArtifactDataset(root="DIV2K/train", transform=transform, artifact_prob=0.7)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    model = ArtifactClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        train_loss /= len(train_loader.dataset)
        acc = correct / total * 100

        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = correct_val / total_val * 100
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Acc: {acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/artifact_selector_model.pth")
    print("Модель классификатора артефактов сохранена в saved_models/artifact_selector_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение модели классификатора артефактов")
    parser.add_argument("--epochs", type=int, default=20, help="Количество эпох обучения")
    args = parser.parse_args()
    train_artifact_classifier()
