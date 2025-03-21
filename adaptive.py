import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.fft
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import numpy as np
from torch.utils.checkpoint import checkpoint


# SE-блок (Squeeze-and-Excitation)
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# Пользовательский датасет для DIV2K
class DIV2KDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, noise_level=0.15):
        self.root = root
        self.transform = transform
        self.noise_level = noise_level
        self.image_files = [os.path.join(root, f) for f in os.listdir(root)
                            if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        clean = img
        noisy = img + self.noise_level * torch.randn_like(img)
        noisy = torch.clamp(noisy, 0.0, 1.0)
        return noisy, clean


# Обновленный блок для обработки низких частот
class LowFreqNet(nn.Module):
    def __init__(self):
        super(LowFreqNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.conv4(out)
        return x + out


# Обновленный блок для обработки высоких частот
class HighFreqNet(nn.Module):
    def __init__(self):
        super(HighFreqNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.conv4(out)
        return x + out


# Обновленный адаптивный блок с улучшенной маскировкой
class AdaptiveBlock(nn.Module):
    def __init__(self):
        super(AdaptiveBlock, self).__init__()
        self.low_net = LowFreqNet()
        self.high_net = HighFreqNet()
        self.high_weight = nn.Parameter(torch.tensor(1.5))
        self.low_weight = nn.Parameter(torch.tensor(1.0))

        # Улучшенная сеть для генерации маски
        self.mask_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # SE-блоки для внимания
        self.se_low = SEBlock(3)
        self.se_high = SEBlock(3)

    def forward(self, x):
        # Применяем FFT и сдвигаем нули в центр
        fft = torch.fft.fft2(x, norm="ortho")
        fft_shifted = torch.fft.fftshift(fft)
        b, c, H, W = x.shape

        # Вычисляем среднюю величину спектра по каналам для входа в mask_net
        mag = torch.abs(fft_shifted)
        mag_avg = torch.mean(mag, dim=1, keepdim=True)  # [B, 1, H, W]

        # Генерация маски
        low_mask = self.mask_net(mag_avg)
        low_mask = low_mask.expand(-1, c, -1, -1)
        high_mask = 1.0 - low_mask

        # Применяем маски к спектральному представлению
        low_fft = fft_shifted * low_mask
        high_fft = fft_shifted * high_mask

        # Обратное сдвиг и обратное преобразование FFT
        low_fft_ishift = torch.fft.ifftshift(low_fft)
        high_fft_ishift = torch.fft.ifftshift(high_fft)
        low_ifft = torch.fft.ifft2(low_fft_ishift, norm="ortho").real
        high_ifft = torch.fft.ifft2(high_fft_ishift, norm="ortho").real

        # Обработка низких и высоких частот с градиентным чекпоинтингом
        low_processed = checkpoint(self.low_net, low_ifft)
        high_processed = checkpoint(self.high_net, high_ifft)

        # Применяем SE-механизм для усиления важных признаков
        low_processed = self.se_low(low_processed)
        high_processed = self.se_high(high_processed)

        # Комбинируем с динамическими весами
        out = self.low_weight * low_processed + self.high_weight * high_processed
        return out


# Оптимизированный DataLoader
def get_optimized_data_loader(dataset, batch_size=4, num_workers=4):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )


# Улучшенная функция обучения с оптимизацией памяти
def train_adaptive_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    num_epochs = 100
    learning_rate = 1e-3
    patience = 5

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = DIV2KDataset(root='DIV2K/train', transform=transform, noise_level=0.15)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = get_optimized_data_loader(train_dataset, batch_size=batch_size)
    val_loader = get_optimized_data_loader(val_dataset, batch_size=batch_size)

    model = AdaptiveBlock().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for noisy, clean in train_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                output = model(noisy)
                loss = criterion(output, clean)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * noisy.size(0)

        scheduler.step()
        train_loss /= len(train_loader.dataset)

        # Валидация
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy = noisy.to(device)
                clean = clean.to(device)
                output = model(noisy)
                loss = criterion(output, clean)
                val_loss += loss.item() * noisy.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            os.makedirs('saved_models', exist_ok=True)
            torch.save(model.state_dict(), 'saved_models/adaptive_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Ранняя остановка активирована.")
                break


if __name__ == "__main__":
    train_adaptive_model()