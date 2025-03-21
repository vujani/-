import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.fft
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import numpy as np
from torch.cuda.amp import autocast, GradScaler


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


# Блок для обработки низких частот, адаптированный для суперразрешения
class SRLowFreqNet(nn.Module):
    def __init__(self):
        super(SRLowFreqNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(64)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.se(out)
        out = self.relu(self.conv3(out))
        out = self.conv4(out)
        return torch.clamp(x + out, 0.0, 1.0)


# Блок для обработки высоких частот, адаптированный для суперразрешения
class SRHighFreqNet(nn.Module):
    def __init__(self):
        super(SRHighFreqNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(128)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.se(out)
        out = self.relu(self.conv3(out))
        out = self.conv4(out)
        return torch.clamp(x + out, 0.0, 1.0)


# Адаптивный блок суперразрешения
class AdaptiveSuperResolutionBlock(nn.Module):
    """
    Адаптивный блок для суперразрешения:
      - Применяет FFT к входному изображению (например, результат bicubic апскейла).
      - Генерирует маску для разделения низких и высоких частот.
      - Обрабатывает низкие и высокие частоты с помощью специализированных сетей.
      - Объединяет результаты с обучаемыми весами и проводит постобработку.
    """

    def __init__(self):
        super(AdaptiveSuperResolutionBlock, self).__init__()
        self.low_net = SRLowFreqNet()
        self.high_net = SRHighFreqNet()
        self.low_weight = nn.Parameter(torch.tensor(1.0))
        self.high_weight = nn.Parameter(torch.tensor(1.5))
        self.mask_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.post_processing = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Применяем FFT и сдвигаем спектр
        fft = torch.fft.fft2(x, norm="ortho")
        fft_shifted = torch.fft.fftshift(fft)
        b, c, H, W = x.shape
        mag = torch.abs(fft_shifted)
        mag_avg = torch.mean(mag, dim=1, keepdim=True)
        low_mask = self.mask_net(mag_avg)
        low_mask = low_mask.expand(-1, c, -1, -1)
        high_mask = 1.0 - low_mask
        low_fft = fft_shifted * low_mask
        high_fft = fft_shifted * high_mask
        low_ifft = torch.fft.ifft2(torch.fft.ifftshift(low_fft), norm="ortho").real
        high_ifft = torch.fft.ifft2(torch.fft.ifftshift(high_fft), norm="ortho").real
        low_processed = self.low_net(low_ifft)
        high_processed = self.high_net(high_ifft)
        combined = self.low_weight * low_processed + self.high_weight * high_processed
        out = self.post_processing(combined)
        return torch.clamp(out, 0.0, 1.0)


##################################
# Датасет для суперразрешения
##################################

class DIV2KSuperResolutionDataset(torch.utils.data.Dataset):
    """
    Датасет DIV2K для суперразрешения.
    HR-изображение берется из датасета, а LR-версия создается путем downscale и последующего bicubic апскейла.
    """

    def __init__(self, root, scale_factor=2, transform=None):
        self.root = root
        self.scale_factor = scale_factor
        self.transform = transform
        self.image_files = [os.path.join(root, f) for f in os.listdir(root)
                            if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        hr_image = Image.open(self.image_files[idx]).convert('RGB')
        if self.transform:
            hr_image = self.transform(hr_image)  # hr_image имеет форму [C, H, W]
        c, H, W = hr_image.shape  # Распаковка без батчевого измерения
        lr_H, lr_W = H // self.scale_factor, W // self.scale_factor
        # Downscale и bicubic апскейлинг
        lr_image = nn.functional.interpolate(hr_image.unsqueeze(0), size=(lr_H, lr_W),
                                             mode='bicubic', align_corners=False)
        lr_upscaled = nn.functional.interpolate(lr_image, size=(H, W),
                                                mode='bicubic', align_corners=False)
        lr_upscaled = lr_upscaled.squeeze(0)
        return lr_upscaled, hr_image


##################################
# Тренировочная функция для адаптивного блока суперразрешения
##################################

def train_adaptive_super_resolution_block():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    num_epochs = 100
    learning_rate = 1e-4
    patience = 10

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = DIV2KSuperResolutionDataset(root='DIV2K/train', scale_factor=2, transform=transform)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    model = AdaptiveSuperResolutionBlock().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for lr_upscaled, hr in train_loader:
            lr_upscaled = lr_upscaled.to(device)
            hr = hr.to(device)
            optimizer.zero_grad()
            with autocast():
                output = model(lr_upscaled)
                loss = criterion(output, hr)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * lr_upscaled.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lr_upscaled, hr in val_loader:
                lr_upscaled = lr_upscaled.to(device)
                hr = hr.to(device)
                with autocast():
                    output = model(lr_upscaled)
                    loss = criterion(output, hr)
                val_loss += loss.item() * lr_upscaled.size(0)
        val_loss /= len(val_loader.dataset)
        print(
            f"Adaptive SR Block Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            os.makedirs('saved_models', exist_ok=True)
            torch.save(model.state_dict(), 'saved_models/resolution_adaptive_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered for Adaptive SR Block.")
                break


if __name__ == "__main__":
    train_adaptive_super_resolution_block()
