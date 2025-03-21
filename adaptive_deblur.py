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


# Блок для обработки низких частот (для деблюринга)
class DeblurLowFreqNet(nn.Module):
    def __init__(self):
        super(DeblurLowFreqNet, self).__init__()
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
        return x + out


# Блок для обработки высоких частот (для деблюринга)
class DeblurHighFreqNet(nn.Module):
    def __init__(self):
        super(DeblurHighFreqNet, self).__init__()
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
        return x + out


# Адаптивный блок деблюринга (обучается отдельно)
class AdaptiveDeblurBlock(nn.Module):
    """
    Адаптивный блок для устранения размытия.
    Выполняет FFT, разделяет спектр на низкие и высокие частоты, обрабатывает их
    отдельными сетями и объединяет результаты с динамическими весами.
    """

    def __init__(self):
        super(AdaptiveDeblurBlock, self).__init__()
        self.low_net = DeblurLowFreqNet()
        self.high_net = DeblurHighFreqNet()
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
        # FFT преобразование и сдвиг спектра
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
        return out


##############################
# Датасет для деблюринга
##############################

class DIV2KDeblurDataset(torch.utils.data.Dataset):
    """
    Датасет DIV2K для устранения размытия.
    Если задано ядро размытия (blur_kernel), то изображение обрабатывается свёрткой,
    иначе имитируется размытие добавлением случайного шума.
    """

    def __init__(self, root, transform=None, blur_kernel=None):
        self.root = root
        self.transform = transform
        self.blur_kernel = blur_kernel
        self.image_files = [os.path.join(root, f) for f in os.listdir(root)
                            if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        clean = img
        if self.blur_kernel is not None:
            blurred = self.apply_blur(img)
        else:
            blurred = img + 0.1 * torch.randn_like(img)
            blurred = torch.clamp(blurred, 0.0, 1.0)
        return blurred, clean

    def apply_blur(self, img):
        kernel = self.blur_kernel
        padding = kernel.size(0) // 2
        blurred = nn.functional.conv2d(img.unsqueeze(0),
                                       kernel.unsqueeze(0).repeat(3, 1, 1, 1),
                                       padding=padding, groups=3)
        return blurred.squeeze(0)


##############################
# Тренировочная функция для адаптивного блока
##############################

def train_adaptive_deblur_block():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    num_epochs = 100
    learning_rate = 1e-4
    patience = 10

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Определяем ядро размытия (например, 5x5 усредняющее)
    blur_kernel = torch.tensor([[1.0 / 25 for _ in range(5)] for _ in range(5)], dtype=torch.float32)

    dataset = DIV2KDeblurDataset(root='DIV2K/train', transform=transform, blur_kernel=blur_kernel)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    model = AdaptiveDeblurBlock().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for blurred, clean in train_loader:
            blurred = blurred.to(device)
            clean = clean.to(device)
            optimizer.zero_grad()
            with autocast():
                output = model(blurred)
                loss = criterion(output, clean)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * blurred.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for blurred, clean in val_loader:
                blurred = blurred.to(device)
                clean = clean.to(device)
                with autocast():
                    output = model(blurred)
                    loss = criterion(output, clean)
                val_loss += loss.item() * blurred.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"Adaptive Block Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            os.makedirs('saved_models', exist_ok=True)
            torch.save(model.state_dict(), 'saved_models/adaptive_deblur_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered for AdaptiveDeblurBlock.")
                break


if __name__ == "__main__":
    train_adaptive_deblur_block()
