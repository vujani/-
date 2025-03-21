import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import argparse


##################################
# Базовая сеть суперразрешения
##################################

class ResidualBlockSR(nn.Module):
    def __init__(self, channels):
        super(ResidualBlockSR, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


class BasicSuperResolutionNet(nn.Module):
    """
    Базовая сеть суперразрешения.
    Вход – апскейленное bicubic изображение (LR), цель – HR изображение.
    Сеть предсказывает поправочный сигнал, который прибавляется к входу.
    """

    def __init__(self):
        super(BasicSuperResolutionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlockSR(64) for _ in range(8)])
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.res_blocks(out)
        out = self.conv2(out)
        output = x + out
        return torch.clamp(output, 0.0, 1.0)


##################################
# Датасет для суперразрешения
##################################

class DIV2KSuperResolutionDataset(torch.utils.data.Dataset):
    """
    Датасет DIV2K для суперразрешения.
    HR-изображение берется из датасета, а входное создается путем downscale и последующего bicubic апскейла.
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
            hr_image = self.transform(hr_image)  # форма [C, H, W]
        c, H, W = hr_image.shape
        lr_H, lr_W = H // self.scale_factor, W // self.scale_factor
        lr_image = nn.functional.interpolate(hr_image.unsqueeze(0), size=(lr_H, lr_W),
                                             mode='bicubic', align_corners=False)
        lr_upscaled = nn.functional.interpolate(lr_image, size=(H, W),
                                                mode='bicubic', align_corners=False)
        lr_upscaled = lr_upscaled.squeeze(0)
        return lr_upscaled, hr_image


##################################
# Импорт адаптивного блока суперразрешения
##################################

from resolution_adaptive import AdaptiveSuperResolutionBlock


##################################
# Метрики: PSNR и SSIM
##################################

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    psnr = 10 * torch.log10(1 / mse)
    return psnr.item()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    window = _2D_window.unsqueeze(0).unsqueeze(0)
    return window.expand(channel, 1, window_size, window_size).contiguous()


def ssim(img1, img2, window_size=11, size_average=True):
    img1 = img1.float()
    img2 = img2.float()
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel).to(img1.device)
    mu1 = nn.functional.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = nn.functional.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = nn.functional.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = nn.functional.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = nn.functional.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item() if size_average else ssim_map.mean(1).mean(1).mean(1).item()


##################################
# Тренировочная функция для суперразрешения
##################################

def train_super_resolution_model(use_adaptive=False, freeze_adaptive=False):
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
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    # Создаем базовую сеть суперразрешения
    sr_model = BasicSuperResolutionNet().to(device)
    sr_model_path = "saved_models/resolution_sr_model.pth"
    if os.path.exists(sr_model_path):
        sr_model.load_state_dict(torch.load(sr_model_path, map_location=device))
        print("Предобученные веса SR модели загружены из", sr_model_path)
    else:
        print("Предобученные веса SR модели не найдены, обучение с нуля.")

    # Если используется адаптивный блок, загружаем его
    adaptive_block = None
    if use_adaptive:
        adaptive_block = AdaptiveSuperResolutionBlock().to(device)
        adaptive_model_path = 'saved_models/resolution_adaptive_model.pth'
        if os.path.exists(adaptive_model_path):
            adaptive_block.load_state_dict(torch.load(adaptive_model_path, map_location=device))
            if freeze_adaptive:
                for param in adaptive_block.parameters():
                    param.requires_grad = False
                adaptive_block.eval()
            print("Адаптивный блок загружен из", adaptive_model_path)
        else:
            print("Адаптивный блок не найден, proceeding without it.")
            adaptive_block = None

    optimizer = optim.Adam(sr_model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scaler = GradScaler()
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        sr_model.train()
        train_loss = 0.0
        for lr_upscaled, hr in train_loader:
            lr_upscaled = lr_upscaled.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)
            optimizer.zero_grad()
            if adaptive_block is not None:
                with torch.no_grad() if freeze_adaptive else torch.enable_grad():
                    adaptive_output = adaptive_block(lr_upscaled)
                input_to_sr = adaptive_output
            else:
                input_to_sr = lr_upscaled
            with autocast():
                output = sr_model(input_to_sr)
                loss = criterion(output, hr)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * lr_upscaled.size(0)
        train_loss /= len(train_loader.dataset)

        sr_model.eval()
        val_loss = 0.0
        psnr_total = 0.0
        ssim_total = 0.0
        with torch.no_grad():
            for lr_upscaled, hr in val_loader:
                lr_upscaled = lr_upscaled.to(device, non_blocking=True)
                hr = hr.to(device, non_blocking=True)
                if adaptive_block is not None:
                    adaptive_output = adaptive_block(lr_upscaled)
                    input_to_sr = adaptive_output
                else:
                    input_to_sr = lr_upscaled
                with autocast():
                    output = sr_model(input_to_sr)
                    loss = criterion(output, hr)
                val_loss += loss.item() * lr_upscaled.size(0)
                for i in range(output.size(0)):
                    out_i = output[i:i + 1].float()
                    hr_i = hr[i:i + 1].float()
                    psnr_total += calculate_psnr(out_i, hr_i)
                    ssim_total += ssim(out_i, hr_i)
        val_loss /= len(val_loader.dataset)
        avg_psnr = psnr_total / len(val_loader.dataset)
        avg_ssim = ssim_total / len(val_loader.dataset)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            os.makedirs('saved_models', exist_ok=True)
            torch.save(sr_model.state_dict(), sr_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Обучение основной сети суперразрешения с опциональным адаптивным блоком")
    parser.add_argument('--use_adaptive', action='store_true', help='Использовать обученный адаптивный блок')
    parser.add_argument('--freeze_adaptive', action='store_true',
                        help='Заморозить веса адаптивного блока при дообучении основной сети')
    args = parser.parse_args()
    train_super_resolution_model(use_adaptive=args.use_adaptive, freeze_adaptive=args.freeze_adaptive)
