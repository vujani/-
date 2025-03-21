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

# Импорт адаптивного блока, который уже обучается отдельно
from adaptive_deblur import AdaptiveDeblurBlock


##############################
# Датасет для деблюринга
##############################

class DIV2KDeblurDataset(torch.utils.data.Dataset):
    """
    Датасет DIV2K для устранения размытия.
    При наличии ядра размытия (blur_kernel) изображение обрабатывается свёрткой,
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
# Базовая сеть деблюринга
##############################

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.relu(self.conv1(x))
        residual = self.conv2(residual)
        return x + residual


class BasicDeblurNet(nn.Module):
    """
    Базовая сеть деблюринга с residual-соединениями.
    """

    def __init__(self):
        super(BasicDeblurNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(5)])
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.res_blocks(out)
        out = self.conv2(out)
        return x + out


##############################
# Метрики: PSNR и SSIM
##############################

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 10 * torch.log10(1 / mse).item()


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


##############################
# Тренировочная функция основной сети
##############################

def train_deblur_model(use_adaptive=False, freeze_adaptive=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    num_epochs = 100
    learning_rate = 1e-4
    patience = 10

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    blur_kernel = torch.tensor([[1.0 / 25 for _ in range(5)] for _ in range(5)], dtype=torch.float32)

    dataset = DIV2KDeblurDataset(root='DIV2K/train', transform=transform, blur_kernel=blur_kernel)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    # Основная сеть деблюринга
    deblur_net = BasicDeblurNet().to(device)

    # Если используется адаптивный блок, загружаем его обученную версию
    adaptive_block = None
    if use_adaptive:
        adaptive_block = AdaptiveDeblurBlock().to(device)
        adaptive_model_path = 'saved_models/adaptive_deblur_model.pth'
        if os.path.exists(adaptive_model_path):
            adaptive_block.load_state_dict(torch.load(adaptive_model_path))
            if freeze_adaptive:
                for param in adaptive_block.parameters():
                    param.requires_grad = False
                adaptive_block.eval()
            print("Adaptive block loaded from", adaptive_model_path)
        else:
            print("Adaptive model not found. Proceeding without adaptive block.")
            adaptive_block = None

    optimizer = optim.Adam(deblur_net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scaler = GradScaler()
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        deblur_net.train()
        train_loss = 0.0
        for blurred, clean in train_loader:
            blurred = blurred.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            optimizer.zero_grad()

            # Если есть адаптивный блок, пропускаем изображение через него
            if adaptive_block is not None:
                with torch.no_grad() if freeze_adaptive else torch.enable_grad():
                    adaptive_output = adaptive_block(blurred)
                input_to_deblur = adaptive_output
            else:
                input_to_deblur = blurred

            with autocast():
                output = deblur_net(input_to_deblur)
                loss = criterion(output, clean)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * blurred.size(0)
        train_loss /= len(train_loader.dataset)

        deblur_net.eval()
        val_loss = 0.0
        psnr_total = 0.0
        ssim_total = 0.0
        with torch.no_grad():
            for blurred, clean in val_loader:
                blurred = blurred.to(device, non_blocking=True)
                clean = clean.to(device, non_blocking=True)
                if adaptive_block is not None:
                    adaptive_output = adaptive_block(blurred)
                    input_to_deblur = adaptive_output
                else:
                    input_to_deblur = blurred
                with autocast():
                    output = deblur_net(input_to_deblur)
                    loss = criterion(output, clean)
                val_loss += loss.item() * blurred.size(0)
                for i in range(output.size(0)):
                    output_i = output[i:i + 1].float()
                    clean_i = clean[i:i + 1].float()
                    psnr_total += calculate_psnr(output_i, clean_i)
                    ssim_total += ssim(output_i, clean_i)
        val_loss /= len(val_loader.dataset)
        avg_psnr = psnr_total / len(val_loader.dataset)
        avg_ssim = ssim_total / len(val_loader.dataset)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            os.makedirs('saved_models', exist_ok=True)
            torch.save(deblur_net.state_dict(), 'saved_models/deblur_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Обучение основной сети деблюринга с опциональным использованием адаптивного блока")
    parser.add_argument('--use_adaptive', action='store_true', help='Использовать обученный адаптивный блок')
    parser.add_argument('--freeze_adaptive', action='store_true',
                        help='Заморозить веса адаптивного блока при дообучении основной сети')
    args = parser.parse_args()
    train_deblur_model(use_adaptive=args.use_adaptive, freeze_adaptive=args.freeze_adaptive)
