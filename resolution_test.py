import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse


##################################
# Датасет для суперразрешения
##################################
class DIV2KSuperResolutionDataset(Dataset):
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
        c, H, W = hr_image.shape  # ожидание формы (C, H, W)
        lr_H, lr_W = H // self.scale_factor, W // self.scale_factor
        # Создаем LR-версию через downscale и последующий bicubic апскейл
        lr_image = nn.functional.interpolate(hr_image.unsqueeze(0), size=(lr_H, lr_W),
                                             mode='bicubic', align_corners=False)
        lr_upscaled = nn.functional.interpolate(lr_image, size=(H, W),
                                                mode='bicubic', align_corners=False)
        lr_upscaled = lr_upscaled.squeeze(0)
        return lr_upscaled, hr_image


##################################
# Импорт моделей
##################################
# Базовая сеть суперразрешения
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


# Импорт адаптивного блока суперразрешения из файла resolution_adaptive.py
from resolution_adaptive import AdaptiveSuperResolutionBlock


##################################
# Функция для вычисления масок
##################################
def compute_masks(x, mask_net):
    """
    Вычисляет low и high маски для изображения x.
    x: тензор [B, 3, H, W] с значениями в [0,1].
    mask_net: блок, отвечающий за генерацию маски (из адаптивного блока).
    """
    fft = torch.fft.fft2(x, norm="ortho")
    fft_shifted = torch.fft.fftshift(fft)
    mag = torch.abs(fft_shifted)
    mag_avg = torch.mean(mag, dim=1, keepdim=True)
    low_mask = mask_net(mag_avg)
    low_mask = low_mask.expand(-1, x.shape[1], -1, -1)
    high_mask = 1.0 - low_mask
    return low_mask, high_mask


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
    gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                          for x in range(window_size)])
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
# Основная тестовая функция
##################################
def main(use_adaptive=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка базовой модели суперразрешения
    sr_model = BasicSuperResolutionNet().to(device)
    sr_model_path = "saved_models/resolution_sr_model.pth"
    if os.path.exists(sr_model_path):
        sr_model.load_state_dict(torch.load(sr_model_path, map_location=device))
        sr_model.eval()
        print("SR модель загружена из", sr_model_path)
    else:
        print("SR модель не найдена.")
        return

    # Загрузка адаптивного блока, если используется
    adaptive_block = None
    if use_adaptive:
        adaptive_block = AdaptiveSuperResolutionBlock().to(device)
        adaptive_model_path = "saved_models/resolution_adaptive_model.pth"
        if os.path.exists(adaptive_model_path):
            adaptive_block.load_state_dict(torch.load(adaptive_model_path, map_location=device))
            adaptive_block.eval()
            print("Адаптивный блок загружен из", adaptive_model_path)
        else:
            print("Адаптивный блок не найден, тест будет выполнен без него.")
            adaptive_block = None

    # Трансформация для датасета
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Инициализация датасета (используем папку DIV2K/train)
    dataset = DIV2KSuperResolutionDataset(root='DIV2K/train', scale_factor=2, transform=transform)
    idx = torch.randint(0, len(dataset), (1,)).item()
    lr, hr = dataset[idx]
    lr = lr.unsqueeze(0).to(device)  # LR изображение (bicubic апскейленное)
    hr = hr.unsqueeze(0).to(device)  # Ground Truth HR

    # Если используется адаптивный блок, пропускаем LR через него
    if adaptive_block is not None:
        with torch.no_grad():
            adaptive_output = adaptive_block(lr)
    else:
        adaptive_output = lr

    # Прогон через модель суперразрешения
    with torch.no_grad():
        sr_output = sr_model(adaptive_output)

    # Вычисляем маски для LR и SR изображений, если адаптивный блок используется
    if adaptive_block is not None:
        low_mask_lr, high_mask_lr = compute_masks(lr, adaptive_block.mask_net)
        low_mask_sr, high_mask_sr = compute_masks(sr_output, adaptive_block.mask_net)
    else:
        low_mask_lr = high_mask_lr = low_mask_sr = high_mask_sr = torch.zeros_like(lr[:, :1])

    # Вычисляем PSNR и SSIM для SR изображения
    psnr_val = calculate_psnr(sr_output, hr)
    ssim_val = ssim(sr_output, hr)
    print(f"PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")

    # Преобразование тензоров в PIL-изображения для отображения
    to_pil = transforms.ToPILImage()
    lr_img = to_pil(lr.squeeze(0).cpu())
    sr_img = to_pil(sr_output.squeeze(0).cpu())
    hr_img = to_pil(hr.squeeze(0).cpu())
    high_mask_lr_img = to_pil(high_mask_lr.squeeze(0).cpu())
    low_mask_lr_img = to_pil(low_mask_lr.squeeze(0).cpu())
    high_mask_sr_img = to_pil(high_mask_sr.squeeze(0).cpu())
    low_mask_sr_img = to_pil(low_mask_sr.squeeze(0).cpu())

    # Отображение исходных изображений
    fig1, axs1 = plt.subplots(1, 3, figsize=(18, 6))
    axs1[0].imshow(lr_img)
    axs1[0].set_title("LR (Входное)")
    axs1[0].axis("off")
    axs1[1].imshow(sr_img)
    axs1[1].set_title(f"SR (Выходное)\nPSNR: {psnr_val:.2f} dB\nSSIM: {ssim_val:.4f}")
    axs1[1].axis("off")
    axs1[2].imshow(hr_img)
    axs1[2].set_title("HR (Ground Truth)")
    axs1[2].axis("off")
    plt.tight_layout()
    plt.show()

    # Отображение масок: два ряда – первый для LR, второй для SR; два столбца – High Mask и Low Mask
    fig2, axs2 = plt.subplots(2, 2, figsize=(12, 10))
    axs2[0, 0].imshow(high_mask_lr_img, cmap="gray")
    axs2[0, 0].set_title("LR: High Mask")
    axs2[0, 0].axis("off")
    axs2[0, 1].imshow(low_mask_lr_img, cmap="gray")
    axs2[0, 1].set_title("LR: Low Mask")
    axs2[0, 1].axis("off")
    axs2[1, 0].imshow(high_mask_sr_img, cmap="gray")
    axs2[1, 0].set_title("SR: High Mask")
    axs2[1, 0].axis("off")
    axs2[1, 1].imshow(low_mask_sr_img, cmap="gray")
    axs2[1, 1].set_title("SR: Low Mask")
    axs2[1, 1].axis("off")
    plt.suptitle("Маски для LR и SR изображений", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Тест суперразрешения с отображением масок (ряды: LR и SR, столбцы: High и Low Mask)")
    parser.add_argument('--use_adaptive', action='store_true', help='Использовать адаптивный блок суперразрешения')
    args = parser.parse_args()
    main(use_adaptive=args.use_adaptive)
