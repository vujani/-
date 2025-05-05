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
# Определение базовой сети деблюринга
##################################

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


##################################
# Импорт адаптивного блока из файла adaptive_deblur.py
##################################

from adaptive_deblur import AdaptiveDeblurBlock


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
    gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
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
# Датасет DIV2K для деблюринга
##################################

class DIV2KDeblurDataset(Dataset):
    """
    Датасет DIV2K для устранения размытия.
    Если задан blur_kernel, изображение обрабатывается свёрткой, иначе имитируется размытие с шумом.
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


##################################
# Основная тестовая функция
##################################

def main(use_adaptive=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка основной модели деблюринга
    deblur_model = BasicDeblurNet().to(device)
    model_path = "saved_models/deblur_model.pth"
    if os.path.exists(model_path):
        deblur_model.load_state_dict(torch.load(model_path, map_location=device))
        deblur_model.eval()
        print("Основная модель загружена из", model_path)
    else:
        print("Основная модель не найдена.")
        return

    # Если требуется, загружаем адаптивный блок
    adaptive_block = None
    if use_adaptive:
        adaptive_block = AdaptiveDeblurBlock().to(device)
        adaptive_model_path = "saved_models/adaptive_deblur_model.pth"
        if os.path.exists(adaptive_model_path):
            adaptive_block.load_state_dict(torch.load(adaptive_model_path, map_location=device))
            adaptive_block.eval()
            print("Адаптивный блок загружен из", adaptive_model_path)
        else:
            print("Адаптивный блок не найден, тест будет выполнен без него.")
            adaptive_block = None

    # Трансформации для датасета
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Определение ядра размытия (например, 5x5 усредняющее)
    blur_kernel = torch.tensor([[1.0 / 25 for _ in range(5)] for _ in range(5)], dtype=torch.float32)

    # Инициализация датасета (используется папка DIV2K/train)
    dataset = DIV2KDeblurDataset(root='DIV2K/train', transform=transform, blur_kernel=blur_kernel)

    # Выбор случайного образца
    idx = torch.randint(0, len(dataset), (1,)).item()
    blurred, clean = dataset[idx]
    blurred = blurred.unsqueeze(0).to(device)
    clean = clean.unsqueeze(0).to(device)

    # Если используется адаптивный блок, обрабатываем заблюренное изображение через него
    if adaptive_block is not None:
        with torch.no_grad():
            processed = adaptive_block(blurred)
    else:
        processed = blurred

    # Основная модель деблюринга
    with torch.no_grad():
        deblurred = deblur_model(processed)

    # Вычисляем метрики
    psnr_val = calculate_psnr(deblurred, clean)
    ssim_val = ssim(deblurred, clean)
    print(f"PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")

    # Преобразование тензоров в изображения для отображения
    to_pil = transforms.ToPILImage()
    blurred_img = to_pil(blurred.squeeze(0).cpu())
    deblurred_img = to_pil(deblurred.squeeze(0).cpu())
    clean_img = to_pil(clean.squeeze(0).cpu())

    # Отображение результатов
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(blurred_img)
    axs[0].set_title("Blurred Image")
    axs[0].axis("off")
    axs[1].imshow(deblurred_img)
    axs[1].set_title(f"Deblurred Image\nPSNR: {psnr_val:.2f} dB\nSSIM: {ssim_val:.4f}")
    axs[1].axis("off")
    axs[2].imshow(clean_img)
    axs[2].set_title("Clean Image")
    axs[2].axis("off")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Тест модели деблюринга с опциональным адаптивным блоком")
    parser.add_argument('--use_adaptive', action='store_true', help='Использовать адаптивный блок')
    args = parser.parse_args()
    main(use_adaptive=args.use_adaptive)

