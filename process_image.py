import os
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse

# Импорт функции анализа артефактов
from artifact_classifier import analyze_artifacts


##############################################
# Функции для симуляции артефактов (для тестирования)
##############################################
def add_noise(pil_image, noise_level=0.15):
    image = np.array(pil_image).astype(np.float32) / 255.0
    noise = np.random.normal(0, noise_level, image.shape)
    noisy = image + noise
    noisy = np.clip(noisy, 0, 1)
    noisy = (noisy * 255).astype(np.uint8)
    return Image.fromarray(noisy)


def add_blur(pil_image, kernel_size=5):
    image = np.array(pil_image)
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return Image.fromarray(blurred)


def add_combined(pil_image):
    return add_blur(add_noise(pil_image), kernel_size=5)


##############################################
# Модель BasicDeblurNet – используется для шумоподавления и деблюринга
##############################################
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
    Универсальная сеть для шумоподавления или деблюринга.
    Использует несколько residual-блоков и суммирует поправочный сигнал с входом.
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
        output = x + out
        return torch.clamp(output, 0.0, 1.0)


##############################################
# Модель суперразрешения
##############################################
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


##############################################
# Функция загрузки модели
##############################################
def load_model(model_class, weight_path):
    model = model_class()
    if os.path.exists(weight_path):
        state = torch.load(weight_path, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
        if missing_keys:
            print(f"Предупреждение: отсутствуют ключи: {missing_keys}")
        if unexpected_keys:
            print(f"Предупреждение: неожиданные ключи: {unexpected_keys}")
        print(f"Загружены веса из {weight_path} (strict=False)")
    else:
        print(f"Веса не найдены по {weight_path}. Модель будет работать в демо-режиме.")
    model.eval()
    return model



##############################################
# Импорт адаптивного блока суперразрешения
##############################################
from resolution_adaptive import AdaptiveSuperResolutionBlock


##############################################
# Функция обработки изображения
##############################################
def process_image(pil_image, device, denoise_model, deblur_model, sr_model):
    """
    Обрабатывает изображение:
      - Анализирует артефакты.
      - В зависимости от типа:
          "noisy": применяется шумоподавление,
          "blurred": применяется деблюринг,
          "combined": сначала шумоподавление, затем деблюринг.
      - Если "clean" – никаких изменений.
      - После этого всегда применяется суперразрешение.
    Возвращает SR изображение (тензор) и тип артефакта.
    """
    artifact_type = analyze_artifacts(pil_image)
    print(f"Классифицировано как: {artifact_type}")

    transform = transforms.ToTensor()  # значения в [0,1]
    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    processed = image_tensor.clone()
    if artifact_type == "noisy":
        processed = denoise_model(processed)
    elif artifact_type == "blurred":
        processed = deblur_model(processed)
    elif artifact_type == "combined":
        processed = denoise_model(processed)
        processed = deblur_model(processed)
    # Если "clean", оставляем без изменений

    sr_output = sr_model(processed)
    return sr_output, artifact_type


##############################################
# Основной тестовый блок
##############################################
def main(use_random_artifacts=False, num_samples=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Пути к весам моделей
    denoise_weights = "saved_models/denoise_model.pth"  # веса для BasicDeblurNet, обученные для шумоподавления
    deblur_weights = "saved_models/deblur_model.pth"  # веса для BasicDeblurNet, обученные для деблюринга
    sr_weights = "saved_models/resolution_sr_model.pth"  # веса для суперразрешения

    # Загружаем модели
    # Используем BasicDeblurNet для обоих: шумоподавления и деблюринга.
    denoise_model = load_model(BasicDeblurNet, denoise_weights).to(device)
    deblur_model = load_model(BasicDeblurNet, deblur_weights).to(device)
    sr_model = load_model(BasicSuperResolutionNet, sr_weights).to(device)

    # Папка с тестовыми изображениями
    test_folder = "DIV"
    image_files = [os.path.join(test_folder, f) for f in os.listdir(test_folder)
                   if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    if not image_files:
        print("Нет изображений в папке test_images.")
        return

    for i in range(num_samples):
        image_path = random.choice(image_files)
        orig_image = Image.open(image_path).convert("RGB")

        if use_random_artifacts:
            artifact_choice = random.choice(["clean", "noisy", "blurred", "combined"])
            print(f"Случайно выбрано искажение: {artifact_choice}")
            if artifact_choice == "noisy":
                test_image = add_noise(orig_image)
            elif artifact_choice == "blurred":
                test_image = add_blur(orig_image)
            elif artifact_choice == "combined":
                test_image = add_combined(orig_image)
            else:
                test_image = orig_image
        else:
            test_image = orig_image

        sr_output_tensor, artifact_type = process_image(test_image, device, denoise_model, deblur_model, sr_model)

        to_pil = transforms.ToPILImage()
        input_img = test_image
        output_img = to_pil(sr_output_tensor.squeeze(0).cpu())

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(input_img)
        axs[0].set_title(f"Входное изображение\n(Артефакт: {artifact_type})")
        axs[0].axis("off")
        axs[1].imshow(output_img)
        axs[1].set_title("Выходное изображение (с SR)")
        axs[1].axis("off")
        plt.suptitle(f"Обработка изображения, пример {i + 1}", fontsize=16)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Обработка изображения с выбором моделей шумоподавления/деблюринга и суперразрешением")
    parser.add_argument("--use_random_artifacts", action="store_true",
                        help="Применять случайные артефакты к изображению для теста")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Количество тестовых изображений для обработки")
    args = parser.parse_args()
    main(use_random_artifacts=args.use_random_artifacts, num_samples=args.num_samples)
model_class()
