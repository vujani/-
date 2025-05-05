import os
import random
import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from skimage.color import rgb2gray

# Импорт моделей
from train import DenoiseNet, calculate_psnr  # Модель денойзинга и функция PSNR
from adaptive import AdaptiveBlock  # Adaptive блок для денойзинга
from deblur import DeblurNet, DeblurConfig  # Модель де-блюринга и её конфигурация
from choise import DistortionClassifier  # Классификатор для определения искажения

# Константы
IMG_SIZE = 256
NOISE_LEVEL = 0.15
BLUR_RADIUS = 2
DATA_DIR = "DIV2K/train"
CLASSIFIER_MODEL_PATH = os.path.join("saved_models", "distortion_classifier.pth")
DENOISE_MODEL_PATH = os.path.join("saved_models", "denoise_model.pth")
DEBLUR_MODEL_PATH = os.path.join("saved_models", "deblur_model.pth")


def get_random_image(dataset_dir=DATA_DIR, img_size=IMG_SIZE):
    valid_exts = ('.png', '.jpg', '.jpeg')
    img_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)
                 if f.lower().endswith(valid_exts)]
    if not img_files:
        raise ValueError("Нет изображений в указанной директории!")
    img_path = random.choice(img_files)
    print(f"Выбрано изображение: {img_path}")
    image = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    return transform(image)


def degrade_image(img, noise_level=NOISE_LEVEL, blur_radius=BLUR_RADIUS):
    # Случайно выбираем искажение: шум (50%) или размытие (50%)
    if random.random() < 0.5:
        noise = noise_level * torch.randn_like(img)
        degraded = torch.clamp(img + noise, 0.0, 1.0)
        true_artifact = "noisy"
    else:
        pil_img = transforms.ToPILImage()(img.cpu())
        pil_blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        degraded = transforms.ToTensor()(pil_blurred)
        true_artifact = "blurred"
    return degraded, true_artifact


def classify_distortion(img_tensor, device):
    # Загружаем классификатор и переводим вход на нужный device
    classifier = DistortionClassifier().to(device)
    classifier.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=device))
    classifier.eval()
    with torch.no_grad():
        outputs = classifier(img_tensor.unsqueeze(0).to(device))
        pred = torch.argmax(outputs, dim=1).item()
    # Предполагаем: 0 - blurred, 1 - noisy
    predicted_artifact = "blurred" if pred == 0 else "noisy"
    return predicted_artifact


def process_image(img_tensor, artifact_type, device):
    img_input = img_tensor.unsqueeze(0).to(device)
    if artifact_type == "noisy":
        denoise_net = DenoiseNet().to(device)
        denoise_net.load_state_dict(torch.load(DENOISE_MODEL_PATH, map_location=device))
        denoise_net.eval()
        adaptive_block = None
        adaptive_path = os.path.join("saved_models", "adaptive_model.pth")
        if os.path.exists(adaptive_path):
            adaptive_block = AdaptiveBlock().to(device)
            adaptive_block.load_state_dict(torch.load(adaptive_path, map_location=device))
            adaptive_block.eval()
            with torch.no_grad():
                processed_input = adaptive_block(img_input)
        else:
            processed_input = img_input
        with torch.no_grad():
            output = denoise_net(processed_input)
    elif artifact_type == "blurred":
        config = DeblurConfig()
        deblur_net = DeblurNet(img_size=config.img_size).to(device)
        deblur_net.load_state_dict(torch.load(DEBLUR_MODEL_PATH, map_location=device))
        deblur_net.eval()
        with torch.no_grad():
            output = deblur_net(img_input)
    else:
        output = img_input
    return output.squeeze(0).cpu()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Загружаем изображение
    original = get_random_image().to(device)
    # Применяем искажение
    degraded, true_artifact = degrade_image(original)
    # Классифицируем искажение (подключаем choise.py)
    predicted_artifact = classify_distortion(degraded, device)
    print(f"Истинное искажение: {true_artifact}, Предсказанное: {predicted_artifact}")

    # Для восстановления используем модель, соответствующую предсказанному типу
    restored = process_image(degraded, predicted_artifact, device)

    # Вычисляем метрики
    original_np = original.cpu().permute(1, 2, 0).numpy()
    degraded_np = degraded.cpu().permute(1, 2, 0).numpy()
    restored_np = restored.cpu().permute(1, 2, 0).numpy()
    psnr_degraded = psnr(original_np, degraded_np)
    psnr_restored = psnr(original_np, restored_np)
    original_gray = rgb2gray(original_np)
    restored_gray = rgb2gray(restored_np)
    ssim_restored = ssim(original_gray, restored_gray, data_range=1.0)

    # Визуализация
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(np.clip(original_np, 0, 1))
    axes[0].set_title("Оригинальное изображение")
    axes[0].axis("off")

    axes[1].imshow(np.clip(degraded_np, 0, 1))
    axes[1].set_title(f"Искажённое изображение\nИстинное: {true_artifact}\nPSNR: {psnr_degraded:.2f} дБ")
    axes[1].axis("off")

    mode_used = "denoise" if predicted_artifact == "noisy" else "deblur"
    axes[2].imshow(np.clip(restored_np, 0, 1))
    axes[2].set_title(
        f"Восстановленное изображение\nМодель: {mode_used}\nPSNR: {psnr_restored:.2f} дБ\nSSIM: {ssim_restored:.4f}")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
