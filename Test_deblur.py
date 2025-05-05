import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from adaptive_deblur import AdaptiveDeblurBlock, DIV2KDataset
from deblur import DeblurNet, calculate_psnr, ssim


# Конфигурация тестирования
class TestConfig:
    adaptive_model_path = 'saved_models/adaptive_deblur_model.pth'
    deblur_model_path = 'saved_models/deblur_model.pth'
    dataset_root = 'DIV2K/train'
    image_size = (256, 256)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    show_details = True


def test_deblur():
    config = TestConfig()

    # Загрузка моделей
    adaptive_block = AdaptiveDeblurBlock().to(config.device)
    if os.path.exists(config.adaptive_model_path):
        adaptive_block.load_state_dict(torch.load(config.adaptive_model_path, map_location=config.device))
        adaptive_block.eval()
        print("Адаптивная модель загружена успешно")
    else:
        print("Файл адаптивной модели не найден")
        return

    deblur_net = DeblurNet().to(config.device)
    if os.path.exists(config.deblur_model_path):
        deblur_net.load_state_dict(torch.load(config.deblur_model_path, map_location=config.device))
        deblur_net.eval()
        print("Модель деблюринга загружена успешно")
    else:
        print("Файл модели деблюринга не найден")
        return

    # Загрузка датасета
    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
    ])

    dataset = DIV2KDataset(root=config.dataset_root, transform=transform)
    test_image_idx = np.random.randint(0, len(dataset))
    blurred_image, clean_image = dataset[test_image_idx]

    # Добавление размытия (если не используется в датасете)
    # blurred_image = apply_blur(clean_image)

    # Применение адаптивного блока
    with torch.no_grad():
        adaptive_output = adaptive_block(blurred_image.unsqueeze(0).to(config.device))

    # Применение деблюринга
    with torch.no_grad():
        deblurred_image = deblur_net(adaptive_output)

    # Конвертация в numpy для визуализации
    blurred_image_np = blurred_image.numpy().transpose(1, 2, 0)
    clean_image_np = clean_image.numpy().transpose(1, 2, 0)
    deblurred_image_np = deblurred_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)

    # Рассчет метрик
    psnr_deblurred = calculate_psnr(torch.tensor(deblurred_image_np.transpose(2, 0, 1)),
                                    torch.tensor(clean_image_np.transpose(2, 0, 1)))
    ssim_deblurred = ssim(torch.tensor(deblurred_image_np.transpose(2, 0, 1)).unsqueeze(0),
                          torch.tensor(clean_image_np.transpose(2, 0, 1)).unsqueeze(0))

    # Визуализация
    plt.figure(figsize=(15, 10))

    # Размытое изображение
    plt.subplot(1, 3, 1)
    plt.imshow(blurred_image_np)
    plt.title("Размытое изображение")
    plt.axis('off')

    # Очищенное изображение
    plt.subplot(1, 3, 2)
    plt.imshow(deblurred_image_np)
    plt.title(f"Деблюренное (PSNR: {psnr_deblurred:.2f} dB, SSIM: {ssim_deblurred:.4f})")
    plt.axis('off')

    # Оригинальное изображение
    plt.subplot(1, 3, 3)
    plt.imshow(clean_image_np)
    plt.title("Оригинальное изображение")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_deblur()