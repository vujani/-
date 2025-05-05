import subprocess
import sys


def run_command(command):
    print(f"Запуск: {' '.join(command)}")
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("Ошибка при выполнении команды:")
        print(result.stderr)
        sys.exit(result.returncode)
    else:
        print(result.stdout)


def main():
    # Запускаем обучение адаптивного блока
    print("Запуск обучения адаптивного блока (adaptive.py)...")
    run_command(["python", "adaptive.py"])

    # После успешного обучения адаптивного блока запускаем обучение основной сети с замороженными весами адаптивного блока
    print("Запуск обучения основной сети (train.py) с использованием замороженных весов адаптивного блока...")
    run_command(["python", "train.py", "--use_adaptive", "--freeze_adaptive"])


if __name__ == "__main__":
    main()
