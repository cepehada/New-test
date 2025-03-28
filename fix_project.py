#!/usr/bin/env python3
"""
Автоматический исправитель кода для проекта.
Исправляет распространенные ошибки, найденные PyLint:
- Неправильный порядок импортов
- Неиспользуемые импорты
- Форматирование логов (f-строки)
"""
import re
import subprocess
from pathlib import Path

# Список файлов для исключения
EXCLUDE_DIRS = ["venv", ".git", "__pycache__"]


# Функция для проверки, нужно ли исключить путь
def should_exclude(path):
    """Проверяет, нужно ли исключить путь из обработки"""
    for exclude_dir in EXCLUDE_DIRS:
        if exclude_dir in path:
            return True
    return False


def install_dependencies():
    """Устанавливает необходимые зависимости"""
    print("Установка необходимых инструментов...")
    subprocess.run(["pip", "install", "isort", "autoflake", "black"], check=True)
    print("Зависимости установлены")


def fix_imports():
    """Исправляет порядок импортов и удаляет неиспользуемые импорты"""
    print("Исправление порядка импортов...")

    # Рекурсивно ищем все Python-файлы
    python_files = []
    for path in Path(".").rglob("*.py"):
        if not should_exclude(str(path)):
            python_files.append(str(path))

    if not python_files:
        print("Python-файлы не найдены")
        return

    print(f"Найдено {len(python_files)} Python-файлов")

    # Сортировка импортов
    print("Применение isort...")
    subprocess.run(["isort", "--profile", "black"] + python_files, check=True)

    # Удаление неиспользуемых импортов
    print("Удаление неиспользуемых импортов...")
    subprocess.run(
        ["autoflake", "--in-place", "--remove-all-unused-imports", "--recursive", "."],
        check=True,
    )

    print("Импорты исправлены")


def fix_logging():
    """Исправляет форматирование логов (заменяет f-строки)"""
    print("Исправление форматирования логов...")

    # Паттерны для замены f-строк в логах
    patterns = [
        # Логи с аргументами: logger.info(f"text {var}")
        (
            re.compile(
                r'(log(?:ger)?\.(?:debug|info|warning|error|critical))\(f(["\'])(.*?)(?:\2)'
            ),
            r"\1(\2\3\2 %",
        ),
        # Логи с аргументами и дополнительными параметрами
        (
            re.compile(
                r'(log(?:ger)?\.(?:debug|info|warning|error|critical))\(f(["\'])(.*?)(?:\2)(,\s*exc_info=.*?)?\)'
            ),
            r"\1(\2\3\2 %\4)",
        ),
    ]

    count = 0
    for path in Path(".").rglob("*.py"):
        if should_exclude(str(path)):
            continue

        # Читаем содержимое файла
        try:
            with open(path, "r", encoding="utf-8") as file:
                content = file.read()

            # Применяем паттерны замены
            original_content = content
            for pattern, replacement in patterns:
                content = pattern.sub(replacement, content)

            # Если были изменения, записываем файл
            if content != original_content:
                with open(path, "w", encoding="utf-8") as file:
                    file.write(content)
                count += 1
                print(f"Исправлен файл: {path}")
        except Exception as e:
            print(f"Ошибка при обработке {path}: {e}")

    print(f"Исправлено {count} файлов с форматированием логов")


def fix_exception_handling():
    """Находит слишком общие обработчики исключений и выводит предупреждение"""
    print("Проверка обработчиков исключений...")

    pattern = re.compile(r"except\s+Exception\s+as\s+\w+:")
    exceptions = []

    for path in Path(".").rglob("*.py"):
        if should_exclude(str(path)):
            continue

        try:
            with open(path, "r", encoding="utf-8") as file:
                content = file.read()

            matches = pattern.findall(content)
            if matches:
                exceptions.append((str(path), len(matches)))
        except Exception as e:
            print(f"Ошибка при проверке {path}: {e}")

    if exceptions:
        print("\nНайдены слишком общие обработчики исключений:")
        for file_path, count in exceptions:
            print(f"  {file_path}: {count} исключений")
        print(
            "\nРекомендация: Замените 'except Exception' на более конкретные типы исключений"
        )


def format_code():
    """Форматирует код с помощью black"""
    print("Форматирование кода с помощью black...")

    # Применяем black ко всем Python-файлам
    try:
        subprocess.run(["black", "."], check=True)
        print("Форматирование кода завершено")
    except Exception as e:
        print(f"Ошибка при форматировании кода: {e}")


def main():
    """Основная функция"""
    print("Начинаем исправление проблем в проекте...")

    # Устанавливаем зависимости
    install_dependencies()

    # Исправляем импорты
    fix_imports()

    # Исправляем логи
    fix_logging()

    # Проверяем обработку исключений
    fix_exception_handling()

    # Форматируем код
    format_code()

    print("\nЗавершено! Большинство проблем исправлено автоматически.")
    print(
        """
Оставшиеся проблемы, которые требуют ручного вмешательства:
1. Синтаксические ошибки (неправильные отступы, незакрытые скобки)
2. Проблемы с импортом (отсутствующие модули)
3. Сложные функции с большим количеством веток и переменных
    """
    )


if __name__ == "__main__":
    main()
