"""
Скрипт для автоматического исправления неправильно оформленных f-строк в проекте.
Исправляет конструкции типа f"текст {переменная}" на правильные f"текст {переменная}".
"""

import re
import os
import glob
from pathlib import Path

# Шаблон для поиска неправильных f-строк
pattern = re.compile(r'(["\'])(.*?{.*?}.*?)(\1)\s*%')

# Функция для исправления f-строк в файле
def fix_f_strings(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Проверяем, требуется ли исправление
    if not pattern.search(content):
        return False
    
    # Исправляем все найденные строки
    modified_content = pattern.sub(r'f\1\2\3', content)
    
    # Записываем изменения
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(modified_content)
    
    return True

# Находим все Python-файлы в проекте
python_files = glob.glob('/workspaces/New-test/**/*.py', recursive=True)

# Счетчики
fixed_count = 0
total_files = len(python_files)

print(f"Начато исправление f-строк в {total_files} файлах...")

# Обрабатываем каждый файл
for file_path in python_files:
    if fix_f_strings(file_path):
        fixed_count += 1
        print(f"Исправлен файл: {file_path}")

print(f"\nИсправление завершено. Исправлено {fixed_count} файлов из {total_files}.")
