import copy
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List

# Third-party imports
try:
    import yaml
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml"])
    import yaml

# Local imports
try:
    from project.utils.logging_utils import setup_logger
except ModuleNotFoundError:
    import logging

    def setup_logger(name: str):
        logger = logging.getLogger(name)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

logger = setup_logger("configuration")

class Configuration:
    """Класс для централизованного управления конфигурацией"""

    _instance = None

    def __new__(cls, config_path: str = None):
        if cls._instance is None:
            cls._instance = super(Configuration, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: str = None):
        if self._initialized:
            return

        # Базовый путь к конфигурации
        self.config_path = config_path or os.getenv("CONFIG_PATH", "config")

        # Конфигурация
        self.config = {}

        # Время последнего обновления
        self.last_update = 0

        # Загружаем конфигурацию
        self.load_config()

        # Переменные окружения
        self.env_prefix = "BOT_"

        # Применяем переменные окружения
        self.apply_environment_variables()

        self._initialized = True
        logger.info("Configuration initialized")

    def load_config(self) -> bool:
        """
        Загружает конфигурацию из файлов

        Returns:
            bool: True, если загрузка успешна, иначе False
        """
        try:
            # Проверяем, существует ли путь
            if not os.path.exists(self.config_path):
                logger.warning("Config path not found: {self.config_path}")
                return False

            # Загружаем базовую конфигурацию
            self.config = {}

            # Если путь это каталог, загружаем все файлы
            if os.path.isdir(self.config_path):
                for root, _, files in os.walk(self.config_path):
                    for file in files:
                        if file.endswith((".json", ".yaml", ".yml")):
                            file_path = os.path.join(root, file)
                            self._load_file(file_path)
            else:
                # Иначе загружаем один файл
                self._load_file(self.config_path)

            # Обновляем время последнего обновления
            self.last_update = time.time()

            logger.info("Configuration loaded from {self.config_path}")
            return True

        except Exception as e:
            logger.error("Error loading configuration: {str(e)}")
            return False

    def _load_file(self, file_path: str):
        """
        Загружает конфигурацию из файла

        Args:
            file_path: Путь к файлу
        """
        try:
            # Определяем формат файла
            file_extension = os.path.splitext(file_path)[1].lower()

            # Читаем файл
            with open(file_path, "r", encoding="utf-8") as f:
                if file_extension == ".json":
                    # JSON формат
                    config_data = json.load(f)
                elif file_extension in [".yaml", ".yml"]:
                    # YAML формат
                    config_data = yaml.safe_load(f)
                else:
                    logger.warning("Unsupported file format: {file_path}")
                    return

            # Определяем секцию конфигурации на основе имени файла
            file_name = os.path.basename(file_path)
            section_name = os.path.splitext(file_name)[0]

            # Добавляем данные в конфигурацию
            if section_name == "config":
                # Основная конфигурация
                self.config.update(config_data)
            else:
                # Секция конфигурации
                self.config[section_name] = config_data

            logger.debug("Loaded configuration from {file_path}")
        except Exception as e:
            logger.error("Error loading configuration from {file_path}: {str(e)}")

    def apply_environment_variables(self):
        """Применяет переменные окружения к конфигурации"""
        # Получаем все переменные окружения с префиксом
        env_vars = {
            key: value
            for key, value in os.environ.items()
            if key.startswith(self.env_prefix)
        }

        for key, value in env_vars.items():
            # Убираем префикс
            config_key = key[len(self.env_prefix):].lower()
            # Разбиваем ключ на части по подчеркиванию
            parts = config_key.split("_")
            # Применяем значение к конфигурации
            self._set_nested_value(parts, value)

        logger.debug("Applied {len(env_vars)} environment variables")

    def _set_nested_value(self, parts: List[str], value: str, config: Dict = None):
        """
        Устанавливает вложенное значение в конфигурации

        Args:
            parts: Части ключа
            value: Значение
            config: Словарь конфигурации
        """
        if config is None:
            config = self.config

        # Если осталась только одна часть, устанавливаем значение
        if len(parts) == 1:
            # Преобразуем строковое значение:
            config[parts[0]] = self._parse_value(value)
            return

        # Создаем вложенные словари, если их нет
        if parts[0] not in config or not isinstance(config[parts[0]], dict):
            config[parts[0]] = {}

        # Рекурсивно устанавливаем значение
        self._set_nested_value(parts[1:], value, config[parts[0]])

    def _parse_value(self, value: str) -> Any:
        """
        Преобразует строковое значение в соответствующий тип

        Args:
            value: Строковое значение

        Returns:
            Any: Преобразованное значение
        """
        # Пробуем преобразовать в число
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass

        # Пробуем преобразовать в булево значение
        if value.lower() in ["true", "yes", "1"]:
            return True
        elif value.lower() in ["false", "no", "0"]:
            return False

        # Пробуем преобразовать в список или словарь
        if value.startswith("{") and value.endswith("}"):
            try:
                return json.loads(value)
            except:
                pass
        elif value.startswith("[") and value.endswith("]"):
            try:
                return json.loads(value)
            except:
                pass

        # Возвращаем строку
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Возвращает значение из конфигурации

        Args:
            key: Ключ конфигурации (с точками для вложенных значений)
            default: Значение по умолчанию

        Returns:
            Any: Значение из конфигурации или значение по умолчанию
        """
        # Разбиваем ключ на части по точке
        parts = key.split(".")

        # Получаем значение
        value = self.config
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value

    def set(self, key: str, value: Any):
        """
        Устанавливает значение в конфигурации

        Args:
            key: Ключ конфигурации (с точками для вложенных значений)
            value: Значение
        """
        # Разбиваем ключ на части по точке
        parts = key.split(".")

        # Устанавливаем значение
        self._set_nested_value(parts, value)

    def save(self, file_path: str = None) -> bool:
        """
        Сохраняет конфигурацию в файл

        Args:
            file_path: Путь к файлу

        Returns:
            bool: True, если сохранение успешно, иначе False
        """
        try:
            # Определяем путь к файлу
            if file_path is None:
                if os.path.isdir(self.config_path):
                    file_path = os.path.join(self.config_path, "config.json")
                else:
                    file_path = self.config_path

            # Определяем формат файла
            file_extension = os.path.splitext(file_path)[1].lower()

            # Создаем директорию, если её нет
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

            # Сохраняем конфигурацию
            with open(file_path, "w", encoding="utf-8") as f:
                if file_extension == ".json":
                    # JSON формат
                    json.dump(self.config, f, indent=2)
                elif file_extension in [".yaml", ".yml"]:
                    # YAML формат
                    yaml.dump(self.config, f, default_flow_style=False)
                else:
                    # По умолчанию JSON
                    json.dump(self.config, f, indent=2)

            # Обновляем время последнего обновления
            self.last_update = time.time()

            logger.info("Configuration saved to {file_path}")
            return True

        except Exception as e:
            logger.error("Error saving configuration: {str(e)}")
            return False

    def reload(self) -> bool:
        """
        Перезагружает конфигурацию

        Returns:
            bool: True, если перезагрузка успешна, иначе False
        """
        return self.load_config()

    def get_section(self, section: str) -> Dict:
        """
        Возвращает секцию конфигурации

        Args:
            section: Название секции

        Returns:
            Dict: Секция конфигурации
        """
        return copy.deepcopy(self.config.get(section, {}))

    def update_section(self, section: str, config: Dict):
        """
        Обновляет секцию конфигурации

        Args:
            section: Название секции
            config: Новая конфигурация
        """
        if section not in self.config:
            self.config[section] = {}
        self.config[section].update(config)

    def to_dict(self) -> Dict:
        """
        Возвращает всю конфигурацию в виде словаря

        Returns:
            Dict: Конфигурация
        """
        return copy.deepcopy(self.config)

    def check_updates(self) -> bool:
        """
        Проверяет, были ли обновления в файлах конфигурации

        Returns:
            bool: True, если были обновления, иначе False
        """
        # Проверяем, существует ли путь
        if not os.path.exists(self.config_path):
            return False

        # Проверяем время последнего изменения
        if os.path.isdir(self.config_path):
            # Для каталога проверяем все файлы
            for root, _, files in os.walk(self.config_path):
                for file in files:
                    if file.endswith((".json", ".yaml", ".yml")):
                        file_path = os.path.join(root, file)
                        mtime = os.path.getmtime(file_path)
                        if mtime > self.last_update:
                            return True
        else:
            # Для файла проверяем его время изменения
            mtime = os.path.getmtime(self.config_path)
            if mtime > self.last_update:
                return True

        return False

    def get_last_update_time(self) -> str:
        """
        Возвращает время последнего обновления конфигурации

        Returns:
            str: Время последнего обновления
        """
        return datetime.fromtimestamp(self.last_update).strftime("%Y-%m-%d %H:%M:%S")

# Create a module-level variable
_config = None

def get_config():
    """
    Get or initialize the configuration singleton.
    
    Returns:
        Configuration instance
    """
    global _config
    if _config is None:
        _config = Configuration()
    return _config

def load_config(config_path: str = None) -> Configuration:
    """
    Загружает конфигурацию из указанного пути

    Args:
        config_path: Путь к конфигурации

    Returns:
        Configuration: Экземпляр конфигурации
    """
    global _config

    _config = Configuration(config_path)
    return _config

def get_config_value(key: str, default: Any = None) -> Any:
    """
    Возвращает значение из конфигурации

    Args:
        key: Ключ конфигурации (с точками для вложенных значений)
        default: Значение по умолчанию

    Returns:
        Any: Значение из конфигурации или значение по умолчанию
    """
    config = get_config()
    return config.get(key, default)

def set_config_value(key: str, value: Any):
    """
    Устанавливает значение в конфигурации

    Args:
        key: Ключ конфигурации (с точками для вложенных значений)
        value: Значение
    """
    config = get_config()
    config.set(key, value)