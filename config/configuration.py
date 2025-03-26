import os
import json
import yaml
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import copy
import time
from datetime import datetime

from project.utils.logging_utils import setup_logger

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
te = 0
        # Загружаем конфигурацию
        self.load_config()        # Загружаем конфигурацию

        # Переменные окружения
        self.env_prefix = "BOT_"        # Переменные окружения

        # Применяем переменные окружения
        self.apply_environment_variables()        # Применяем переменные окружения
_variables()
        self._initialized = True
        self._initialized = True
        logger.info("Configuration initialized")
ized")
    def load_config(self) -> bool:
        """l:
        Загружает конфигурацию из файлов        """

        Returns:
            bool: True, если загрузка успешна, иначе False
        """ bool: True, если загрузка успешна, иначе False
        try:
            # Проверяем, существует ли путь        try:
            if not os.path.exists(self.config_path):оверяем, существует ли путь
                logger.warning(f"Config path not found: {self.config_path}")
                return False     logger.warning(f"Config path not found: {self.config_path}")
    return False
            # Загружаем базовую конфигурацию
            self.config = {}

            # Если путь это каталог, загружаем все файлы
            if os.path.isdir(self.config_path):            # Если путь это каталог, загружаем все файлы
                for root, _, files in os.walk(self.config_path):h):
                    for file in files: files in os.walk(self.config_path):
                        if file.endswith((".json", ".yaml", ".yml")):                    for file in files:
                            file_path = os.path.join(root, file)l", ".yml")):
                            self._load_file(file_path).join(root, file)
            else:
                # Иначе загружаем один файл
                self._load_file(self.config_path)

            # Обновляем время последнего обновления
            self.last_update = time.time()овляем время последнего обновления

            logger.info(f"Configuration loaded from {self.config_path}")
            return True            logger.info(f"Configuration loaded from {self.config_path}")

        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")        except Exception as e:
            return False
e
    def _load_file(self, file_path: str):
        """path: str):
        Загружает конфигурацию из файла
урацию из файла
        Args:
            file_path: Путь к файлу
        """ file_path: Путь к файлу
        try:
            # Определяем формат файла        try:
            file_extension = os.path.splitext(file_path)[1].lower() Определяем формат файла
h.splitext(file_path)[1].lower()
            # Читаем файл
            with open(file_path, "r", encoding="utf-8") as f:# Читаем файл
                if file_extension == ".json": encoding="utf-8") as f:
                    # JSON формат
                    config_data = json.load(f)                    # JSON формат
                elif file_extension in [".yaml", ".yml"]:g_data = json.load(f)
                    # YAML формат
                    config_data = yaml.safe_load(f)
                else: yaml.safe_load(f)
                    logger.warning(f"Unsupported file format: {file_path}")
                    returnmat: {file_path}")

            # Определяем секцию конфигурации на основе имени файла
            file_name = os.path.basename(file_path)яем секцию конфигурации на основе имени файла
            section_name = os.path.splitext(file_name)[0]
 os.path.splitext(file_name)[0]
            # Добавляем данные в конфигурацию
            if section_name == "config":
                # Основная конфигурация
                self.config.update(config_data)
            else:                self.config.update(config_data)
                # Секция конфигурации
                self.config[section_name] = config_data
e] = config_data
            logger.debug(f"Loaded configuration from {file_path}")
r.debug(f"Loaded configuration from {file_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {file_path}: {str(e)}")
            logger.error(f"Error loading configuration from {file_path}: {str(e)}")
    def apply_environment_variables(self):
        """Применяет переменные окружения к конфигурации"""    def apply_environment_variables(self):
        # Получаем все переменные окружения с префиксоме окружения к конфигурации"""
        env_vars = {
            key: value        env_vars = {
            for key, value in os.environ.items()
            if key.startswith(self.env_prefix)
        }

        for key, value in env_vars.items():
            # Убираем префикс
            config_key = key[len(self.env_prefix):].lower()
   config_key = key[len(self.env_prefix):].lower()
            # Разбиваем ключ на части по подчеркиванию
            parts = config_key.split("_")дчеркиванию
y.split("_")
            # Применяем значение к конфигурации
            self._set_nested_value(parts, value)            # Применяем значение к конфигурации

        logger.debug(f"Applied {len(env_vars)} environment variables")
        logger.debug(f"Applied {len(env_vars)} environment variables")
    def _set_nested_value(self, parts: List[str], value: str, config: Dict = None):
        """, value: str, config: Dict = None):
        Устанавливает вложенное значение в конфигурации        """

        Args:
            parts: Части ключа
            value: Значение parts: Части ключа
            config: Словарь конфигурации
        """            config: Словарь конфигурации
        if config is None:
            config = self.config
onfig
        # Если осталась только одна часть, устанавливаем значение
        if len(parts) == 1:сли осталась только одна часть, устанавливаем значение
            # Преобразуем строковое значение:
            config[parts[0]] = self._parse_value(value)вое значение
            return            config[parts[0]] = self._parse_value(value)

        # Создаем вложенные словари, если их нет
        if parts[0] not in config or not isinstance(config[parts[0]], dict): нет
            config[parts[0]] = {}fig[parts[0]], dict):
[parts[0]] = {}
        # Рекурсивно устанавливаем значение
        self._set_nested_value(parts[1:], value, config[parts[0]])

    def _parse_value(self, value: str) -> Any:
        """    def _parse_value(self, value: str) -> Any:
        Преобразует строковое значение в соответствующий тип

        Args:
            value: Строковое значение
 value: Строковое значение
        Returns:
            Any: Преобразованное значение        Returns:
        """ny: Преобразованное значение
        # Пробуем преобразовать в число
        try:        # Пробуем преобразовать в число
            if "." in value:
                return float(value)
            else:     return float(value)
                return int(value)
        except ValueError:    return int(value)
            pass

        # Пробуем преобразовать в булево значение
        if value.lower() in ["true", "yes", "1"]: булево значение
            return Truen ["true", "yes", "1"]:
        elif value.lower() in ["false", "no", "0"]:rn True
            return False        elif value.lower() in ["false", "no", "0"]:

        # Пробуем преобразовать в список или словарь
        if value.startswith("{") and value.endswith("}"):разовать в список или словарь
            try:("}"):
                return json.loads(value)
            except:                return json.loads(value)
                pass
        elif value.startswith("[") and value.endswith("]"):
            try:ue.startswith("[") and value.endswith("]"):
                return json.loads(value)
            except:urn json.loads(value)
                pass

        # Возвращаем строку
        return value
e
    def get(self, key: str, default: Any = None) -> Any:
        """    def get(self, key: str, default: Any = None) -> Any:
        Возвращает значение из конфигурации
начение из конфигурации
        Args:
            key: Ключ конфигурации (с точками для вложенных значений)
            default: Значение по умолчанию key: Ключ конфигурации (с точками для вложенных значений)

        Returns:
            Any: Значение из конфигурации или значение по умолчаниюns:
        """
        # Разбиваем ключ на части по точке
        parts = key.split(".")        # Разбиваем ключ на части по точке
key.split(".")
        # Получаем значение
        value = self.configолучаем значение
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]            if isinstance(value, dict) and part in value:
            else:ue[part]
                return default
ault
        return value

    def set(self, key: str, value: Any):
        """lue: Any):
        Устанавливает значение в конфигурации        """
т значение в конфигурации
        Args:
            key: Ключ конфигурации (с точками для вложенных значений)
            value: Значение key: Ключ конфигурации (с точками для вложенных значений)
        """
        # Разбиваем ключ на части по точке        """
        parts = key.split(".")биваем ключ на части по точке

        # Устанавливаем значение
        self._set_nested_value(parts, value)станавливаем значение
e)
    def save(self, file_path: str = None) -> bool:
        """    def save(self, file_path: str = None) -> bool:
        Сохраняет конфигурацию в файл

        Args:
            file_path: Путь к файлу
 file_path: Путь к файлу
        Returns:
            bool: True, если сохранение успешно, иначе False        Returns:
        """ool: True, если сохранение успешно, иначе False
        try:
            # Определяем путь к файлу        try:
            if file_path is None:ределяем путь к файлу
                if os.path.isdir(self.config_path):
                    file_path = os.path.join(self.config_path, "config.json")     if os.path.isdir(self.config_path):
                else:        file_path = os.path.join(self.config_path, "config.json")
                    file_path = self.config_path
elf.config_path
            # Определяем формат файла
            file_extension = os.path.splitext(file_path)[1].lower()
nsion = os.path.splitext(file_path)[1].lower()
            # Создаем директорию, если её нет
            directory = os.path.dirname(file_path)            # Создаем директорию, если её нет
            if directory and not os.path.exists(directory):me(file_path)
                os.makedirs(directory)
                os.makedirs(directory)
            # Сохраняем конфигурацию
            with open(file_path, "w", encoding="utf-8") as f:
                if file_extension == ".json":f:
                    # JSON формат.json":
                    json.dump(self.config, f, indent=2)                    # JSON формат
                elif file_extension in [".yaml", ".yml"]:onfig, f, indent=2)
                    # YAML формат
                    yaml.dump(self.config, f, default_flow_style=False)
                else:f.config, f, default_flow_style=False)
                    # По умолчанию JSON
                    json.dump(self.config, f, indent=2)
f.config, f, indent=2)
            # Обновляем время последнего обновления
            self.last_update = time.time()ем время последнего обновления
e()
            logger.info(f"Configuration saved to {file_path}")
            return True            logger.info(f"Configuration saved to {file_path}")

        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")        except Exception as e:
            return False}")
e
    def reload(self) -> bool:
        """
        Перезагружает конфигурацию
нфигурацию
        Returns:
            bool: True, если перезагрузка успешна, иначе False
        """ bool: True, если перезагрузка успешна, иначе False
        return self.load_config()
        return self.load_config()
    def get_section(self, section: str) -> Dict:
        """
        Возвращает секцию конфигурации
рации
        Args:
            section: Название секции
 section: Название секции
        Returns:
            Dict: Секция конфигурации        Returns:
        """ict: Секция конфигурации
        return copy.deepcopy(self.config.get(section, {}))
        return copy.deepcopy(self.config.get(section, {}))
    def update_section(self, section: str, config: Dict):
        """ str, config: Dict):
        Обновляет секцию конфигурации

        Args:
            section: Название секции
            config: Новая конфигурация section: Название секции
        """я
        if section not in self.config:        """
            self.config[section] = {}ction not in self.config:
}
        self.config[section].update(config)
f.config[section].update(config)
    def to_dict(self) -> Dict:
        """
        Возвращает всю конфигурацию в виде словаря        """
словаря
        Returns:
            Dict: Конфигурация
        """ Dict: Конфигурация
        return copy.deepcopy(self.config)
        return copy.deepcopy(self.config)
    def check_updates(self) -> bool:
        """ bool:
        Проверяет, были ли обновления в файлах конфигурации
айлах конфигурации
        Returns:
            bool: True, если были обновления, иначе False
        """ bool: True, если были обновления, иначе False
        # Проверяем, существует ли путь
        if not os.path.exists(self.config_path):        # Проверяем, существует ли путь
            return Falses.path.exists(self.config_path):

        # Проверяем время последнего изменения
        if os.path.isdir(self.config_path):менения
            # Для каталога проверяем все файлы
            for root, _, files in os.walk(self.config_path):га проверяем все файлы
                for file in files:            for root, _, files in os.walk(self.config_path):
                    if file.endswith((".json", ".yaml", ".yml")):
                        file_path = os.path.join(root, file)n", ".yaml", ".yml")):
                        mtime = os.path.getmtime(file_path)in(root, file)
                        if mtime > self.last_update:
                            return True self.last_update:
        else:
            # Для файла проверяем его время изменения
            mtime = os.path.getmtime(self.config_path)
            if mtime > self.last_update:h)
                return True:
   return True
        return False

    def get_last_update_time(self) -> str:
        """e(self) -> str:
        Возвращает время последнего обновления конфигурации        """
ремя последнего обновления конфигурации
        Returns:
            str: Время последнего обновления
        """ str: Время последнего обновления
        return datetime.fromtimestamp(self.last_update).strftime("%Y-%m-%d %H:%M:%S")
        return datetime.fromtimestamp(self.last_update).strftime("%Y-%m-%d %H:%M:%S")

# Create a module-level variable
_config = Noneй экземпляр конфигурации

def get_config():
    """
    Get or initialize the configuration singleton.
    
    Returns:    Возвращает глобальный экземпляр конфигурации
        Configuration instance
    """
    global _config Configuration: Экземпляр конфигурации
    if _config is None:
        _config = Configuration()    global _config
    return _config

def load_config(config_path: str = None) -> Configuration: _config = Configuration()
    """
    Загружает конфигурацию из указанного пути    return _config

    Args:
        config_path: Путь к конфигурацииdef load_config(config_path: str = None) -> Configuration:

    Returns:    Загружает конфигурацию из указанного пути
        Configuration: Экземпляр конфигурации
    """
    global _config config_path: Путь к конфигурации

    _config = Configuration(config_path)    Returns:
    return _configonfiguration: Экземпляр конфигурации

    global _config
def get_config_value(key: str, default: Any = None) -> Any:
    """
    Возвращает значение из конфигурацииurn _config

    Args:
        key: Ключ конфигурации (с точками для вложенных значений)Any = None) -> Any:
        default: Значение по умолчанию
    Возвращает значение из конфигурации
    Returns:
        Any: Значение из конфигурации или значение по умолчанию
    """ key: Ключ конфигурации (с точками для вложенных значений)
    config = get_config()
    return config.get(key, default)
ns:

def set_config_value(key: str, value: Any):
    """    config = get_config()
    Устанавливает значение в конфигурацииonfig.get(key, default)

    Args:
        key: Ключ конфигурации (с точками для вложенных значений) str, value: Any):
        value: Значение
    """    Устанавливает значение в конфигурации
    config = get_config()
    config.set(key, value)