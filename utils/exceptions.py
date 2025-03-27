"""
Модуль для определения кастомных исключений проекта.
"""
class RequestError(Exception):
    """Ошибка запроса к API."""
    pass

class NetworkError(Exception):
    """Ошибка сети."""
    pass

class ConfigurationError(Exception):
    """Ошибка конфигурации."""
    pass

class DatabaseError(Exception):
    """Ошибка базы данных."""
    pass

class AuthenticationError(Exception):
    """Ошибка аутентификации."""
    pass

class PermissionError(Exception):
    """Ошибка прав доступа."""
    pass

class ValidationError(Exception):
    """Ошибка валидации."""
    pass

class BusinessLogicError(Exception):
    """Ошибка бизнес-логики."""
    pass
