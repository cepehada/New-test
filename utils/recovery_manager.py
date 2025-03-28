"""
Менеджер восстановления для автоматического перезапуска компонентов после сбоев
"""

import asyncio
import time
import traceback
from typing import Dict, Any, Callable, Coroutine, List, Optional

from project.utils.logging_utils import setup_logger

logger = setup_logger("recovery_manager")


class ComponentStatus:
    """Статус компонента системы"""
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    RECOVERING = "recovering"
    PAUSED = "paused"


class ComponentInfo:
    """Информация о компоненте системы"""
    def __init__(self, name: str, start_func: Callable, stop_func: Callable = None, 
                 check_func: Callable = None, dependencies: List[str] = None):
        self.name = name
        self.start_func = start_func
        self.stop_func = stop_func
        self.check_func = check_func  # Функция для проверки работоспособности
        self.dependencies = dependencies or []
        self.status = ComponentStatus.STOPPED
        self.last_start_time = 0
        self.last_check_time = 0
        self.error_count = 0
        self.restart_count = 0
        self.last_error = None


class RecoveryManager:
    """Менеджер восстановления системы после сбоев"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RecoveryManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.components: Dict[str, ComponentInfo] = {}
        self.watchdog_task = None
        self.max_restart_attempts = 5
        self.restart_cooldown = 60  # секунды
        self.check_interval = 30  # секунды
        self._initialized = True
        logger.info("Recovery Manager initialized")
    
    async def start(self):
        """Запускает менеджер восстановления"""
        if self.watchdog_task is not None:
            logger.warning("Recovery Manager is already running")
            return
            
        self.watchdog_task = asyncio.create_task(self._watchdog_loop())
        logger.info("Recovery Manager watchdog started")
    
    async def stop(self):
        """Останавливает менеджер восстановления"""
        if self.watchdog_task is None:
            logger.warning("Recovery Manager is not running")
            return
            
        self.watchdog_task.cancel()
        try:
            await self.watchdog_task
        except asyncio.CancelledError:
            pass
        
        self.watchdog_task = None
        logger.info("Recovery Manager watchdog stopped")
    
    def register_component(self, name: str, start_func: Callable, stop_func: Callable = None, 
                           check_func: Callable = None, dependencies: List[str] = None):
        """Регистрирует компонент для мониторинга"""
        if name in self.components:
            logger.warning(f"Component {name} is already registered")
            return
            
        self.components[name] = ComponentInfo(
            name=name,
            start_func=start_func,
            stop_func=stop_func,
            check_func=check_func,
            dependencies=dependencies
        )
        logger.info(f"Component {name} registered for monitoring")
    
    async def start_component(self, name: str) -> bool:
        """Запускает компонент"""
        if name not in self.components:
            logger.error(f"Unknown component: {name}")
            return False
        
        component = self.components[name]
        
        # Проверяем зависимости
        for dep_name in component.dependencies:
            if dep_name not in self.components:
                logger.error(f"Missing dependency {dep_name} for component {name}")
                return False
            
            dep_component = self.components[dep_name]
            if dep_component.status != ComponentStatus.RUNNING:
                logger.warning(f"Dependency {dep_name} is not running for component {name}")
                # Пытаемся запустить зависимость
                success = await self.start_component(dep_name)
                if not success:
                    logger.error(f"Failed to start dependency {dep_name} for {name}")
                    return False
        
        try:
            # Запускаем компонент
            logger.info(f"Starting component {name}...")
            await component.start_func()
            
            component.status = ComponentStatus.RUNNING
            component.last_start_time = time.time()
            component.error_count = 0
            logger.info(f"Component {name} started successfully")
            return True
            
        except Exception as e:
            component.status = ComponentStatus.ERROR
            component.last_error = str(e)
            component.error_count += 1
            logger.error(f"Error starting component {name}: {str(e)}")
            logger.debug(traceback.format_exc())
            return False
    
    async def stop_component(self, name: str) -> bool:
        """Останавливает компонент"""
        if name not in self.components:
            logger.error(f"Unknown component: {name}")
            return False
        
        component = self.components[name]
        
        # Проверяем, есть ли другие компоненты, которые зависят от этого
        dependents = []
        for comp_name, comp in self.components.items():
            if name in comp.dependencies and comp.status == ComponentStatus.RUNNING:
                dependents.append(comp_name)
        
        if dependents:
            logger.warning(f"Cannot stop {name} because components depend on it: {', '.join(dependents)}")
            return False
        
        if component.stop_func is None:
            logger.warning(f"No stop function defined for component {name}")
            component.status = ComponentStatus.STOPPED
            return True
        
        try:
            # Останавливаем компонент
            logger.info(f"Stopping component {name}...")
            await component.stop_func()
            
            component.status = ComponentStatus.STOPPED
            logger.info(f"Component {name} stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping component {name}: {str(e)}")
            logger.debug(traceback.format_exc())
            return False
    
    async def restart_component(self, name: str) -> bool:
        """Перезапускает компонент"""
        if name not in self.components:
            logger.error(f"Unknown component: {name}")
            return False
        
        component = self.components[name]
        
        # Проверяем, не превышено ли количество попыток перезапуска
        if component.restart_count >= self.max_restart_attempts:
            logger.warning(f"Maximum restart attempts ({self.max_restart_attempts}) reached for {name}")
            component.status = ComponentStatus.ERROR
            return False
        
        # Проверяем время остывания
        if time.time() - component.last_start_time < self.restart_cooldown:
            cooldown_remaining = self.restart_cooldown - (time.time() - component.last_start_time)
            logger.warning(f"Cooldown period not expired for {name}, waiting {cooldown_remaining:.1f}s")
            await asyncio.sleep(cooldown_remaining)
        
        # Останавливаем компонент
        await self.stop_component(name)
        
        # Запускаем компонент
        component.status = ComponentStatus.RECOVERING
        component.restart_count += 1
        
        success = await self.start_component(name)
        
        if success:
            logger.info(f"Component {name} restarted successfully (attempt {component.restart_count}/{self.max_restart_attempts})")
            component.restart_count = 0  # Сбрасываем счетчик после успешного перезапуска
        else:
            logger.error(f"Failed to restart component {name} (attempt {component.restart_count}/{self.max_restart_attempts})")
        
        return success
    
    async def check_component(self, name: str) -> bool:
        """Проверяет работоспособность компонента"""
        if name not in self.components:
            logger.error(f"Unknown component: {name}")
            return False
        
        component = self.components[name]
        
        if component.status != ComponentStatus.RUNNING:
            # Компонент не запущен, проверка не требуется
            return False
        
        if component.check_func is None:
            # Функция проверки не определена, считаем компонент работоспособным
            return True
        
        try:
            # Проверяем компонент
            component.last_check_time = time.time()
            is_healthy = await component.check_func()
            
            if not is_healthy:
                logger.warning(f"Component {name} health check failed")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking component {name}: {str(e)}")
            logger.debug(traceback.format_exc())
            return False
    
    async def _watchdog_loop(self):
        """Основной цикл мониторинга компонентов"""
        try:
            while True:
                # Проверяем все компоненты
                for name, component in self.components.items():
                    if component.status == ComponentStatus.RUNNING:
                        # Проверяем работоспособность
                        is_healthy = await self.check_component(name)
                        
                        if not is_healthy:
                            logger.warning(f"Component {name} is unhealthy, attempting restart")
                            await self.restart_component(name)
                    
                    elif component.status == ComponentStatus.ERROR:
                        # Если компонент в ошибке, пытаемся его перезапустить
                        if time.time() - component.last_start_time > self.restart_cooldown:
                            logger.info(f"Attempting to recover component {name} from error state")
                            await self.restart_component(name)
                
                # Ждем до следующей проверки
                await asyncio.sleep(self.check_interval)
                
        except asyncio.CancelledError:
            logger.info("Watchdog loop cancelled")
        except Exception as e:
            logger.error(f"Error in watchdog loop: {str(e)}")
            logger.debug(traceback.format_exc())


# Глобальный экземпляр менеджера восстановления
_recovery_manager = None

def get_recovery_manager() -> RecoveryManager:
    """Возвращает глобальный экземпляр менеджера восстановления"""
    global _recovery_manager
    if _recovery_manager is None:
        _recovery_manager = RecoveryManager()
    return _recovery_manager
