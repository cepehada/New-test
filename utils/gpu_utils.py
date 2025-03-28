import time
from typing import Callable, Dict, Tuple, Union

import numpy as np
import pandas as pd
from project.utils.logging_utils import setup_logger

logger = setup_logger("gpu_utils")

# Проверяем доступность CUDA
try:
    import cudf
    import cupy as cp
    from numba import cuda

    CUDA_AVAILABLE = cuda.is_available()
    if CUDA_AVAILABLE:
        logger.info("CUDA is available. GPU acceleration enabled.")

        # Получаем информацию о доступных устройствах
        devices = []
        for i in range(cuda.gpus.lst):
            device = cuda.get_current_device()
            devices.append(
                {
                    "id": i,
                    "name": device.name.decode("utf-8"),
                    "total_memory": device.total_memory,
                    "compute_capability": f"{
                        device.compute_capability[0]}.{
                        device.compute_capability[1]}",
                }
            )
        logger.info(f"Available CUDA devices: {devices}")
    else:
        logger.warning("CUDA is not available. Using CPU fallback.")
except ImportError:
    logger.warning("CUDA libraries not found. Using CPU fallback.")
    CUDA_AVAILABLE = False

    # Создаем заглушки для модулей cupy и cudf
    class DummyModule:
        def __getattr__(self, name):
            return getattr(np, name)

    cp = DummyModule()

    class DummyDataFrame:
        def __init__(self, *args, **kwargs):
            self.df = pd.DataFrame(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.df, name)

    cudf = type("cudf", (), {"DataFrame": DummyDataFrame})


def get_gpu_info() -> Dict:
    """
    Получает информацию о доступных GPU

    Returns:
        Dict: Информация о GPU
    """
    if not CUDA_AVAILABLE:
        return {"available": False, "devices": []}

    try:
        # Получаем информацию о GPU
        devices = []
        for i in range(cuda.gpus.lst):
            device = cuda.get_current_device()
            memory_info = cuda.current_context().get_memory_info()

            devices.append(
                {
                    "id": i,
                    "name": device.name.decode("utf-8"),
                    "total_memory": device.total_memory,
                    "free_memory": memory_info[0],
                    "used_memory": memory_info[1],
                    "compute_capability": f"{
                        device.compute_capability[0]}.{
                        device.compute_capability[1]}",
                }
            )

        return {
            "available": True,
            "devices": devices,
            "current_device": cuda.get_current_device().id,
        }
    except Exception as e:
        logger.error(f"Error getting GPU info: {str(e)}")
        return {"available": False, "devices": [], "error": str(e)}


def set_gpu_device(device_id: int) -> bool:
    """
    Устанавливает активное GPU устройство

    Args:
        device_id: ID устройства

    Returns:
        bool: True, если устройство успешно установлено, иначе False
    """
    if not CUDA_AVAILABLE:
        logger.warning("CUDA is not available.")
        return False

    try:
        # Проверяем, существует ли устройство
        if device_id >= cuda.gpus.lst:
            logger.error(f"Device ID {device_id} is out of range.")
            return False

        # Устанавливаем текущее устройство
        cuda.select_device(device_id)
        logger.info(f"Selected GPU device: {device_id}")
        return True
    except Exception as e:
        logger.error(f"Error setting GPU device: {str(e)}")
        return False


def to_gpu(
    data: Union[np.ndarray, pd.DataFrame],
) -> Union[cp.ndarray, "cudf.DataFrame"]:
    """
    Переносит данные на GPU

    Args:
        data: Данные для переноса

    Returns:
        Union[cp.ndarray, cudf.DataFrame]: Данные на GPU
    """
    if not CUDA_AVAILABLE:
        logger.warning("CUDA is not available. Returning original data.")
        return data

    try:
        if isinstance(data, np.ndarray):
            # Переносим NumPy массив на GPU
            return cp.array(data)
        elif isinstance(data, pd.DataFrame):
            # Переносим pandas DataFrame на GPU
            return cudf.DataFrame.from_pandas(data)
        else:
            logger.warning(
                f"Unsupported data type: {type(data)}. Returning original data."
            )
            return data
    except Exception as e:
        logger.error(f"Error transferring data to GPU: {str(e)}")
        return data


def to_cpu(
    data: Union[cp.ndarray, "cudf.DataFrame"],
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Переносит данные с GPU на CPU

    Args:
        data: Данные для переноса

    Returns:
        Union[np.ndarray, pd.DataFrame]: Данные на CPU
    """
    if not CUDA_AVAILABLE:
        return data

    try:
        if isinstance(data, cp.ndarray):
            # Переносим CuPy массив на CPU
            return data.get()
        elif hasattr(data, "to_pandas"):
            # Переносим cuDF DataFrame на CPU
            return data.to_pandas()
        else:
            logger.warning(
                f"Unsupported data type: {type(data)}. Returning original data."
            )
            return data
    except Exception as e:
        logger.error(f"Error transferring data to CPU: {str(e)}")
        return data


def benchmark_gpu(
    data_size: Tuple[int, int] = (10000, 100), iterations: int = 10
) -> Dict:
    """
    Сравнивает производительность CPU и GPU

    Args:
        data_size: Размер тестовых данных (строки, столбцы)
        iterations: Количество итераций для каждого теста

    Returns:
        Dict: Результаты тестирования
    """
    results = {
        "cpu_times": [],
        "gpu_times": [],
        "speedup": None,
        "data_size": data_size,
        "cuda_available": CUDA_AVAILABLE,
    }

    if not CUDA_AVAILABLE:
        logger.warning("CUDA is not available. Skipping GPU benchmark.")
        return results

    try:
        rows, cols = data_size

        # Создаем случайные данные
        cpu_data = np.random.random((rows, cols))

        # Тест на CPU
        for i in range(iterations):
            start_time = time.time()

            # Операции для тестирования
            result = np.dot(cpu_data, cpu_data.T)
            u, s, v = np.linalg.svd(result[:1000, :1000])

            cpu_time = time.time() - start_time
            results["cpu_times"].append(cpu_time)

        # Тест на GPU
        gpu_data = cp.array(cpu_data)

        for i in range(iterations):
            start_time = time.time()

            # Операции для тестирования
            result = cp.dot(gpu_data, gpu_data.T)
            u, s, v = cp.linalg.svd(result[:1000, :1000])

            # Синхронизация GPU
            cp.cuda.Stream.null.synchronize()

            gpu_time = time.time() - start_time
            results["gpu_times"].append(gpu_time)

        # Рассчитываем ускорение
        avg_cpu_time = sum(results["cpu_times"]) / len(results["cpu_times"])
        avg_gpu_time = sum(results["gpu_times"]) / len(results["gpu_times"])

        if avg_cpu_time > 0:
            results["speedup"] = avg_cpu_time / avg_gpu_time

        logger.info(
            f"Benchmark results: CPU: {
                avg_cpu_time:.4f}s, GPU: {
                avg_gpu_time:.4f}s, Speedup: {
                results['speedup']:.2f}x"
        )

        return results
    except Exception as e:
        logger.error(f"Error during benchmark: {str(e)}")
        return results


@cuda.jit
def _moving_average_kernel(data, window_size, result):
    """
    CUDA-ядро для расчета скользящего среднего

    Args:
        data: Исходные данные
        window_size: Размер окна
        result: Результат
    """
    i = cuda.grid(1)

    if i < data.shape[0]:
        # Инициализируем сумму и счетчик
        total = 0.0
        count = 0

        # Рассчитываем скользящее среднее
        for j in range(max(0, i - window_size + 1), i + 1):
            total += data[j]
            count += 1

        # Записываем результат
        result[i] = total / count


def moving_average_gpu(data: np.ndarray, window: int) -> np.ndarray:
    """
    Рассчитывает скользящее среднее с использованием GPU

    Args:
        data: Исходные данные
        window: Размер окна

    Returns:
        np.ndarray: Скользящее среднее
    """
    if not CUDA_AVAILABLE:
        logger.warning("CUDA is not available. Using CPU implementation.")
        return np.array(pd.Series(data).rolling(window=window, min_periods=1).mean())

    try:
        # Переносим данные на GPU
        gpu_data = cp.array(data, dtype=cp.float64)
        gpu_result = cp.zeros_like(gpu_data)

        # Настраиваем размеры блоков и сетки
        threads_per_block = 256
        blocks_per_grid = (
            gpu_data.shape[0] + threads_per_block - 1
        ) // threads_per_block

        # Запускаем ядро
        _moving_average_kernel[blocks_per_grid, threads_per_block](
            gpu_data, window, gpu_result
        )

        # Возвращаем результат на CPU
        return gpu_result.get()
    except Exception as e:
        logger.error(f"Error calculating moving average on GPU: {str(e)}")
        return np.array(pd.Series(data).rolling(window=window, min_periods=1).mean())


@cuda.jit
def _crossover_kernel(data1, data2, result):
    """
    CUDA-ядро для определения пересечений

    Args:
        data1: Первый набор данных
        data2: Второй набор данных
        result: Результат (1 - пересечение снизу вверх, -1 - пересечение сверху вниз, 0 - нет пересечения)
    """
    i = cuda.grid(1)

    if i > 0 and i < data1.shape[0]:
        # Проверяем пересечение снизу вверх
        if data1[i - 1] < data2[i - 1] and data1[i] >= data2[i]:
            result[i] = 1
        # Проверяем пересечение сверху вниз
        elif data1[i - 1] > data2[i - 1] and data1[i] <= data2[i]:
            result[i] = -1
        else:
            result[i] = 0


def crossover_gpu(data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
    """
    Определяет пересечения двух наборов данных с использованием GPU

    Args:
        data1: Первый набор данных
        data2: Второй набор данных

    Returns:
        np.ndarray: Массив с результатами пересечений
    """
    if not CUDA_AVAILABLE:
        logger.warning("CUDA is not available. Using CPU implementation.")
        result = np.zeros(data1.shape[0])
        for i in range(1, data1.shape[0]):
            if data1[i - 1] < data2[i - 1] and data1[i] >= data2[i]:
                result[i] = 1
            elif data1[i - 1] > data2[i - 1] and data1[i] <= data2[i]:
                result[i] = -1
        return result

    try:
        # Переносим данные на GPU
        gpu_data1 = cp.array(data1, dtype=cp.float64)
        gpu_data2 = cp.array(data2, dtype=cp.float64)
        gpu_result = cp.zeros(gpu_data1.shape[0], dtype=cp.int32)

        # Настраиваем размеры блоков и сетки
        threads_per_block = 256
        blocks_per_grid = (
            gpu_data1.shape[0] + threads_per_block - 1
        ) // threads_per_block

        # Запускаем ядро
        _crossover_kernel[blocks_per_grid, threads_per_block](
            gpu_data1, gpu_data2, gpu_result
        )

        # Возвращаем результат на CPU
        return gpu_result.get()
    except Exception as e:
        logger.error(f"Error calculating crossover on GPU: {str(e)}")
        result = np.zeros(data1.shape[0])
        for i in range(1, data1.shape[0]):
            if data1[i - 1] < data2[i - 1] and data1[i] >= data2[i]:
                result[i] = 1
            elif data1[i - 1] > data2[i - 1] and data1[i] <= data2[i]:
                result[i] = -1
        return result


@cuda.jit
def _rsi_kernel(data, period, result):
    """
    CUDA-ядро для расчета RSI

    Args:
        data: Исходные данные цен закрытия
        period: Период RSI
        result: Результат
    """
    i = cuda.grid(1)

    if i >= period and i < data.shape[0]:
        up_sum = 0.0
        down_sum = 0.0

        # Рассчитываем сумму положительных и отрицательных изменений
        for j in range(i - period + 1, i + 1):
            change = data[j] - data[j - 1]
            if change > 0:
                up_sum += change
            else:
                down_sum -= change

        # Избегаем деления на ноль
        if down_sum == 0:
            result[i] = 100.0
        else:
            rs = up_sum / down_sum
            result[i] = 100.0 - (100.0 / (1.0 + rs))


def rsi_gpu(data: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Рассчитывает индикатор RSI с использованием GPU

    Args:
        data: Массив цен закрытия
        period: Период RSI

    Returns:
        np.ndarray: Значения RSI
    """
    if not CUDA_AVAILABLE:
        logger.warning("CUDA is not available. Using CPU implementation.")
        delta = np.diff(data)
        gains = np.copy(delta)
        losses = np.copy(delta)

        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = -losses

        avg_gain = np.concatenate(([np.nan] * (period - 1), [np.mean(gains[:period])]))
        avg_loss = np.concatenate(([np.nan] * (period - 1), [np.mean(losses[:period])]))

        for i in range(period, len(delta)):
            avg_gain = np.append(
                avg_gain, (avg_gain[-1] * (period - 1) + gains[i]) / period
            )
            avg_loss = np.append(
                avg_loss, (avg_loss[-1] * (period - 1) + losses[i]) / period
            )

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return np.concatenate(([np.nan], rsi))

    try:
        # Переносим данные на GPU
        gpu_data = cp.array(data, dtype=cp.float64)
        gpu_result = cp.zeros_like(gpu_data)

        # Заполняем период NaN значениями
        gpu_result[:period] = cp.nan

        # Настраиваем размеры блоков и сетки
        threads_per_block = 256
        blocks_per_grid = (
            gpu_data.shape[0] + threads_per_block - 1
        ) // threads_per_block

        # Запускаем ядро
        _rsi_kernel[blocks_per_grid, threads_per_block](gpu_data, period, gpu_result)

        # Возвращаем результат на CPU
        return gpu_result.get()
    except Exception as e:
        logger.error(f"Error calculating RSI on GPU: {str(e)}")
        delta = np.diff(data)
        gains = np.copy(delta)
        losses = np.copy(delta)

        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = -losses

        avg_gain = np.concatenate(([np.nan] * (period - 1), [np.mean(gains[:period])]))
        avg_loss = np.concatenate(([np.nan] * (period - 1), [np.mean(losses[:period])]))

        for i in range(period, len(delta)):
            avg_gain = np.append(
                avg_gain, (avg_gain[-1] * (period - 1) + gains[i]) / period
            )
            avg_loss = np.append(
                avg_loss, (avg_loss[-1] * (period - 1) + losses[i]) / period
            )

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return np.concatenate(([np.nan], rsi))


def parallel_apply_gpu(df: pd.DataFrame, func: Callable, column: str) -> pd.DataFrame:
    """
    Применяет функцию к колонке DataFrame с использованием GPU

    Args:
        df: DataFrame
        func: Функция для применения
        column: Имя колонки

    Returns:
        pd.DataFrame: Обработанный DataFrame
    """
    if not CUDA_AVAILABLE:
        logger.warning("CUDA is not available. Using CPU implementation.")
        return df.assign(result=df[column].apply(func))

    try:
        # Переносим DataFrame на GPU
        gpu_df = cudf.DataFrame.from_pandas(df)

        # Применяем функцию
        if hasattr(gpu_df[column], "apply"):
            gpu_df["result"] = gpu_df[column].apply(func)
        else:
            # Если применить функцию напрямую нельзя, переносим данные на CPU
            cpu_result = df[column].apply(func)
            gpu_df["result"] = cudf.Series(cpu_result)

        # Возвращаем результат на CPU
        return gpu_df.to_pandas()
    except Exception as e:
        logger.error(f"Error applying function on GPU: {str(e)}")
        return df.assign(result=df[column].apply(func))


def cleanup_gpu_memory():
    """Очищает память GPU"""
    if not CUDA_AVAILABLE:
        return

    try:
        # Очищаем память
        cuda.current_context().pop()
        cuda.close()

        logger.info("GPU memory cleaned up")
    except Exception as e:
        logger.error(f"Error cleaning up GPU memory: {str(e)}")


def is_gpu_available() -> bool:
    """
    Проверяет доступность GPU

    Returns:
        bool: True, если GPU доступен, иначе False
    """
    return CUDA_AVAILABLE


def check_cupy_availability():
    """Проверяет доступность cupy для работы с GPU"""
    # Убираем лишнее 'elif' после return
    try:
        return True
    except ImportError:
        return False


def get_gpu_accelerator():
    """Возвращает доступный ускоритель для вычислений на GPU"""
    # Убираем лишнее 'elif' после return
    try:
        import cupy as cp

        return cp
    except ImportError:
        try:
            import numba.cuda as cuda

            if cuda.is_available():
                return cuda
        except ImportError:
            pass
    return None


# Инициализация при импорте модуля
logger.info(f"GPU acceleration: {'Available' if CUDA_AVAILABLE else 'Not available'}")
