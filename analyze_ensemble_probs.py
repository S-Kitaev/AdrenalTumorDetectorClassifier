import numpy as np
import os
import sys

# Установка кодировки для Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def analyze_ensemble_probabilities(file_path):
    """
    Анализирует файл с вероятностями ансамбля моделей
    
    Args:
        file_path: путь к файлу .npy
    """
    print("=" * 80)
    print(f"Анализ файла: {file_path}")
    print("=" * 80)
    
    # Проверка существования файла
    if not os.path.exists(file_path):
        print(f"ОШИБКА: Файл {file_path} не найден!")
        return
    
    # Загрузка данных
    try:
        data = np.load(file_path)
        print(f"\n✓ Файл успешно загружен")
    except Exception as e:
        print(f"ОШИБКА при загрузке файла: {e}")
        return
    
    # Основная информация о данных
    print("\n" + "-" * 80)
    print("ОСНОВНАЯ ИНФОРМАЦИЯ")
    print("-" * 80)
    print(f"Тип данных: {type(data)}")
    print(f"Форма (shape): {data.shape}")
    print(f"Размерность: {data.ndim}D")
    print(f"Тип элементов: {data.dtype}")
    print(f"Общее количество элементов: {data.size}")
    
    # Анализ структуры данных
    print("\n" + "-" * 80)
    print("СТРУКТУРА ДАННЫХ")
    print("-" * 80)
    
    if data.ndim == 1:
        print("Одномерный массив (вероятности для одного образца)")
        print(f"Количество классов: {len(data)}")
    elif data.ndim == 2:
        print("Двумерный массив")
        print(f"Количество образцов: {data.shape[0]}")
        print(f"Количество классов: {data.shape[1]}")
    elif data.ndim == 3:
        print("Трехмерный массив")
        print(f"Размерности: {data.shape}")
        print("Возможная интерпретация:")
        print(f"  - Образцы: {data.shape[0]}")
        print(f"  - Модели в ансамбле: {data.shape[1]}")
        print(f"  - Классы: {data.shape[2]}")
    else:
        print(f"Многомерный массив с {data.ndim} измерениями")
    
    # Статистика по вероятностям
    print("\n" + "-" * 80)
    print("СТАТИСТИКА ПО ВЕРОЯТНОСТЯМ")
    print("-" * 80)
    
    # Если это 3D массив (образцы x модели x классы)
    if data.ndim == 3:
        print("\nПо моделям в ансамбле:")
        for model_idx in range(data.shape[1]):
            model_probs = data[:, model_idx, :]
            print(f"\n  Модель {model_idx + 1}:")
            print(f"    Средние вероятности по классам: {np.mean(model_probs, axis=0)}")
            print(f"    Стандартные отклонения: {np.std(model_probs, axis=0)}")
            print(f"    Минимумы: {np.min(model_probs, axis=0)}")
            print(f"    Максимумы: {np.max(model_probs, axis=0)}")
        
        # Усредненные вероятности ансамбля
        ensemble_avg = np.mean(data, axis=1)  # Усреднение по моделям
        print("\n  Усредненные вероятности ансамбля:")
        print(f"    Средние по классам: {np.mean(ensemble_avg, axis=0)}")
        print(f"    Стандартные отклонения: {np.std(ensemble_avg, axis=0)}")
        
    elif data.ndim == 2:
        # Если это 2D массив (образцы x классы)
        print(f"\nСредние вероятности по классам: {np.mean(data, axis=0)}")
        print(f"Стандартные отклонения: {np.std(data, axis=0)}")
        print(f"Минимумы: {np.min(data, axis=0)}")
        print(f"Максимумы: {np.max(data, axis=0)}")
        
        # Проверка сумм вероятностей (должны быть близки к 1)
        row_sums = np.sum(data, axis=1)
        print(f"\nСуммы вероятностей по образцам:")
        print(f"  Средняя сумма: {np.mean(row_sums):.6f}")
        print(f"  Стандартное отклонение: {np.std(row_sums):.6f}")
        print(f"  Минимум: {np.min(row_sums):.6f}")
        print(f"  Максимум: {np.max(row_sums):.6f}")
        
        if np.allclose(row_sums, 1.0, atol=1e-5):
            print("  ✓ Суммы вероятностей близки к 1.0 (нормализованы)")
        else:
            print("  ⚠ Суммы вероятностей не равны 1.0!")
    
    elif data.ndim == 1:
        # Если это 1D массив (вероятности для одного образца)
        print(f"\nВероятности: {data}")
        print(f"Сумма: {np.sum(data):.6f}")
        print(f"Максимальная вероятность (предсказанный класс): {np.max(data):.6f}")
        print(f"Индекс предсказанного класса: {np.argmax(data)}")
        
        if np.allclose(np.sum(data), 1.0, atol=1e-5):
            print("✓ Вероятности нормализованы (сумма = 1.0)")
        else:
            print("⚠ Вероятности не нормализованы!")
    
    # Проверка диапазона значений
    print("\n" + "-" * 80)
    print("ДИАПАЗОН ЗНАЧЕНИЙ")
    print("-" * 80)
    print(f"Минимальное значение: {np.min(data):.6f}")
    print(f"Максимальное значение: {np.max(data):.6f}")
    print(f"Среднее значение: {np.mean(data):.6f}")
    print(f"Медиана: {np.median(data):.6f}")
    
    # Проверка на наличие NaN и Inf
    print("\n" + "-" * 80)
    print("ПРОВЕРКА НА ОШИБКИ")
    print("-" * 80)
    nan_count = np.isnan(data).sum()
    inf_count = np.isinf(data).sum()
    print(f"Количество NaN: {nan_count}")
    print(f"Количество Inf: {inf_count}")
    
    if nan_count > 0:
        print("⚠ Обнаружены NaN значения!")
    if inf_count > 0:
        print("⚠ Обнаружены Inf значения!")
    if nan_count == 0 and inf_count == 0:
        print("✓ Ошибок не обнаружено")
    
    # Примеры данных
    print("\n" + "-" * 80)
    print("ПРИМЕРЫ ДАННЫХ")
    print("-" * 80)
    
    if data.ndim == 3:
        print("\nПервые 3 образца, первая модель:")
        print(data[:min(3, data.shape[0]), 0, :])
        if data.shape[1] > 1:
            print("\nПервые 3 образца, усредненные по ансамблю:")
            print(np.mean(data[:min(3, data.shape[0]), :, :], axis=1))
    elif data.ndim == 2:
        print("\nПервые 5 образцов:")
        print(data[:min(5, data.shape[0]), :])
    elif data.ndim == 1:
        print("\nВсе данные:")
        print(data)
    
    print("\n" + "=" * 80)
    print("Анализ завершен")
    print("=" * 80)


if __name__ == "__main__":
    # Путь к файлу
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Пробуем найти файл в текущей директории и поддиректориях
        possible_paths = [
            "ensemble_pobabilities.npy",
            "ensemble_probabilities.npy",
            "./ensemble_pobabilities.npy",
            "./ensemble_probabilities.npy",
            "../ensemble_pobabilities.npy",
            "../ensemble_probabilities.npy",
        ]
        
        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break
        
        if file_path is None:
            print("Usage: python analyze_ensemble_probs.py <path_to_file.npy>")
            print("\nOr place ensemble_pobabilities.npy file in current directory")
            sys.exit(1)
    
    analyze_ensemble_probabilities(file_path)
