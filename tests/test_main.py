import sys
import os
import time
import traceback
import numpy as np

from nnmodel.CTModel import CTModel

# Конфиг

TEST_INPUT = "path"   # Папка с DICOM / путь к видео / .zip / .npy / любой поддерживаемый формат видео см. README.md

def _run(name: str, fn):
    """Запускает тест и печатает результат."""
    try:
        fn()
        print(f"  [PASS] {name}")
        return True
    except AssertionError as e:
        print(f"  [FAIL] {name}")
        print(f"         AssertionError: {e}")
        return False
    except Exception:
        print(f"  [FAIL] {name}")
        traceback.print_exc()
        return False


print("[setup] Загрузка моделей и обработка тестового файла...")
t0 = time.time()

model = CTModel()
model.load()
model.predict(TEST_INPUT)

elapsed = time.time() - t0
print(f"[setup] Готово за {elapsed:.1f} сек\n")


# Тесты атрибутов после predict

def test_video_loaded():
    """np_video должен быть загружен и иметь правильную форму."""
    assert model.np_video is not None, "np_video is None"
    assert model.np_video.ndim == 3, f"Ожидалось 3D, получено {model.np_video.ndim}D"
    assert model.np_video.shape[0] == 53, \
        f"Ожидалось 53 кадра, получено {model.np_video.shape[0]}"
    assert model.np_video.shape[1] == 224 and model.np_video.shape[2] == 224, \
        f"Ожидался размер кадра 224×224, получено {model.np_video.shape[1:3]}"


def test_mask_shape():
    """np_mask должна иметь ту же форму, что и np_video."""
    assert model.np_mask is not None, "np_mask is None"
    assert model.np_mask.shape == model.np_video.shape, (
        f"Форма маски {model.np_mask.shape} не совпадает с видео {model.np_video.shape}"
    )


def test_roi_structure():
    """detected_roi — список длиной N (по числу кадров), каждый элемент — список ROI."""
    assert model.detected_roi is not None, "detected_roi is None"
    n_frames = model.np_video.shape[0]
    assert len(model.detected_roi) == n_frames, (
        f"Ожидалось {n_frames} записей ROI, получено {len(model.detected_roi)}"
    )

    for frame_idx, frame_rois in enumerate(model.detected_roi):
        assert isinstance(frame_rois, list), \
            f"Кадр {frame_idx}: ROI должны быть списком, получено {type(frame_rois)}"
        for roi in frame_rois:
            assert len(roi) == 4, \
                f"Кадр {frame_idx}: ROI должен содержать 4 точки, получено {len(roi)}"
            for point in roi:
                assert len(point) == 2, \
                    f"Кадр {frame_idx}: точка ROI должна содержать 2 координаты"


def test_lesion_masks_exist():
    """lesion_masks — непустой список масок."""
    assert model.lesion_masks, "lesion_masks пуст"
    for idx, lm in enumerate(model.lesion_masks):
        assert isinstance(lm, np.ndarray), \
            f"Маска образования {idx + 1} не является np.ndarray"
        assert lm.shape == model.np_video.shape, (
            f"Маска образования {idx + 1}: форма {lm.shape} "
            f"не совпадает с видео {model.np_video.shape}"
        )


def test_lesion_masks_coverage():
    """Объединение всех масок образований должно совпадать с объединённой маской."""
    combined = np.zeros_like(model.np_mask)
    for lm in model.lesion_masks:
        combined = np.maximum(combined, lm)
    # Каждый ненулевой пиксель объединённой маски должен быть хотя бы в одной маске образования
    original_nonzero = (model.np_mask > 0)
    covered_nonzero  = (combined > 0)
    uncovered = original_nonzero & ~covered_nonzero
    uncovered_count = int(uncovered.sum())
    assert uncovered_count == 0, (
        f"Объединение масок образований не покрывает {uncovered_count} пикселей "
        "из оригинальной маски"
    )


def test_results_list():
    """results должен содержать по одному результату на каждую маску образования."""
    assert len(model.results) == len(model.lesion_masks), (
        f"Количество результатов ({len(model.results)}) "
        f"не совпадает с числом масок ({len(model.lesion_masks)})"
    )


def test_classification_output_types():
    """Каждый результат классификации должен содержать proba, label_idx, label_name."""
    assert model.results, "Список results пуст"

    valid_labels = {"Benign", "Indeterminate", "Malignant"}

    for i, (proba, label_idx, label_name) in enumerate(model.results, start=1):
        assert proba is not None, f"Образование {i}: proba is None"
        assert label_idx in (0, 1, 2), \
            f"Образование {i}: неожиданный label_idx={label_idx}"
        assert label_name in valid_labels, \
            f"Образование {i}: неожиданный label_name='{label_name}'"


def test_proba_is_probability_distribution():
    """Вероятности каждого образования должны суммироваться к ~1.0."""
    assert model.results, "Список results пуст"

    for i, (proba, _, _) in enumerate(model.results, start=1):
        proba_np = proba.detach().numpy() if hasattr(proba, 'detach') else np.array(proba)
        total = float(proba_np.sum())
        assert abs(total - 1.0) < 1e-3, (
            f"Образование {i}: сумма вероятностей {total:.6f} ≠ 1.0"
        )


# Вывод данных

def print_summary():
    print("\n" + "─" * 60)
    print("  Детали predict:")
    print(f"  Форма видео:             {model.np_video.shape}")
    print(f"  Форма маски:             {model.np_mask.shape}")
    print(f"  Обнаружено образований:  {len(model.lesion_masks)}")
    for i, (proba, label_idx, label_name) in enumerate(model.results, start=1):
        proba_np = proba.detach().numpy() if hasattr(proba, 'detach') else np.array(proba)
        print(f"  Образование {i}:  {label_name}  (вероятности: {np.round(proba_np, 4)})")
    print(f"  Время обработки:         {elapsed:.1f} сек")
    print("─" * 60)

# Запуск

TESTS = [
    ("Видео загружено с правильной формой",          test_video_loaded),
    ("Форма маски совпадает с видео",                test_mask_shape),
    ("Структура ROI корректна",                      test_roi_structure),
    ("Маски образований существуют и корректны",     test_lesion_masks_exist),
    ("Маски образований покрывают всю сегментацию",  test_lesion_masks_coverage),
    ("results содержит все образования",             test_results_list),
    ("Типы данных результатов классификации",        test_classification_output_types),
    ("Вероятности суммируются к 1.0",                test_proba_is_probability_distribution),
]

if __name__ == "__main__":
    print("=" * 60)
    print("  CTModel — интеграционные тесты")
    print("=" * 60)

    passed = 0
    for name, fn in TESTS:
        if _run(name, fn):
            passed += 1

    print_summary()

    total = len(TESTS)
    print(f"\n  Итог: {passed}/{total} тестов прошло")
    print("=" * 60)

    sys.exit(0 if passed == total else 1)

