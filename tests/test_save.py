import os
import sys
import shutil
import traceback

import cv2
from nnmodel.CTModel import CTModel


# Конфиг

TEST_INPUT  = "path"   # Путь к тестовому видео
OUTPUT_DIR  = "./test_saving"                       # Директория для сохранения
VIDEO_NAME  = "test_run"                            # Базовое имя файлов


def _run(name: str, fn):
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


def _expected_files(n_lesions: int) -> list[str]:
    """
    Возвращает список ожидаемых имён файлов при n образованиях.
    Всегда создаётся: _detection.mp4 и _mask.mp4.
    При n > 1 дополнительно: _mask_1.mp4, _mask_2.mp4, ...
    """
    names = [
        f"{VIDEO_NAME}_detection.mp4",
        f"{VIDEO_NAME}_mask.mp4",
    ]
    if n_lesions > 1:
        for i in range(1, n_lesions + 1):
            names.append(f"{VIDEO_NAME}_mask_{i}.mp4")
    return names


def _assert_valid_mp4(filepath: str) -> None:
    """Проверяет, что файл существует, не пуст и читается как видео."""
    assert os.path.exists(filepath), f"Файл не найден: {filepath}"
    assert os.path.getsize(filepath) > 0, f"Файл пуст: {filepath}"

    cap = cv2.VideoCapture(filepath)
    assert cap.isOpened(), f"cv2 не смог открыть файл: {filepath}"
    ret, _ = cap.read()
    assert ret, f"Не удалось прочитать первый кадр: {filepath}"
    cap.release()


print("[setup] Подготовка директории и запуск predict...")

# Очищаем директорию перед тестом
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = CTModel()
model.load()
model.predict(
    TEST_INPUT,
    result_dir=OUTPUT_DIR,
    video_name=VIDEO_NAME,
    save_detection_video=True,
    save_segmentation_video=True,
)

n_lesions = len(model.lesion_masks)
expected  = _expected_files(n_lesions)
print(f"[setup] Готово. Образований: {n_lesions}, ожидаемых файлов: {len(expected)}\n")


# Тесты

def test_output_dir_exists():
    """Директория результатов должна существовать."""
    assert os.path.isdir(OUTPUT_DIR), f"Директория {OUTPUT_DIR} не создана"


def test_all_expected_files_created():
    """Все ожидаемые файлы должны быть созданы."""
    for filename in expected:
        path = os.path.join(OUTPUT_DIR, filename)
        assert os.path.exists(path), f"Ожидаемый файл не создан: {path}"


def test_no_unexpected_files():
    """Не должно быть лишних mp4-файлов в директории результатов."""
    actual_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".mp4")]
    unexpected = set(actual_files) - set(expected)
    assert not unexpected, f"Созданы неожиданные файлы: {unexpected}"


def test_detection_video_is_valid():
    """Видео детекции должно быть корректным mp4."""
    path = os.path.join(OUTPUT_DIR, f"{VIDEO_NAME}_detection.mp4")
    _assert_valid_mp4(path)


def test_combined_mask_video_is_valid():
    """Объединённое видео маски должно быть корректным mp4."""
    path = os.path.join(OUTPUT_DIR, f"{VIDEO_NAME}_mask.mp4")
    _assert_valid_mp4(path)


def test_individual_mask_videos_are_valid():
    """Индивидуальные видео масок должны быть корректными mp4 (если n > 1)."""
    if n_lesions <= 1:
        print("      (пропущено: только одно образование, индивидуальных масок нет)")
        return
    for i in range(1, n_lesions + 1):
        path = os.path.join(OUTPUT_DIR, f"{VIDEO_NAME}_mask_{i}.mp4")
        _assert_valid_mp4(path)


def test_video_frame_count():
    """Количество кадров в видео должно совпадать с числом кадров в np_video."""
    expected_frames = model.np_video.shape[0]
    path = os.path.join(OUTPUT_DIR, f"{VIDEO_NAME}_detection.mp4")

    cap = cv2.VideoCapture(path)
    actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    assert actual_frames == expected_frames, (
        f"Ожидалось {expected_frames} кадров в видео детекции, получено {actual_frames}"
    )


def test_video_frame_size():
    """Размер кадров в видео должен совпадать с размером np_video."""
    expected_h, expected_w = model.np_video.shape[1], model.np_video.shape[2]
    path = os.path.join(OUTPUT_DIR, f"{VIDEO_NAME}_detection.mp4")

    cap = cv2.VideoCapture(path)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    assert actual_h == expected_h and actual_w == expected_w, (
        f"Ожидался размер кадра {expected_h}×{expected_w}, "
        f"получен {actual_h}×{actual_w}"
    )


def test_individual_mask_count_matches_lesions():
    """Число файлов _mask_N.mp4 должно точно соответствовать числу образований."""
    individual = [
        f for f in os.listdir(OUTPUT_DIR)
        if f.startswith(f"{VIDEO_NAME}_mask_") and f.endswith(".mp4")
    ]
    if n_lesions <= 1:
        assert len(individual) == 0, (
            f"При 1 образовании индивидуальных масок быть не должно, "
            f"найдено {len(individual)}"
        )
    else:
        assert len(individual) == n_lesions, (
            f"Ожидалось {n_lesions} индивидуальных масок, найдено {len(individual)}"
        )

# Запуск

TESTS = [
    ("Директория результатов создана",               test_output_dir_exists),
    ("Все ожидаемые файлы созданы",                  test_all_expected_files_created),
    ("Нет лишних mp4-файлов",                        test_no_unexpected_files),
    ("Видео детекции валидно",                       test_detection_video_is_valid),
    ("Объединённое видео маски валидно",              test_combined_mask_video_is_valid),
    ("Индивидуальные видео масок валидны",            test_individual_mask_videos_are_valid),
    ("Количество кадров в видео совпадает",           test_video_frame_count),
    ("Размер кадров в видео совпадает",               test_video_frame_size),
    ("Число файлов масок = числу образований",        test_individual_mask_count_matches_lesions),
]

if __name__ == "__main__":
    print("=" * 60)
    print("  CTModel — тесты сохранения видео")
    print("=" * 60)

    passed = 0
    for name, fn in TESTS:
        if _run(name, fn):
            passed += 1

    total = len(TESTS)
    print(f"\n  Созданные файлы в {OUTPUT_DIR}/:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) // 1024
        print(f"    {f}  ({size_kb} KB)")

    print(f"\n  Итог: {passed}/{total} тестов прошло")
    print("=" * 60)

    sys.exit(0 if passed == total else 1)