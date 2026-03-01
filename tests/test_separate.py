import numpy as np
import sys
import traceback

from nnmodel.utils.separate import separate_lesions

# Вспомогательные функции

def _make_mask(n_frames: int, H: int, W: int) -> np.ndarray:
    """Создаёт пустую маску (N, H, W)."""
    return np.zeros((n_frames, H, W), dtype=np.uint8)


def _draw_rect(mask: np.ndarray, frame_idx: int, x1: int, y1: int, x2: int, y2: int) -> None:
    """Закрашивает прямоугольник в маске на заданном кадре."""
    mask[frame_idx, y1:y2, x1:x2] = 255


def _make_roi(x1: float, y1: float, x2: float, y2: float) -> list:
    """Создаёт ROI в формате [[x1,y1],[x2,y1],[x1,y2],[x2,y2]]."""
    return [[x1, y1], [x2, y1], [x1, y2], [x2, y2]]


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


# Тесты

def test_no_detections_returns_original_mask():
    """Если ни на одном кадре нет детекций — возвращаем исходную маску как единственную."""
    mask = _make_mask(10, 64, 64)
    rois = [[] for _ in range(10)]

    result = separate_lesions(mask, rois)

    assert len(result) == 1, f"Ожидался 1 результат, получено {len(result)}"
    assert np.array_equal(result[0], mask), "Маска должна совпадать с исходной"


def test_single_lesion_throughout():
    """Одно образование на всех кадрах — должен вернуться список из 1 маски."""
    N, H, W = 10, 64, 64
    mask = _make_mask(N, H, W)
    rois = []

    for i in range(N):
        _draw_rect(mask, i, 10, 10, 30, 30)
        rois.append([_make_roi(10, 10, 30, 30)])

    result = separate_lesions(mask, rois)

    assert len(result) == 1, f"Ожидался 1 трек, получено {len(result)}"
    # Область образования должна присутствовать в маске
    assert result[0].sum() > 0, "Маска образования не должна быть пустой"


def test_two_lesions_non_overlapping_frames():
    """
    1-е образование на кадрах 2–6, 2-е образование на кадрах 4–8.
    Ожидается 2 отдельных трека.
    """
    N, H, W = 10, 64, 64
    mask = _make_mask(N, H, W)
    rois = [[] for _ in range(N)]

    # Образование A: левый верхний угол
    for i in range(2, 7):
        _draw_rect(mask, i, 5, 5, 20, 20)
        rois[i].append(_make_roi(5, 5, 20, 20))

    # Образование B: правый нижний угол
    for i in range(4, 9):
        _draw_rect(mask, i, 40, 40, 60, 60)
        rois[i].append(_make_roi(40, 40, 60, 60))

    result = separate_lesions(mask, rois)

    assert len(result) == 2, f"Ожидалось 2 трека, получено {len(result)}"

    # Суммарная площадь каждого трека не должна быть нулевой
    for idx, lesion_mask in enumerate(result):
        assert lesion_mask.sum() > 0, f"Трек {idx + 1} содержит пустую маску"


def test_two_lesions_correct_spatial_separation():
    """
    Проверяем, что два пространственно разделённых образования не попадают
    в чужие маски (пиксели из области A не в маске B и наоборот).
    """
    N, H, W = 6, 64, 64
    mask = _make_mask(N, H, W)
    rois = [[] for _ in range(N)]

    # Образование A: строго слева
    for i in range(N):
        _draw_rect(mask, i, 2, 2, 15, 15)
        rois[i].append(_make_roi(2, 2, 15, 15))

    # Образование B: строго справа
    for i in range(N):
        _draw_rect(mask, i, 45, 45, 60, 60)
        rois[i].append(_make_roi(45, 45, 60, 60))

    result = separate_lesions(mask, rois)
    assert len(result) == 2, f"Ожидалось 2 трека, получено {len(result)}"

    # Определяем, какой трек — A, какой — B
    sum_left  = [r[:, 2:15, 2:15].sum() for r in result]
    sum_right = [r[:, 45:60, 45:60].sum() for r in result]

    # Один трек должен быть «левым», другой — «правым»
    assert (sum_left[0] > 0) != (sum_left[1] > 0), \
        "Оба трека содержат пиксели из левой области — разделение некорректно"
    assert (sum_right[0] > 0) != (sum_right[1] > 0), \
        "Оба трека содержат пиксели из правой области — разделение некорректно"


def test_lesion_appearing_midway():
    """
    Одно образование появляется на кадре 5 из 10.
    До кадра 5 — пустые кадры, после — тоже пустые.
    Ожидается 1 трек, в котором активны только кадры 5–9.
    """
    N, H, W = 10, 64, 64
    mask = _make_mask(N, H, W)
    rois = [[] for _ in range(N)]

    for i in range(5, N):
        _draw_rect(mask, i, 20, 20, 40, 40)
        rois[i].append(_make_roi(20, 20, 40, 40))

    result = separate_lesions(mask, rois)

    assert len(result) == 1
    # Кадры до 5 должны быть пустыми
    assert result[0][:5].sum() == 0, "Кадры до появления образования должны быть пустыми"
    # Кадры с 5 — непустыми
    assert result[0][5:].sum() > 0, "Кадры после появления образования должны содержать маску"


def test_output_shape_matches_input():
    """Форма каждой выходной маски должна совпадать с формой входной."""
    N, H, W = 53, 224, 224
    mask = _make_mask(N, H, W)
    rois = [[] for _ in range(N)]

    for i in range(10, 30):
        _draw_rect(mask, i, 50, 50, 100, 100)
        rois[i].append(_make_roi(50, 50, 100, 100))

    result = separate_lesions(mask, rois)

    for idx, lesion_mask in enumerate(result):
        assert lesion_mask.shape == (N, H, W), \
            f"Трек {idx + 1}: ожидалась форма {(N, H, W)}, получена {lesion_mask.shape}"


def test_three_lesions():
    """Три одновременно присутствующих образования — должно быть 3 трека."""
    N, H, W = 8, 128, 128
    mask = _make_mask(N, H, W)
    rois = [[] for _ in range(N)]

    regions = [
        (5, 5, 30, 30),    # A
        (50, 5, 75, 30),   # B
        (5, 80, 30, 105),  # C
    ]

    for i in range(N):
        frame_rois = []
        for x1, y1, x2, y2 in regions:
            _draw_rect(mask, i, x1, y1, x2, y2)
            frame_rois.append(_make_roi(x1, y1, x2, y2))
        rois[i] = frame_rois

    result = separate_lesions(mask, rois)

    assert len(result) == 3, f"Ожидалось 3 трека, получено {len(result)}"
    for idx, lesion_mask in enumerate(result):
        assert lesion_mask.sum() > 0, f"Трек {idx + 1} содержит пустую маску"


# Запуск

TESTS = [
    ("Нет детекций → исходная маска",           test_no_detections_returns_original_mask),
    ("Одно образование на всех кадрах",          test_single_lesion_throughout),
    ("Два образования, разные временные окна",   test_two_lesions_non_overlapping_frames),
    ("Два образования, пространственное разделение", test_two_lesions_correct_spatial_separation),
    ("Образование появляется в середине",        test_lesion_appearing_midway),
    ("Размерность выходных масок совпадает",     test_output_shape_matches_input),
    ("Три одновременных образования",            test_three_lesions),
]

if __name__ == "__main__":
    print("=" * 60)
    print("  separate_lesions — юнит-тесты")
    print("=" * 60)

    passed = 0
    for name, fn in TESTS:
        if _run(name, fn):
            passed += 1

    total = len(TESTS)
    print("-" * 60)
    print(f"  Итог: {passed}/{total} тестов прошло")
    print("=" * 60)

    sys.exit(0 if passed == total else 1)
