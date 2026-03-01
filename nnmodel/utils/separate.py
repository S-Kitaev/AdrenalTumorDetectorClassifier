import numpy as np
from scipy.optimize import linear_sum_assignment


def _roi_centroid(roi: list) -> tuple:
    """
    Вычисляет центр прямоугольной области детекции.

    Args:
        roi (list): Список из 4 точек [[x1,y1],[x2,y1],[x1,y2],[x2,y2]]

    Returns:
        tuple: Координаты центра (cx, cy)
    """
    x1, y1 = roi[0]
    x2, y2 = roi[3]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _match_centroids(prev_centroids: list, curr_centroids: list) -> dict:
    """
    Сопоставляет текущие центроиды предыдущим методом венгерского алгоритма
    (минимизация суммарного квадрата расстояний).

    Args:
        prev_centroids (list): Центроиды образований на предыдущем шаге [(cx, cy), ...]
        curr_centroids (list): Центроиды образований на текущем кадре [(cx, cy), ...]

    Returns:
        dict: Словарь {curr_idx: prev_idx} — сопоставление текущих индексов предыдущим
    """
    n_prev = len(prev_centroids)
    n_curr = len(curr_centroids)

    # Матрица стоимостей: квадрат евклидова расстояния
    cost = np.zeros((n_prev, n_curr), dtype=np.float64)
    for i, (px, py) in enumerate(prev_centroids):
        for j, (cx, cy) in enumerate(curr_centroids):
            cost[i, j] = (px - cx) ** 2 + (py - cy) ** 2

    if n_prev <= n_curr:
        row_ind, col_ind = linear_sum_assignment(cost)
        return {int(c): int(r) for r, c in zip(row_ind, col_ind)}
    else:
        row_ind, col_ind = linear_sum_assignment(cost.T)
        return {int(r): int(c) for r, c in zip(row_ind, col_ind)}


def separate_lesions(
    segmentation_mask: np.ndarray,
    rois_in_frames: list
) -> list:
    """
    Разделяет маску с несколькими образованиями на список отдельных масок,
    по одной на каждое обнаруженное образование.

    Алгоритм:
        1. По каждому кадру определяем центры детекций.
        2. Сопоставляем текущие центры с предыдущими (венгерский алгоритм),
           считая, что центр опухоли практически не смещается между соседними кадрами.
        3. Если число детекций увеличилось — добавляем новые треки.
        4. Если уменьшилось — соответствующие треки больше не пополняются.
        5. Для каждого трека и каждого кадра, где трек активен,
           вырезаем из комбинированной маски область данного образования
           и помещаем её в отдельную маску (остальное — нули).

    Args:
        segmentation_mask (np.ndarray): Объединённая маска (N, H, W)
        rois_in_frames (list): Список детекций по кадрам.
            Структура: List[List[List[List[x, y]]]]
            rois_in_frames[frame_idx] — список ROI в кадре,
            ROI = [[x1,y1],[x2,y1],[x1,y2],[x2,y2]]

    Returns:
        list[np.ndarray]: Список масок, по одной на образование, каждая (N, H, W).
            Если образований не обнаружено — возвращает [segmentation_mask].
    """
    n_frames, H, W = segmentation_mask.shape

    # lesion_tracks[i] = {frame_idx: roi} — история трека i-го образования
    lesion_tracks: list[dict] = []

    # active = [(track_idx, centroid)] — активные на текущем кадре треки
    active: list[tuple] = []

    for frame_idx, frame_rois in enumerate(rois_in_frames):
        if len(frame_rois) == 0:
            # Кадр без детекций — активные треки «пропускают» кадр, не закрываются
            continue

        curr_centroids = [_roi_centroid(roi) for roi in frame_rois]

        if len(active) == 0:
            # Первые детекции — инициализируем треки
            for roi, centroid in zip(frame_rois, curr_centroids):
                lesion_tracks.append({frame_idx: roi})
                active.append((len(lesion_tracks) - 1, centroid))
            continue

        prev_centroids = [a[1] for a in active]
        n_prev = len(prev_centroids)
        n_curr = len(curr_centroids)

        # Сопоставление: curr_idx -> prev_idx (или track_idx если n_prev <= n_curr)
        matching = _match_centroids(prev_centroids, curr_centroids)

        new_active: list[tuple] = []
        matched_curr = set()

        if n_prev <= n_curr:
            # matching: {curr_idx: prev_active_idx}
            for curr_idx, prev_active_idx in matching.items():
                track_idx = active[prev_active_idx][0]
                lesion_tracks[track_idx][frame_idx] = frame_rois[curr_idx]
                new_active.append((track_idx, curr_centroids[curr_idx]))
                matched_curr.add(curr_idx)

            # Новые образования (не вошли в сопоставление)
            for curr_idx in range(n_curr):
                if curr_idx not in matched_curr:
                    lesion_tracks.append({frame_idx: frame_rois[curr_idx]})
                    new_active.append((len(lesion_tracks) - 1, curr_centroids[curr_idx]))

        else:
            # n_prev > n_curr: часть образований исчезла
            # matching: {curr_idx: prev_active_idx}
            for curr_idx, prev_active_idx in matching.items():
                track_idx = active[prev_active_idx][0]
                lesion_tracks[track_idx][frame_idx] = frame_rois[curr_idx]
                new_active.append((track_idx, curr_centroids[curr_idx]))

        active = new_active

    # Нет ни одного трека — возвращаем исходную маску как единственную
    if not lesion_tracks:
        return [segmentation_mask]

    # Строим отдельную маску для каждого трека
    lesion_masks = []
    for track in lesion_tracks:
        lesion_mask = np.zeros((n_frames, H, W), dtype=segmentation_mask.dtype)

        for frame_idx, roi in track.items():
            x1 = max(0, int(roi[0][0]))
            y1 = max(0, int(roi[0][1]))
            x2 = min(W, int(roi[3][0]))
            y2 = min(H, int(roi[3][1]))

            frame_canvas = np.zeros((H, W), dtype=segmentation_mask.dtype)
            frame_canvas[y1:y2, x1:x2] = segmentation_mask[frame_idx, y1:y2, x1:x2]
            lesion_mask[frame_idx] = frame_canvas

        lesion_masks.append(lesion_mask)

    return lesion_masks