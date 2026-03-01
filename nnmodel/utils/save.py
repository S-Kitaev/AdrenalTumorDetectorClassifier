import cv2
import numpy as np
from pathlib import Path
from datetime import datetime


def make_writer(numpy_video: np.ndarray, path: str, fps: int) -> cv2.VideoWriter:
    """
    Создаёт объект cv2.VideoWriter для записи видео в формате mp4.

    Args:
        numpy_video (np.ndarray): Numpy-видео, используется только для получения размеров кадра (N, H, W)
        path (str): Путь к выходному файлу
        fps (int): Частота кадров

    Returns:
        cv2.VideoWriter: Объект для записи видео
    """
    H, W = numpy_video.shape[1], numpy_video.shape[2]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(str(p), fourcc, fps, (W, H))


def save_videos(
    numpy_video: np.ndarray,
    segmentation_mask: np.ndarray,
    rois_in_frames: list,
    lesion_masks: list,
    result_dir: str,
    video_name: str = None,
    fps: int = 10,
    detection_color: tuple = (0, 255, 0),
    mask_color: tuple = (255, 255, 255),
    roi_width: int = 2
) -> None:
    """
    Сохраняет видео детекции и сегментации.

    Логика именования:
        - Всегда сохраняется одно видео детекции (все bbox) и одно объединённое видео маски.
        - Если образований больше одного, дополнительно сохраняется по одному видео маски
          для каждого образования отдельно (с суффиксом _mask_1, _mask_2, ...).

    Args:
        numpy_video (np.ndarray): Исходное видео (N, H, W), оттенки серого
        segmentation_mask (np.ndarray): Объединённая маска всех образований (N, H, W)
        rois_in_frames (list): Список ROI по кадрам (из SegmentationModel.predict)
        lesion_masks (list[np.ndarray]): Список масок по одной на образование (N, H, W)
        result_dir (str): Директория для сохранения результатов
        video_name (str | None): Базовое имя файлов; если None — текущее время
        fps (int): Частота кадров
        detection_color (tuple): Цвет bounding box (BGR)
        mask_color (tuple): Цвет маски (яркость канала при конвертации серого → BGR)
        roi_width (int): Толщина рамки bounding box
    """
    if video_name is None:
        video_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # ── Видео детекции (все bbox на одном видео) ──────────────────────────────
    detection_path = f"{result_dir}/{video_name}_detection.mp4"
    writer_det = make_writer(numpy_video, detection_path, fps)

    for frame_idx in range(numpy_video.shape[0]):
        frame_bgr = cv2.cvtColor(numpy_video[frame_idx], cv2.COLOR_GRAY2BGR)
        for roi in rois_in_frames[frame_idx]:
            pt1 = (int(roi[0][0]), int(roi[0][1]))
            pt2 = (int(roi[3][0]), int(roi[3][1]))
            cv2.rectangle(frame_bgr, pt1, pt2, detection_color, roi_width)
        writer_det.write(frame_bgr)

    writer_det.release()
    print(f"[Save] Видео детекции сохранено: {detection_path}")

    # ── Объединённая маска (все образования) ──────────────────────────────────
    combined_mask_path = f"{result_dir}/{video_name}_mask.mp4"
    _write_mask_video(numpy_video, segmentation_mask, combined_mask_path, fps)
    print(f"[Save] Объединённая маска сохранена: {combined_mask_path}")

    # ── Индивидуальные маски (только если образований > 1) ───────────────────
    if len(lesion_masks) > 1:
        for i, lesion_mask in enumerate(lesion_masks, start=1):
            lesion_path = f"{result_dir}/{video_name}_mask_{i}.mp4"
            _write_mask_video(numpy_video, lesion_mask, lesion_path, fps)
            print(f"[Save] Маска образования {i} сохранена: {lesion_path}")


def _write_mask_video(
    numpy_video: np.ndarray,
    mask: np.ndarray,
    path: str,
    fps: int
) -> None:
    """
    Записывает видео маски сегментации в файл.

    Args:
        numpy_video (np.ndarray): Источник размеров кадра (N, H, W)
        mask (np.ndarray): Маска (N, H, W)
        path (str): Путь к выходному файлу
        fps (int): Частота кадров
    """
    writer = make_writer(numpy_video, path, fps)
    for frame_idx in range(mask.shape[0]):
        mask_bgr = cv2.cvtColor(mask[frame_idx], cv2.COLOR_GRAY2BGR)
        writer.write(mask_bgr)
    writer.release()