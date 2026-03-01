from ultralytics import YOLO
import os
import pydicom
from pydicom.errors import InvalidDicomError
import zipfile
import cv2
import numpy as np

from nnmodel.models.BaseNNModel import BaseNNModel
from nnmodel.settings import settings



class SegmentationModel(BaseNNModel):
    """
    Модель детекции и сегментации образований надпочечников на КТ изображениях брюшной полости
    """
    def __init__(self, model_type = "segmentation"):
        super().__init__()
        self.model_path = settings[model_type]["all"]   # Путь к модели
        self.np_video: np.ndarray | None = None         # Numpy изначальное видео


    def load(self):
        self._model = YOLO(self.model_path)   # Загрузка модели


    def preprocessing(self, path: str) -> object:
        """
        Конвертирует видео файл в numpy массив кадров в градациях серого.

        Args:
            path: путь к видео файлу mp4 или zip если загружается набор dicom кадров

        Returns:
            numpy массив кадров в градациях серого
        """

        frames = self._video_loader(path)
        if frames is None or len(frames) == 0:
            print("[Segmentation] Ошибка: не удалось загрузить кадры из файла")
            return None

        indices = np.linspace(0, len(frames) - 1, 53, dtype=int)
        frames = frames[indices]

        self.np_video = frames
        return frames


    def predict(
        self,
        numpy_video: np.ndarray,
        conf_threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> tuple[np.ndarray, list]:
        """
        Покадровая детекция и сегментация образований.

        Args:
            numpy_video (np.ndarray): Видео (N, H, W), оттенки серого.
            conf_threshold (float): Порог уверенности для детекции.
            mask_threshold (float): Порог бинаризации маски сегментации.

        Returns:
            tuple:
                - np.ndarray: Объединённая бинарная маска (N, H, W).
                - list: ROI по кадрам.
                  Структура: List[List[List[List[x, y]]]]
                  rois[frame_idx] — список ROI в кадре,
                  ROI = [[x1,y1],[x2,y1],[x1,y2],[x2,y2]].
        """
        if numpy_video is None:
            print("[Segmentation] Ошибка: видео не передано в predict")
            return None, None

        rois_in_frames = []             # List[List[List[List[x1, y1], List[x2, y1], List[x1, y2], List[x2, y2]]]]
                                        # Для каждого кадра берем список всех его ROI
                                        # Для каждого ROI берем списки X и Y координат
                                        # координаты: [левый верхний угол, правый верхний угол, левый нижний угол, правый нижний угол]

        segmentation_mask = []

        for frame_idx in range(numpy_video.shape[0]):

            frame_bgr = cv2.cvtColor(numpy_video[frame_idx], cv2.COLOR_GRAY2BGR)
            prediction = self._model.predict(frame_bgr, conf=conf_threshold, task="segment", verbose=False, retina_masks=True)[0]

            # Сбор ROI
            frame_rois = []
            for box in prediction.boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = box
                if conf < conf_threshold:
                    continue
                frame_rois.append([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])

            rois_in_frames.append(frame_rois)

            # Сбор бинарной маски
            H, W = numpy_video.shape[1], numpy_video.shape[2]
            mask_output = np.zeros((H, W), dtype=np.uint8)

            if prediction.masks is not None and prediction.masks.data.numel() > 0:
                masks = prediction.masks.data.cpu().numpy()
                for mask in masks:
                    binary = (mask > mask_threshold).astype(np.uint8) * 255
                    mask_output = np.maximum(mask_output, binary)

            segmentation_mask.append(mask_output)

        return np.array(segmentation_mask), rois_in_frames


    @staticmethod
    def _cv_video_loader(path: str) -> np.ndarray | None:
        """
        Загружает видеофайл форматов MP4/AVI/MKV/MOV/MPEG/WMV в массив кадров.

        Args:
            path (str): Путь к видеофайлу.

        Returns:
            np.ndarray | None: Массив кадров (N, 224, 224) в оттенках серого.
        """
        frames = []
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            print(f"[Segmentation] Ошибка: не удалось открыть видеофайл {path}")
            return None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA)
            frames.append(gray)

        cap.release()
        return np.array(frames)


    @staticmethod
    def _npy_video_loader(path: str) -> np.ndarray:
        """
        Загружает видео из файла numpy (.npy).

        Args:
            path (str): Путь к .npy-файлу.

        Returns:
            np.ndarray: Массив кадров.
        """
        return np.load(path)


    @staticmethod
    def _zip_video_loader(path: str) -> np.ndarray:
        """
        Загружает набор DICOM-кадров из ZIP-архива.
        Поддерживаемые форматы кадров: .dcm, .dicom, без расширения.

        Args:
            path (str): Путь к ZIP-архиву.

        Returns:
            np.ndarray: Нормализованный массив кадров (N, 224, 224) uint8.
        """
        frames = []
        file_names = []

        with zipfile.ZipFile(path, 'r') as zf:
            for name in zf.namelist():
                if len(name.split("/")[-1]) > 0:
                    file_names.append(name)

        file_names.sort(key=lambda x: int(x.split("/")[-1][1:]))

        with zipfile.ZipFile(path, 'r') as zf:
            for name in file_names:
                with zf.open(name) as f:
                    try:
                        ds = pydicom.dcmread(f)
                        frame = ds.pixel_array
                        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                        frame = np.where(frame < 0, 0, frame)
                        frames.append(frame)
                    except InvalidDicomError as e:
                        print(f"[Segmentation] Файл {name} не может быть прочитан: {e}")

        frames = np.array(frames, dtype=np.float32)
        img_min, img_max = np.min(frames), np.max(frames)
        frames = (255 * (frames - img_min) / (img_max - img_min)).astype(np.uint8)
        return frames


    @staticmethod
    def _folder_video_loader(path: str) -> np.ndarray:
        """
        Загружает набор DICOM-кадров из папки.
        Поддерживаемые форматы кадров: .dcm, .dicom, без расширения.

        Args:
            path (str): Путь к папке с DICOM-файлами.

        Returns:
            np.ndarray: Нормализованный массив кадров (N, 224, 224) uint8.
        """
        frames = []
        files = sorted(os.listdir(path), key=lambda x: int(x[1:]))

        for file in files:
            try:
                ds = pydicom.dcmread(os.path.join(path, file))
                frame = ds.pixel_array
                frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                frame = np.where(frame < 0, 0, frame)
                frames.append(frame)
            except InvalidDicomError as e:
                print(f"[Segmentation] Файл {file} не может быть прочитан: {e}")

        frames = np.array(frames, dtype=np.float32)
        img_min, img_max = np.min(frames), np.max(frames)
        frames = (255 * (frames - img_min) / (img_max - img_min)).astype(np.uint8)
        return frames


    def _video_loader(self, path: str) -> np.ndarray | None:
        """
        Определяет тип входного файла/папки и вызывает соответствующий загрузчик.

        Args:
            path (str): Путь к файлу или папке.

        Returns:
            np.ndarray | None: Массив кадров или None при неизвестном формате.
        """
        ext = path.lower()

        if any(ext.endswith(e) for e in (".mp4", ".avi", ".mkv", ".mov", ".mpeg", ".wmv")):
            return self._cv_video_loader(path)
        elif ext.endswith(".npy"):
            return self._npy_video_loader(path)
        elif ext.endswith(".zip"):
            return self._zip_video_loader(path)
        else:
            return self._folder_video_loader(path)
