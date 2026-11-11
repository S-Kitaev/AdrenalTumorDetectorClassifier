from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import os
import pydicom
from pydicom.errors import InvalidDicomError
import zipfile
import cv2
import numpy as np
import torch

from nnmodel.BaseNNModel import BaseNNModel
from nnmodel.settings import settings



class SegmentationModel(BaseNNModel):
    """
    Detection and segmentation of adrenal lesions on abdominal CT volumes.
    """
    def __init__(self, model_type = "segmentation"):
        super().__init__()
        self.model_path = settings[model_type]["all"]
        self.detected_roi = None
        self.np_mask = None
        self.np_video = None
        self.use_amp = torch.cuda.is_available()

    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"[Segmentation] Weights not found: {self.model_path}"
            )
        self._model = YOLO(self.model_path)

    def preprocessing(self, path: str) -> object:
        """
        Convert video / DICOM / npy input to a grayscale numpy volume
        subsampled to 54 frames at 224x224.
        """
        frames = self._video_loader(path)
        if frames is None or len(frames) == 0:
            print("[Segmentation] Error: failed to load frames from input")
            return None

        indices = np.linspace(0, len(frames) - 1, 54, dtype=int)
        frames = frames[indices]
        if len(frames) == 0:
            print("[Segmentation] Error: not enough frames after subsampling")
            return None

        self.np_video = frames
        return frames

    @staticmethod
    def make_writer(numpy_video, path, fps) -> object:
        """
        Создает объект для сохранения видео в формате mp4

        Args:
            numpy_video (np.ndarray): Numpy видео
            path (str): Директория для сохранения результатов
            fps (int): Частота кадров в сохраняемом видео

        Returns:
            object: Объект для записи видео (cv2.VideoWriter)
        """

        N, H, W = numpy_video.shape[0], numpy_video.shape[1], numpy_video.shape[2]
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(str(p), fourcc, fps, (W, H))



    def predict(self,
                numpy_video: np.ndarray = None,
                conf_threshold: float = 0.5,
                mask_threshold: float = 0.5,
                fps: int = 10,
                detection_color: tuple = (0, 255, 0),
                mask_color: tuple = (255, 255, 255),
                save_detection_video: bool = False,
                save_segmentation_video: bool = False,
                result_dir: str = None,
                video_name: str = None,
                roi_width: int = 2) -> tuple:
        """
        Метод для предсказания и сегментации образований надпочечников на КТ изображениях брюшной полости и
        сохранения видео с областями детекции и сегментацией

        Args:
            numpy_video (np.ndarray): Видео в формате np.ndarray
            conf_threshold (float): Порог уверенности для детекции
            mask_threshold (float): Порог преобразования маски в бинарное изображение (1 - белый, 0 - черный)
            fps (int): Частота кадров в сохраняемом видео
            detection_color (tuple): Цвет bounding box для детекции (зеленый)
            mask_color (tuple): Цвет маски (белый)
            save_detection_video (bool): Флаг сохранения видео с областями детекции
            save_segmentation_video (bool): Флаг сохранения видео с сегментацией
            result_dir (str): Директория для сохранения результатов
            video_name (str): Название видео при сохранении
            roi_width (int): Ширина ROI

        Returns:
            tuple: Кортеж из (Маска сегментации, список списков ROI в кадрах)
        """
        if numpy_video is None or getattr(numpy_video, "size", 0) == 0:
            print("[Segmentation] Error: video was not preprocessed")
            return None, None

        if self._model is None:
            raise RuntimeError("[Segmentation] Model is not loaded")

        writer_detector = None
        writer_mask = None
        detection_path = None
        mask_path = None

        if result_dir and (save_detection_video or save_segmentation_video):
            if not video_name:
                video_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            if save_detection_video:
                detection_path = f"{result_dir}/{video_name}_detection.mp4"
                writer_detector = self.make_writer(numpy_video, detection_path, fps)

            if save_segmentation_video:
                mask_path = f"{result_dir}/{video_name}_mask.mp4"
                writer_mask = self.make_writer(numpy_video, mask_path, fps)

        # Batch inference: list of BGR frames for YOLO
        frames_bgr = [
            cv2.cvtColor(numpy_video[i], cv2.COLOR_GRAY2BGR)
            for i in range(numpy_video.shape[0])
        ]

        predict_kwargs = dict(
            conf=conf_threshold,
            task="segment",
            verbose=False,
            stream=False,
        )
        if self.use_amp:
            predict_kwargs["half"] = True

        predictions = self._model.predict(frames_bgr, **predict_kwargs)

        rois_in_frames = []
        segmentation_mask = []
        h, w = numpy_video.shape[1], numpy_video.shape[2]

        for i, prediction in enumerate(predictions):
            det_frame = frames_bgr[i].copy()
            frame_rois = []

            if prediction.boxes is not None and len(prediction.boxes):
                for box in prediction.boxes.data.cpu().numpy():
                    x1, y1, x2, y2, conf, cls = box
                    if conf < conf_threshold:
                        continue
                    frame_rois.append([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
                    cv2.rectangle(
                        det_frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        detection_color,
                        roi_width,
                    )

            rois_in_frames.append(frame_rois)

            if writer_detector is not None:
                writer_detector.write(det_frame)

            mask_output = np.zeros((h, w), dtype=np.uint8)
            if prediction.masks is not None and prediction.masks.data.numel() > 0:
                masks = prediction.masks.data.cpu().numpy()
                for mask in masks:
                    binary = (mask > mask_threshold).astype(np.uint8)
                    if binary.shape != (h, w):
                        binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
                    binary = binary * mask_color[0]
                    mask_output = np.maximum(mask_output, binary)

            segmentation_mask.append(mask_output)

            if writer_mask is not None:
                writer_mask.write(cv2.cvtColor(mask_output, cv2.COLOR_GRAY2BGR))

        if writer_detector is not None:
            writer_detector.release()
            print(f"[Segmentation] Detection video saved to {detection_path}")

        if writer_mask is not None:
            writer_mask.release()
            print(f"[Segmentation] Segmentation video saved to {mask_path}")

        return (np.array(segmentation_mask), rois_in_frames)


    @staticmethod
    def _cv_video_loader(path: str) -> object:
        """
        Конвертирует видео файл в numpy массив кадров в градациях серого.

        Args:
            path: путь к видео файлу форматов [MP4, AVI, MKV, MOV, MPEG, WMV]

        Returns:
            np.ndarray: numpy массив кадров в градациях серого
        """

        frames = []
        cap = cv2.VideoCapture(path)  # Загрузка видео

        if not cap.isOpened():
            print(f"[Segmentation] Ошибка: не удалось открыть видео файл {path}")
            return

        while True:  # Цикл с предобработкой кадого кадра и сбором кадров видео в единый массив
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Конвертация кадра в градации серого и ресайзинг
            gray_frame_resized = cv2.resize(gray_frame, (224, 224), interpolation=cv2.INTER_AREA)
            frames.append(gray_frame_resized)

        cap.release()  # Закрываем обработку видео

        return np.array(frames)

    def _npy_video_loader(self, path: str) -> object:
        """
        Загружает numpy массив кадров в градациях серого

        Args:
            path: путь к видео файлу формата .npy

        Returns:
            np.ndarray: numpy массив кадров в градациях серого
        """

        return np.load(path)

    def _zip_video_loader(self, path: str) -> object:
        """
        Конвертирует .zip архив с кадрами форматов .dicom .dcm или dicom без расширения в numpy массив кадров в градациях серого.

        Args:
            path: путь к видео файлу формата .zip

        Returns:
            np.ndarray: numpy массив кадров в градациях серого
        """

        frames = []
        file_names = []

        with zipfile.ZipFile(path, 'r') as zip_file:

            file_list = zip_file.namelist()

            for file_name in file_list:
                if len(file_name.split("/")[-1]) > 0:
                    file_names.append(file_name)

        file_names.sort(key=lambda x: int(x.split("/")[-1][1:]))

        with zipfile.ZipFile(path, 'r') as zip_file:
            for file_name in file_names:
                with zip_file.open(file_name) as file:
                    try:
                        ds = pydicom.dcmread(file)
                        frame = ds.pixel_array
                        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                        frame = np.where(frame < 0, 0, frame)
                        frames.append(frame)
                    except InvalidDicomError as e:
                        print(f"[Segmentation] Файл {file} не может быть прочитан: {e}")

        frames = np.array(frames)
        img_float = frames.astype(np.float32)
        img_min = np.min(img_float)
        img_max = np.max(img_float)
        img_normalized = 255 * (img_float - img_min) / (img_max - img_min)
        frames = img_normalized.astype(np.uint8)

        return frames

    def _folder_video_loader(self, path: str) -> object:
        """
        Принимает на вход папку с кадрами форматов .dicom .dcm или dicom без расширения и возвращает numpy массив кадров в градациях серого.

        Args:
            path: путь к видео файлу формата .zip

        Returns:
            np.ndarray: numpy массив кадров в градациях серого
        """
        frames = []

        files = os.listdir(path)
        files.sort(key=lambda x: int(x[1:]))

        for file in files:
            try:
                ds = pydicom.dcmread(path + "/" + file)
                frame = ds.pixel_array
                frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                frame = np.where(frame < 0, 0, frame)
                frames.append(frame)
            except InvalidDicomError as e:
                print(f"[Segmentation] Файл {file} не может быть прочитан: {e}")

        frames = np.array(frames)
        img_float = frames.astype(np.float32)
        img_min = np.min(img_float)
        img_max = np.max(img_float)
        img_normalized = 255 * (img_float - img_min) / (img_max - img_min)
        frames = img_normalized.astype(np.uint8)

        return np.array(frames)

    def _video_loader(self, path: str) -> object:

        if (
                path.endswith(".mp4") or
                path.endswith(".avi") or
                path.endswith(".mkv") or
                path.endswith(".mov") or
                path.endswith(".mpeg") or
                path.endswith(".wmv")
        ):
            return self._cv_video_loader(path)

        elif path.endswith(".npy"):
            return self._npy_video_loader(path)

        elif path.endswith(".zip"):
            return self._zip_video_loader(path)

        else:
            return self._folder_video_loader(path)



