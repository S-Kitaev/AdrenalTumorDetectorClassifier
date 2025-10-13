from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np

from nnmodel.BaseNNModel import BaseNNModel
from nnmodel.settings import settings

class SegmentationModel(BaseNNModel):
    """
    Модель детекции и сегментации образований надпочечников на КТ изображениях брюшной полости
    """
    def __init__(self, model_type = "segmentation"):
        super().__init__()
        self.model_path = settings[model_type]["all"]   # Путь к модели
        self.detected_roi = None                        # Метки bbox образований
        self.np_mask = None                             # Numpy маска сегментации
        self.np_video = None                            # Numpy изначальное видео

    def load(self):
        self._model = YOLO(self.model_path)   # Загрузка модели

    def preprocessing(self, path: str) -> object:
        """
        Конвертирует видео файл в numpy массив кадров в градациях серого.

        Args:
            path: путь к видео файлу

        Returns:
            numpy массив кадров в градациях серого
        """
        cap = cv2.VideoCapture(path)                    # Загрузка видео

        if not cap.isOpened():
            print(f"Ошибка: не удалось открыть видео файл {path}")
            return

        frames = []                                     # Массив кадров
        while True:                                     # Цикл с предобработкой кадого кадра и сбором кадров видео в единый массив
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Конвертация кадра в градации серого и ресайзинг
            gray_frame_resized = cv2.resize(gray_frame, (224, 224), interpolation=cv2.INTER_AREA)
            frames.append(gray_frame_resized)

        cap.release()                                   # Закрываем обработку видео

        indices = np.linspace(0, len(frames) - 1, 54, dtype=int)
        frames = np.array(frames)[indices]              # При необходимости используем часть кадров                     # При необходимости используем часть кадров
        if len(frames) == 0:
            print("Ошибка: не удалось найти видео")
            return

        self.np_video = frames
        return np.array(frames)

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
        if numpy_video is None:
            print("Ошибка: видео не было обработано")
            return


        if result_dir and (save_detection_video or save_segmentation_video):    # Создание объектов для записи видео
            if not video_name:                                                  # По умолчанию имя видео - время создания
                time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                video_name = time

            if save_detection_video:                                            # Создание объекта для сохранения видео детекции
                detection_path = f"{result_dir}/{video_name}_detection.mp4"
                writer_detector = self.make_writer(numpy_video, detection_path, fps)

            if save_segmentation_video:                                         # Создание объекта для сохранения видео маски сегментации
                mask_path = f"{result_dir}/{video_name}_mask.mp4"
                writer_mask = self.make_writer(numpy_video, mask_path, fps)

        rois_in_frames = []             # List[List[List[List[x1, y1], List[x2, y1], List[x1, y2], List[x2, y2]]]]
                                        # Для каждого кадра берем список всех его ROI
                                        # Для каждого ROI берем списки X и Y координат
                                        # координаты: [левый верхний угол, правый верхний угол, левый нижний угол, правый нижний угол]

        segmentation_mask = []
        for i in range(numpy_video.shape[0]):  # Цикл по кадрам видео
            frame = numpy_video[i]
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            prediction = self._model.predict(frame_bgr, conf=conf_threshold, task="segment", verbose=False)[0]

            # Детекция
            det_frame = frame_bgr.copy()
            frame_rois = []
            for box in prediction.boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = box
                if conf < conf_threshold:
                    continue

                frame_roi = [[x1, y1], [x2, y1], [x1, y2], [x2, y2]]
                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))
                cv2.rectangle(det_frame, pt1, pt2, detection_color, roi_width)

                frame_rois.append(frame_roi)

            rois_in_frames.append(frame_rois)

            if save_detection_video:
                writer_detector.write(det_frame)

            # Сегментация
            mask_output = np.zeros((numpy_video.shape[1], numpy_video.shape[2]), dtype=np.uint8)

            if prediction.masks is not None and prediction.masks.data.numel() > 0:
                masks = prediction.masks.data.cpu().numpy()
                for mask in masks:
                    binary = (mask > mask_threshold).astype(np.uint8) * mask_color[0]  # маска белая
                    mask_output = np.maximum(mask_output, binary)

            segmentation_mask.append(mask_output)

            if save_segmentation_video:
                mask_bgr = cv2.cvtColor(mask_output, cv2.COLOR_GRAY2BGR)
                writer_mask.write(mask_bgr)

        if save_detection_video:
            writer_detector.release()
            print(f"[Segmentation] Видео с детекцией сохранено в {detection_path}")

        if save_segmentation_video:
            writer_mask.release()
            print(f"[Segmentation] Видео с сегментацией сохранено в {mask_path}")

        return (np.array(segmentation_mask), rois_in_frames)