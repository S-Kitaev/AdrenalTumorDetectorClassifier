from .BaseNNModel import BaseNNModel

from .models.ClassificationModel import ClassificationModel
from .models.SegmentationModel import SegmentationModel

class CTModel(BaseNNModel):
    def __init__(self):
        self.model_segmentation = None
        self.model_classification = None
        self.detected_roi = None
        self.np_mask = None
        self.np_video = None
        self.video_with_mask = None
        self.proba = None
        self.label = None
        self.label_name = None

    def load(self) -> None:
        self.model_classification = ClassificationModel(model_type = "classification")
        self.model_segmentation = SegmentationModel(model_type = "segmentation")

        self.model_classification.load()
        self.model_segmentation.load()

    def preprocessing(self, path: str) -> object:
        """
        Подготовка данных выполняется в классах ClassificationModel и SegmentationModel.
        Так как модели работают последовательно, то реализоваться предподготовка будет в predict
        """
        pass

    def predict(self,
                path_input: str,
                conf_threshold: float = 0.5,
                mask_threshold: float = 0.5,
                fps: int = 10,
                detection_color: tuple = (0, 255, 0),
                mask_color: tuple = (255, 255, 255),
                save_detection_video: bool = False,
                save_segmentation_video: bool = False,
                result_dir: str = None,
                video_name: str = None,
                roi_width: int = 2) -> None:

        """
        Предсказание координат области с образовнаием, формирование видео и маски в формате np.ndarray,
        сохранение видео с областями детекции и сегментацией в директорию result_dir, а также предсказание класса
        образования (0 - Злокачетвенное, 1 - Неопределенное, 2 - Доброкачетвенное) и названия образования по даннному
        списку

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
            Ничего не возвращает, но печатает лог выполнения предсказания
            Все результаты сохраняются в локальные атрибуты класса
        """

        if self.model_segmentation is not None:

            self.np_video = self.model_segmentation.preprocessing(path_input)

            if self.np_video is not None:
                print("[CTModel] Видео успешно загружено")

            self.np_mask, self.detected_roi = self.model_segmentation.predict(self.np_video,
                                                                              conf_threshold = conf_threshold,
                                                                              mask_threshold = mask_threshold,
                                                                              fps = fps,
                                                                              detection_color = detection_color,
                                                                              mask_color = mask_color,
                                                                              save_detection_video = save_detection_video,
                                                                              save_segmentation_video = save_segmentation_video,
                                                                              result_dir = result_dir,
                                                                              video_name = video_name,
                                                                              roi_width = roi_width
                                                                              )
            if self.np_mask is not None and self.detected_roi is not None:
                print("[CTModel] Область с образованием успешно выделена")

            if result_dir is not None and save_detection_video == True:
                print("[CTModel] Видео с областью детекции успешно сохранено в папку", result_dir)

            if result_dir is not None and save_segmentation_video == True:
                print("[CTModel] Видео с областью сегментации успешно сохранено в папку", result_dir)

        if self.model_classification is not None and self.np_mask is not None and self.np_video is not None:

            self.video_with_mask = self.model_classification.preprocessing(self.np_video, self.np_mask)

            if self.video_with_mask is not None:
                print("[CTModel] Образование успешно выделено маской на видео")

            self.proba, self.label, self.label_name = self.model_classification.predict(self.video_with_mask)

            if self.proba is not None and self.label is not None and self.label_name is not None:
                print("[CTModel] Модель успешно предсказала класс образования")
                print("[CTModel] Предсказание класса образования:", self.label_name)