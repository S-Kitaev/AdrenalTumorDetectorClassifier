from .models.BaseNNModel import BaseNNModel
from .models.ClassificationModel import ClassificationModel
from .models.SegmentationModel import SegmentationModel
from .utils.separate import separate_lesions
from .utils.save import save_videos

class CTModel(BaseNNModel):
    """
    Метамодель для анализа КТ-снимков брюшной полости.
    Оркестрирует последовательность:
        1. Загрузка и предобработка видео (SegmentationModel.preprocessing)
        2. Покадровая детекция и сегментация образований (SegmentationModel.predict)
        3. Разделение нескольких образований на отдельные маски (separate_lesions)
        4. Классификация каждого образования независимо (ClassificationModel)
    Результаты классификации доступны через атрибут results.
    """

    def __init__(self):
        self.model_segmentation: SegmentationModel | None = None
        self.model_classification: ClassificationModel | None = None

        self.np_video = None
        self.np_mask = None
        self.detected_roi = None

        self.lesion_masks: list = []            # Разделённые маски: по одной на каждое образование
        self.results: list[tuple] = []          # Результаты классификации: список (proba, label_idx, label_name)

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

    def predict(
            self,
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
            roi_width: int = 2
        ) -> None:

        """
        Полный цикл анализа КТ-снимка:
        загрузка → сегментация → разделение образований → классификация каждого.

        Все результаты сохраняются в атрибутах экземпляра.
        Если образований несколько, атрибут results содержит список результатов.
        Атрибуты proba / label / label_name указывают на первое образование
        (для обратной совместимости со сценарием с одним образованием).

        Args:
            path_input (str): Путь к входному файлу или папке.
            conf_threshold (float): Порог уверенности для детектора YOLO
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

        # Шаг 1: загрузка и предобработка видео
        if self.model_segmentation is None:
            return

        self.np_video = self.model_segmentation.preprocessing(path_input)
        if self.np_video is None:
            return

        print("[CTModel] Видео успешно загружено")

        # Шаг 2: детекция и сегментация
        self.np_mask, self.detected_roi = self.model_segmentation.predict(
            self.np_video,
            conf_threshold= 0.16,               # Лучшее значение согласно исследованию, информация из отчета от Китаев С.М. 19.11.25
            mask_threshold=mask_threshold,
        )

        if self.np_mask is None or self.detected_roi is None:
            return

        print("[CTModel] Область с образованием успешно выделена")

        # Шаг 3: разделение образований
        self.lesion_masks = separate_lesions(self.np_mask, self.detected_roi)
        n_lesions = len(self.lesion_masks)
        print(f"[CTModel] Обнаружено образований: {n_lesions}")

        # Шаг 4: сохранение видео
        if result_dir is not None and (save_detection_video or save_segmentation_video):
            save_videos(
                numpy_video=self.np_video,
                segmentation_mask=self.np_mask,
                rois_in_frames=self.detected_roi,
                lesion_masks=self.lesion_masks,
                result_dir=result_dir,
                video_name=video_name,
                fps=fps,
                detection_color=detection_color,
                mask_color=mask_color,
                roi_width=roi_width,
            )
            if save_detection_video:
                print(f"[CTModel] Видео детекции сохранено в папку {result_dir}")
            if save_segmentation_video:
                print(f"[CTModel] Видео сегментации сохранено в папку {result_dir}")

        # Шаг 5: классификация каждого образования
        if self.model_classification is None:
            return

        self.results = []
        for i, lesion_mask in enumerate(self.lesion_masks, start=1):
            video_with_mask = self.model_classification.preprocessing(
                self.np_video, lesion_mask
            )
            if video_with_mask is None:
                print(f"[CTModel] Образование {i}: ошибка предобработки, пропущено")
                continue

            proba, label_idx, label_name = self.model_classification.predict(video_with_mask)
            self.results.append((proba, label_idx, label_name))
            print(f"[CTModel] Образование {i}: Индекс - {label_idx}; Фенотип - {label_name}; Вероятность - {proba}")

        if not self.results:
            return