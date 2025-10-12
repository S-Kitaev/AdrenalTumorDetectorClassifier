import torch
import numpy as np
from torchvision import transforms

from nnmodel.BaseNNModel import BaseNNModel
from nnmodel.settings import settings

class ClassificationModel(BaseNNModel):
    def __init__(self, model_type = "classification"):
        super().__init__()
        self.model_path = settings[model_type]["all"]
        self.transforms = transforms.Compose([transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32) / 255.0),
                                             transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.labels = ["Benign", "Indererminate", "Malignant"]


    def load(self):
        self._model = torch.load(self.model_path)
        self._model.eval()

    def preprocessing(self,
                      np_video: np.ndarray = None,
                      np_mask: np.ndarray = None) -> object:

        """
        Формирует видео, на котором обозначено только образование, при помощи умножения матрицы на маску

        Args:
            np_video (np.ndarray): Видео в формате np.ndarray
            np_mask (np.ndarray): Маска в формате np.ndarray
        Returns:
            numpy массив кадров найденного образования
        """

        # Проверка наличия и размерности маски и видео
        assert np_mask is not None, "Маска не задана"
        assert np_video is not None, "Видео не задано"
        assert np_mask.shape == np_video.shape, "Размерности маски и видео не совпадают"

        videos_mult_mask = np_video * np_mask

        return videos_mult_mask


    def predict(self, np_videos_mult_mask: np.ndarray = None):

        """
        Описание

        Args:
            np_videos_mult_mask (np.ndarray): Видео выделенного образования в формате np.ndarray
        Returns:
            Tuple: кортеж (вероятность класса, метка класса, название класса)
        """
        with torch.no_grad():
            np_videos_mult_mask = torch.tensor(np_videos_mult_mask, dtype=torch.float32).unsqueeze(0)
            transformed_videos_mult_mask = self.transforms(np_videos_mult_mask)

            model_output = self._model(transformed_videos_mult_mask)
            predicted_proba = torch.softmax(model_output, dim=1)
            predicted_label = torch.argmax(predicted_proba, dim=1)

        return (predicted_proba, predicted_label, self.labels[predicted_label])






