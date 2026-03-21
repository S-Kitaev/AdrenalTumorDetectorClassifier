import torch
import numpy as np
from torchvision import transforms
from skimage.transform import resize

from .Simple3DCNN import Simple3DCNN
from nnmodel.models.BaseNNModel import BaseNNModel
from nnmodel.settings import settings

class ClassificationModel(BaseNNModel):
    """
    Модель классификации образований надпочечников на три класса:
    Benign (доброкачественные), Indeterminate (неопределенные), Malignant (злокачественные).
    Использует ансамбль из 3-х моделей Simple3DCNN.
    """
    def __init__(self, model_type="classification"):
        super().__init__()
        self.model_path = settings[model_type]

        self.train_mean = 0.0009
        self.train_std = 0.0211

        self.transforms = transforms.Compose([
            transforms.Lambda(lambda x: self._preprocess_volume(x)),
        ])

        self.labels = ["Benign", "Indeterminate", "Malignant"]
        self.models = []
        self.num_models = 3  # Количество моделей в ансамбле

    def _preprocess_volume(self, volume):
        """
        Предобработка 3D объема для подачи в модель.
        """
        volume = np.array(volume, dtype=np.float32)

        # Нормализация к [0, 1] если данные в диапазоне [0, 255]
        if np.max(volume) > 1.0:
            volume = volume / 255.0

        volume = torch.tensor(volume, dtype=torch.float32)
        volume = (volume - self.train_mean) / self.train_std

        return volume

    def load(self):
        """
        Загрузка ансамбля моделей Simple3DCNN.
        Загружает веса 3-х предобученных моделей.
        """
        for i in range(self.num_models):

            model_path = self.model_path[str(i+1)]

            if not model_path:
                raise FileNotFoundError(f"[ClassificationModel] Модели не найдены, проверьте settings")

            model = Simple3DCNN(num_classes=3)

            try:
                state_dict = torch.load(model_path, map_location='cpu')

                if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                    model.load_state_dict(state_dict['model_state_dict'])
                else:
                    model.load_state_dict(state_dict)

                model.eval()
                self.models.append(model)
            except Exception as e:
                print(f"[ClassificationModel] Ошибка загрузки модели {model_path}: {e}")

    def preprocessing(self, np_video: np.ndarray = None, np_mask: np.ndarray = None) -> object:
        """
        Предобработка видео и маски для классификации.

        Args:
            np_video (np.ndarray): Входное видео (N, H, W)
            np_mask (np.ndarray): Бинарная маска сегментации (N, H, W)

        Returns:
            np.ndarray: Предобработанный 3D объем (53, 100, 100)
        """

        # Проверка наличия и размерности маски и видео
        assert np_mask is not None, "Маска не задана"
        assert np_video is not None, "Видео не задано"
        assert np_mask.shape == np_video.shape, f"Размерности маски и видео не совпадают: {np_mask.shape} vs {np_video.shape}"

        # Применяем маску к видео
        videos_mult_mask = np_video * np_mask

        target_size = (100, 100)
        # Изменение размера с {videos_mult_mask.shape[1:]} на {target_size}")
        if videos_mult_mask.shape[1:] != target_size:
            data_resized = np.zeros((videos_mult_mask.shape[0], target_size[0], target_size[1]),
                                    dtype=np.float64)

            for i in range(videos_mult_mask.shape[0]):
                data_resized[i] = resize(videos_mult_mask[i], target_size,
                                         preserve_range=True,
                                         anti_aliasing=True)

            videos_mult_mask = data_resized

        expected_shape = (53, 100, 100)
        if videos_mult_mask.shape != expected_shape:
            raise ValueError(f"Размерность видео не соответствует ожидаемому: {videos_mult_mask.shape} != {expected_shape}")

        return videos_mult_mask

    def _soft_voting(self, probabilities):
        """
        Применяет мягкое голосование для объединения предсказаний ансамбля моделей.

        Args:
            probabilities (list): Список вероятностей от каждой модели

        Returns:
            np.ndarray: Усредненные вероятности по всем моделям
        """
        probabilities = np.array(probabilities, dtype=np.float32)
        avg_probs = np.mean(probabilities, axis=0)
        return avg_probs

    def predict(self, np_videos_mult_mask: np.ndarray = None):
        """
        Выполняет классификацию 3D объема с использованием ансамбля моделей.

        Args:
            np_videos_mult_mask (np.ndarray): Предобработанный 3D объем (53, 100, 100)

        Returns:
            tuple:
                - torch.Tensor: Вероятности для каждого класса
                - int: Индекс предсказанного класса (0: Benign, 1: Indeterminate, 2: Malignant)
                - str: Название предсказанного класса
        """
        with torch.no_grad():
            # Преобразование данных
            transformed = self.transforms(np_videos_mult_mask)

            model_input = transformed.unsqueeze(0).unsqueeze(0)

            all_probs = []
            for model in self.models:
                output = model(model_input)
                probs = torch.softmax(output, dim=1).cpu().numpy()
                all_probs.append(probs[0])

            avg_probs = self._soft_voting(all_probs)

            predicted_proba = torch.from_numpy(np.array(avg_probs, dtype=np.float32)).unsqueeze(0)
            predicted_label = torch.argmax(predicted_proba, dim=1)
            predicted_label_idx = int(predicted_label.item())

        return (predicted_proba, predicted_label_idx, self.labels[predicted_label_idx])






