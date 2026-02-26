from abc import ABC, abstractmethod

class BaseNNModel(ABC):

    def __init__(self):
        self._model = None

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Функция, в которой обределяется структура NN и
        происходит загрузка весов модели в self._model

        params:
          path - путь к файлу, в котором содержатся веса модели
        """
        ...

    @abstractmethod
    def preprocessing(self, path: str) -> object:
        """
        Функция, котороя предобрабатывает изображение к виду,
        с которым может взаимодействовать модель из self._model

        params:
          path - путь к файлу (изображению .tiff/.png/.mp4), который будет
                использоваться для предсказания

        return - возвращает предобработанное изображение/видео
        """
        ...

    @abstractmethod
    def predict(self, path: str) -> object:
        """
        Функция, в которой предобработанное изображение подается
        на входы NN (self._model) и возвращается результат работы NN

        params:
          path - путь к файлу (изображению .tiff/.png/.mp4), который будет
                использоваться для предсказания

        return - результаты предсказания
        """
        ...