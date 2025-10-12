# AdrenalTumorDetectorClasifier
НИР "Диагностика заболеваний (образований) надпочечников с помощью ИИ"

### branch - ```backend```

Векта подготовки моделей для встраивания в backend

Cтруктура:
```commandline
nnmodel/
    ├── models/
    │   ├── SegmentationModel.py        # Модель сегментатора
    │   └── ClassificationModel.py      # Модель классификатора
    ├── BaseNNModel.py                  # Базовый класс для моделей
    ├── CTModel.py                      # Общая модель нейросети КТ
    └── settings.py                     # Настройки загрузки моделей
test.py                                 # Тесты модели и пример использования(comming soon)
```