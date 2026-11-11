# AdrenalTumorDetectorClassifier
НИР «Диагностика заболеваний (образований) надпочечников с помощью ИИ»

Ветка `backend` — пакет подготовки моделей для встраивания в backend.

## Структура

```
project/
├── nnmodel/
│   ├── __init__.py
│   ├── settings.py              # CLASS_NAMES, пути к весам
│   ├── BaseNNModel.py
│   ├── CTModel.py               # оркестратор пайплайна
│   └── models/
│       ├── __init__.py
│       ├── SegmentationModel.py # YOLO detect/segment
│       ├── ClassificationModel.py  # фенотип (bagging) + диагноз (MedicalNet ResNet-18)
│       ├── MedicalNetResNet.py     # 3D ResNet-18 под веса Tencent MedicalNet
│       ├── Simple3DCNN.py
│       └── ClassificationModel_bagging.py  # deprecated shim
├── model_files/                 # веса (не в git)
├── tests/
│   ├── conftest.py
│   ├── test_main.py
│   ├── test_save.py
│   ├── test_separate.py
│   ├── test_classification.py
│   └── test_segmentation.py
├── scripts/
│   └── create_diagnosis_stubs.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Пайплайн

1. **Сегментация/детекция** (YOLO) → маска и ROI  
2. **Фенотип** (ансамбль Simple3DCNN, soft voting) → `Benign` / `Indeterminate` / `Malignant`  
3. **Диагноз** (3D ResNet-18 + pretrained **Tencent MedicalNet**):  
   - Malignant → Adrenocortical Carcinoma / Pheochromocytoma  
   - Benign → каскад Solid vs Cystic/Lipomatous → Adenoma/Hyperplasia или Cyst/Myelolipoma  
   - Indeterminate → диагноз не определяется  

Порядок классов фенотипа единый: `CLASS_NAMES = ['Benign', 'Indeterminate', 'Malignant']`.

## Установка

```bash
pip install -r requirements.txt
```

Веса положите в `model_files/`:

| Файл | Назначение |
|------|------------|
| `best.pt` | YOLO сегментация |
| `model_1/2/3_best.pth` | ансамбль фенотипа |
| `resnet_18_23dataset.pth` | Tencent MedicalNet (в `MedicalNet/`) |
| `malignant_best.pth` | диагноз злокачественных |
| `benign_level1.pth` | доброкач. уровень 1 |
| `benign_level2_solid.pth` | аденома vs гиперплазия |
| `benign_level2_cystic.pth` | киста vs миелолипома |

MedicalNet: [Tencent/MedicalNet](https://github.com/Tencent/MedicalNet) — файл `resnet_18_23dataset.pth` (shortcut A).

Stub-чекпоинты диагноза (опционально с MedicalNet-инициализацией):

```bash
python scripts/create_diagnosis_stubs.py --with-medicalnet
```

## Быстрый запуск

```python
from nnmodel import CTModel

model = CTModel()
model.load()  # модели грузятся один раз
model.predict("./test_data/sample.mp4", result_dir="./test_saving",
              save_detection_video=True, save_segmentation_video=True)

print(model.label_name, model.diagnosis_name)
print(model.timings)
```

Вход: MP4/AVI/…, `.npy`, ZIP с DICOM или папка DICOM.

## Тесты

```bash
pytest tests/ -v --cov=nnmodel --cov-report=html
```

Покрытие: откройте `htmlcov/index.html`.
