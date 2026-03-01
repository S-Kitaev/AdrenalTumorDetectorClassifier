from nnmodel.CTModel import CTModel

# Show usage of the model
path = "D:\\AdrenalTumorDetectorClassifier\\PACS\\�� PACS\\ID1\\P1\\E1\\S1"

model = CTModel()
model.load()
model.predict(path)

# Show model params
print(f"[Main] Параметры модели: {model.__dict__.keys()}")
print(f"[Main] Параметры Сегментатора: {model.model_segmentation.__dict__.keys()}")
print(f"[Main] Параметры Классификатора: {model.model_classification.__dict__.keys()}")
print(f"[Main] Размерность видео (можно вывести и само видео): {model.np_video.shape}")
print(f"[Main] Размерность маски (можно вывести и саму маску): {model.np_mask.shape}")
print(f"[Main] Область детекции: {model.detected_roi}")
print(f"[Main] Результаты: {model.results}")
