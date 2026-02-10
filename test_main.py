from nnmodel.CTModel import CTModel
import time

start_time = time.time()

predict_test = CTModel()
predict_test.load()
predict_test.predict("./PACS/�� PACS/ID1/P1/E1/S1")


# print(predict_test.model_segmentation)
# print(predict_test.model_classification.shape)
print("[test] ROI детекции", predict_test.detected_roi)
print("[test] Размерность маски", predict_test.np_mask.shape)
print("[test] Размерность видео", predict_test.np_video.shape)
print("[test] Размерность видео с примененной маской", predict_test.video_with_mask.shape)
print("[test] Вероятности предсказаний", predict_test.proba)
print("[test] Метка класса", predict_test.label)
print("[test] Название класса", predict_test.label_name)

end_time = time.time()
execution_time = end_time - start_time
print("Время обработки:", execution_time, "секунд")

