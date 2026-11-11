"""Deprecated entrypoint. Prefer: pytest tests/test_save.py -v"""

from nnmodel.CTModel import CTModel

if __name__ == "__main__":
    load_test = CTModel()
    load_test.load()
    load_test.predict(
        "./test_data/ID17_NATIVE_SE1_Ben.mp4",
        result_dir="./test_saving",
        save_detection_video=True,
        save_segmentation_video=True,
    )
    print(load_test.label_name, load_test.diagnosis_name)
