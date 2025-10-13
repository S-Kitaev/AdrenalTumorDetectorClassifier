from nnmodel.CTModel import CTModel

load_test = CTModel()
load_test.load()
load_test.predict("./test_data/ID12_NATIVE_SE1.mp4",
                     result_dir="./test_saving",
                     save_detection_video=True,
                     save_segmentation_video=True)