"""Deprecated entrypoint. Prefer: pytest tests/ -v"""

from nnmodel.CTModel import CTModel
import time

if __name__ == "__main__":
    start_time = time.time()
    predict_test = CTModel()
    predict_test.load()
    predict_test.predict("./test_data/ID17_NATIVE_SE1_Ben.mp4")
    print("[test] ROI", predict_test.detected_roi is not None)
    print("[test] phenotype", predict_test.label_name, predict_test.proba)
    print("[test] diagnosis", predict_test.diagnosis_name, predict_test.diagnosis_proba)
    print("[test] timings", predict_test.timings)
    print("Elapsed:", time.time() - start_time, "s")
