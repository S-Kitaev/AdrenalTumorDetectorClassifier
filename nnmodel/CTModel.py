import time
from typing import Optional

import numpy as np
import torch

from .BaseNNModel import BaseNNModel
from .models.ClassificationModel import ClassificationModel
from .models.SegmentationModel import SegmentationModel
from .settings import CLASS_NAMES


class CTModel(BaseNNModel):
    def __init__(self):
        self.model_segmentation = None
        self.model_classification = None
        self.detected_roi = None
        self.np_mask = None
        self.np_video = None
        self.video_with_mask = None
        self.proba = None
        self.label = None
        self.label_name = None
        self.diagnosis_proba = None
        self.diagnosis_label = None
        self.diagnosis_name = None
        self.diagnosis_result = None
        self.timings = {}

    def load(self, load_diagnosis: bool = True) -> None:
        """Load segmentation + phenotype (+ optional diagnosis) models once."""
        self.model_classification = ClassificationModel(model_type="classification")
        self.model_segmentation = SegmentationModel(model_type="segmentation")

        self.model_classification.load()
        self.model_segmentation.load()

        if load_diagnosis:
            try:
                self.model_classification.load_diagnosis_models()
            except FileNotFoundError as e:
                print(f"[CTModel] Diagnosis models not loaded: {e}")
                print("[CTModel] Phenotype inference remains available.")

    def preprocessing(self, path: str) -> object:
        """Preprocessing is performed inside SegmentationModel / ClassificationModel."""
        pass

    def predict(
        self,
        path_input: str,
        conf_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        fps: int = 10,
        detection_color: tuple = (0, 255, 0),
        mask_color: tuple = (255, 255, 255),
        save_detection_video: bool = False,
        save_segmentation_video: bool = False,
        result_dir: str = None,
        video_name: str = None,
        roi_width: int = 2,
    ) -> None:
        """
        Full pipeline: load CT volume, detect/segment lesions, predict phenotype
        and (when applicable) second-stage diagnosis.

        Phenotype class order (CLASS_NAMES):
            0 - Benign
            1 - Indeterminate
            2 - Malignant

        Diagnosis is computed only for Benign / Malignant phenotypes.
        Results are stored in instance attributes; nothing is returned.
        """
        self.timings = {}
        self.diagnosis_proba = None
        self.diagnosis_label = None
        self.diagnosis_name = None
        self.diagnosis_result = None

        t_total = time.time()

        if self.model_segmentation is not None:
            t0 = time.time()
            self.np_video = self.model_segmentation.preprocessing(path_input)

            if self.np_video is not None:
                print("[CTModel] Video loaded successfully")
            else:
                raise ValueError(f"[CTModel] Failed to load input: {path_input}")

            self.np_mask, self.detected_roi = self.model_segmentation.predict(
                self.np_video,
                conf_threshold=conf_threshold,
                mask_threshold=mask_threshold,
                fps=fps,
                detection_color=detection_color,
                mask_color=mask_color,
                save_detection_video=save_detection_video,
                save_segmentation_video=save_segmentation_video,
                result_dir=result_dir,
                video_name=video_name,
                roi_width=roi_width,
            )
            self.timings["segmentation"] = time.time() - t0
            print(f"[CTModel] Segmentation: {self.timings['segmentation']:.2f}s")

            if self.np_mask is not None and self.detected_roi is not None:
                print("[CTModel] Lesion region detected")

            if result_dir is not None and save_detection_video:
                print("[CTModel] Detection video saved to", result_dir)

            if result_dir is not None and save_segmentation_video:
                print("[CTModel] Segmentation video saved to", result_dir)

        if (
            self.model_classification is not None
            and self.np_mask is not None
            and self.np_video is not None
        ):
            t1 = time.time()
            self.video_with_mask = self.model_classification.preprocessing(
                self.np_video, self.np_mask
            )

            if self.video_with_mask is not None:
                print("[CTModel] Lesion masked on video")

            self.proba, self.label, self.label_name = self.model_classification.predict(
                self.video_with_mask
            )
            self.timings["phenotype"] = time.time() - t1
            print(f"[CTModel] Phenotype classification: {self.timings['phenotype']:.2f}s")

            if self.proba is not None and self.label is not None and self.label_name is not None:
                print("[CTModel] Phenotype predicted:", self.label_name)
                if self.label_name not in CLASS_NAMES:
                    print("[CTModel] Warning: unexpected phenotype label", self.label_name)

            t2 = time.time()
            if (
                self.model_classification.diagnosis_loaded
                and self.label_name in (CLASS_NAMES[0], CLASS_NAMES[2])
            ):
                roi_tensor = self.model_classification.prepare_roi_tensor(self.video_with_mask)
                diagnosis = self.model_classification.predict_diagnosis(
                    roi_tensor, self.label_name
                )
                self.diagnosis_result = diagnosis
                if diagnosis is not None:
                    self.diagnosis_name = diagnosis.get("diagnosis_name")
                    self.diagnosis_label = diagnosis.get("diagnosis_label")
                    self.diagnosis_proba = diagnosis.get("diagnosis_proba")
                    print("[CTModel] Diagnosis predicted:", self.diagnosis_name)
            elif self.label_name == CLASS_NAMES[1]:
                print("[CTModel] Indeterminate phenotype — diagnosis skipped")
            else:
                print("[CTModel] Diagnosis models unavailable — diagnosis skipped")

            self.timings["diagnosis"] = time.time() - t2
            print(f"[CTModel] Diagnosis: {self.timings['diagnosis']:.2f}s")

        self.timings["total"] = time.time() - t_total
        print(f"[CTModel] Total: {self.timings['total']:.2f}s")
