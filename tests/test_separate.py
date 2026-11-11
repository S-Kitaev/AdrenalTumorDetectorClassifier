"""Separate unit checks for diagnosis helpers and error handling."""

import pytest
import torch

from nnmodel.models.ClassificationModel import ClassificationModel
from nnmodel.settings import (
    BENIGN_LEVEL1_PATH,
    BENIGN_LEVEL2_CYSTIC_PATH,
    BENIGN_LEVEL2_SOLID_PATH,
    MALIGNANT_MODEL_PATH,
)


def test_load_r3d_missing_weights():
    clf = ClassificationModel(device="cpu")
    with pytest.raises(FileNotFoundError):
        clf._load_r3d_model("./model_files/__missing_diag__.pth", num_classes=2)


def test_load_diagnosis_models_missing(monkeypatch, tmp_path):
    import importlib

    cm_mod = importlib.import_module("nnmodel.models.ClassificationModel")
    clf = ClassificationModel(device="cpu")
    monkeypatch.setattr(cm_mod, "MALIGNANT_MODEL_PATH", str(tmp_path / "nope.pth"))
    with pytest.raises(FileNotFoundError):
        clf.load_diagnosis_models()


def test_predict_without_models_raises(sample_roi_53):
    clf = ClassificationModel(device="cpu")
    with pytest.raises(RuntimeError):
        clf.predict(sample_roi_53)


def test_prepare_roi_empty_raises():
    clf = ClassificationModel(device="cpu")
    with pytest.raises(ValueError):
        clf.prepare_roi_tensor(None)
