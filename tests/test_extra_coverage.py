"""Extra coverage for loaders, Simple3DCNN and diagnosis load helpers."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from nnmodel.models.ClassificationModel import ClassificationModel
from nnmodel.models.SegmentationModel import SegmentationModel
from nnmodel.models.Simple3DCNN import Simple3DCNN
from nnmodel.settings import CLASS_NAMES


def test_simple3dcnn_forward():
    model = Simple3DCNN(num_classes=3)
    model.eval()
    x = torch.randn(1, 1, 53, 100, 100)
    y = model(x)
    assert y.shape == (1, 3)


def test_phenotype_load_from_dir(tmp_path):
    # Save a tiny compatible state_dict via Simple3DCNN
    net = Simple3DCNN(num_classes=3)
    for i in range(1, 4):
        torch.save(net.state_dict(), tmp_path / f"model_{i}_best.pth")

    clf = ClassificationModel(device="cpu")
    clf.model_dir = str(tmp_path)
    clf.model_path = str(tmp_path) + "/"
    clf.load()
    assert len(clf.models) == 3
    info = clf.get_model_info()
    assert info["labels"] == CLASS_NAMES
    assert info["num_models"] == 3


def test_phenotype_load_missing_dir(tmp_path):
    clf = ClassificationModel(device="cpu")
    clf.model_dir = str(tmp_path / "absent")
    clf.model_path = str(tmp_path / "absent" / "x.pth")
    with pytest.raises(FileNotFoundError):
        clf.load()


def test_load_medicalnet_diagnosis(tmp_path, monkeypatch):
    import importlib

    from nnmodel.models.MedicalNetResNet import MedicalNetResNet18, build_medicalnet_resnet18

    paths = {
        "mal": tmp_path / "malignant_best.pth",
        "l1": tmp_path / "benign_level1.pth",
        "l2s": tmp_path / "benign_level2_solid.pth",
        "l2c": tmp_path / "benign_level2_cystic.pth",
    }
    for p in paths.values():
        m = build_medicalnet_resnet18(num_classes=2, medicalnet_path=None)
        torch.save(m.state_dict(), p)

    # Fake MedicalNet pretrain (backbone-compatible keys only)
    backbone = MedicalNetResNet18(num_classes=2)
    medicalnet_ckpt = tmp_path / "resnet_18_23dataset.pth"
    torch.save({"state_dict": {f"module.{k}": v for k, v in backbone.state_dict().items()
                               if not k.startswith("fc.")}}, medicalnet_ckpt)

    cm_mod = importlib.import_module("nnmodel.models.ClassificationModel")
    monkeypatch.setattr(cm_mod, "MALIGNANT_MODEL_PATH", str(paths["mal"]))
    monkeypatch.setattr(cm_mod, "BENIGN_LEVEL1_PATH", str(paths["l1"]))
    monkeypatch.setattr(cm_mod, "BENIGN_LEVEL2_SOLID_PATH", str(paths["l2s"]))
    monkeypatch.setattr(cm_mod, "BENIGN_LEVEL2_CYSTIC_PATH", str(paths["l2c"]))
    monkeypatch.setattr(cm_mod, "MEDICALNET_WEIGHTS_PATH", str(medicalnet_ckpt))

    clf = ClassificationModel(device="cpu")
    clf.load_diagnosis_models()
    assert clf.diagnosis_loaded
    assert clf.malignant_model is not None
    x = torch.randn(1, 1, 16, 32, 32)
    y = clf.malignant_model(x)
    assert y.shape == (1, 2)


def test_segmentation_load_missing(tmp_path):
    model = SegmentationModel()
    model.model_path = str(tmp_path / "missing.pt")
    with pytest.raises(FileNotFoundError):
        model.load()


def test_cv_video_loader_roundtrip(tmp_path):
    import cv2

    video_path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (32, 32)
    )
    for _ in range(10):
        writer.write(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    writer.release()

    model = SegmentationModel()
    frames = model._cv_video_loader(str(video_path))
    assert frames is not None
    assert frames.ndim == 3
    assert frames.shape[1:] == (224, 224)


def test_npy_loader(tmp_path):
    path = tmp_path / "vol.npy"
    np.save(path, np.random.randint(0, 255, (20, 64, 64), dtype=np.uint8))
    model = SegmentationModel()
    arr = model._npy_video_loader(str(path))
    assert arr.shape[0] == 20


def test_video_loader_dispatch(tmp_path):
    path = tmp_path / "vol.npy"
    np.save(path, np.ones((5, 8, 8), dtype=np.uint8))
    model = SegmentationModel()
    out = model._video_loader(str(path))
    assert out.shape[0] == 5


def test_bagging_shim_import():
    from nnmodel.models.ClassificationModel_bagging import ClassificationModel as CM

    assert CM is ClassificationModel


def test_ctmodel_malignant_branch(mock_diagnosis_classifier, tmp_video_file, sample_video, sample_mask):
    from unittest.mock import MagicMock
    from nnmodel.CTModel import CTModel

    class _Mal(torch.nn.Module):
        def forward(self, x):
            return torch.log(torch.tensor([[0.05, 0.05, 0.9]]))

    mock_diagnosis_classifier.models = [_Mal(), _Mal(), _Mal()]
    ct = CTModel()
    ct.model_classification = mock_diagnosis_classifier
    seg = MagicMock()
    seg.preprocessing.return_value = sample_video.astype(np.uint8)
    seg.predict.return_value = (sample_mask, [[] for _ in range(sample_video.shape[0])])
    ct.model_segmentation = seg
    ct.predict(tmp_video_file)
    assert ct.label_name == "Malignant"
    assert ct.diagnosis_name in ("Adrenocortical Carcinoma", "Pheochromocytoma")


def test_predict_with_amp_flag(mock_diagnosis_classifier, sample_roi_53):
    mock_diagnosis_classifier.use_amp = False
    proba, label, name = mock_diagnosis_classifier.predict(sample_roi_53)
    assert name in CLASS_NAMES
