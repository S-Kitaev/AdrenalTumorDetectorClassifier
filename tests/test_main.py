from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from nnmodel.CTModel import CTModel
from nnmodel.settings import CLASS_NAMES


def test_ctmodel_predict_pipeline(mock_diagnosis_classifier, tmp_video_file, sample_video, sample_mask):
    ct = CTModel()
    ct.model_classification = mock_diagnosis_classifier

    seg = MagicMock()
    seg.preprocessing.return_value = sample_video.astype(np.uint8)
    seg.predict.return_value = (sample_mask, [[] for _ in range(sample_video.shape[0])])
    ct.model_segmentation = seg

    ct.predict(tmp_video_file)

    assert ct.label_name == "Benign"
    assert ct.label_name in CLASS_NAMES
    assert ct.diagnosis_name in ("Adenoma", "Nodular Hyperplasia", "Cyst", "Myelolipoma")
    assert ct.diagnosis_proba is not None
    assert "segmentation" in ct.timings
    assert "phenotype" in ct.timings
    assert "diagnosis" in ct.timings
    assert "total" in ct.timings


def test_ctmodel_indeterminate_skips_diagnosis(mock_diagnosis_classifier, tmp_video_file, sample_video, sample_mask):
    # Force Indeterminate phenotype
    class _Indet(torch.nn.Module):
        def forward(self, x):
            return torch.log(torch.tensor([[0.1, 0.8, 0.1]]))

    mock_diagnosis_classifier.models = [_Indet(), _Indet(), _Indet()]

    ct = CTModel()
    ct.model_classification = mock_diagnosis_classifier
    seg = MagicMock()
    seg.preprocessing.return_value = sample_video.astype(np.uint8)
    seg.predict.return_value = (sample_mask, [[] for _ in range(sample_video.shape[0])])
    ct.model_segmentation = seg

    ct.predict(tmp_video_file)
    assert ct.label_name == "Indeterminate"
    assert ct.diagnosis_name is None


def test_ctmodel_load_caches_models(monkeypatch):
    """Models should be created once in load(), not reloaded inside predict()."""
    import importlib

    created = {"clf": 0, "seg": 0}

    class FakeClf:
        def __init__(self, *a, **k):
            created["clf"] += 1
            self.diagnosis_loaded = False

        def load(self):
            pass

        def load_diagnosis_models(self):
            self.diagnosis_loaded = True

    class FakeSeg:
        def __init__(self, *a, **k):
            created["seg"] += 1

        def load(self):
            pass

    ct_mod = importlib.import_module("nnmodel.CTModel")
    monkeypatch.setattr(ct_mod, "ClassificationModel", FakeClf)
    monkeypatch.setattr(ct_mod, "SegmentationModel", FakeSeg)

    ct = CTModel()
    ct.load(load_diagnosis=True)
    assert created["clf"] == 1
    assert created["seg"] == 1
    assert ct.model_classification is not None
    assert ct.model_segmentation is not None
