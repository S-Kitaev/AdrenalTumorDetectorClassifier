import numpy as np
import pytest
import torch

from nnmodel.settings import CLASS_NAMES


def test_class_names_order():
    assert CLASS_NAMES == ["Benign", "Indeterminate", "Malignant"]


def test_import_nnmodel():
    import nnmodel

    assert nnmodel.CTModel is not None
    assert nnmodel.CLASS_NAMES == CLASS_NAMES


def test_classification_preprocessing_shape(sample_video, sample_mask):
    from nnmodel.models.ClassificationModel import ClassificationModel

    model = ClassificationModel(device="cpu")
    roi = model.preprocessing(sample_video, sample_mask)
    assert roi.shape == (53, 100, 100), f"Wrong shape: {roi.shape}"


def test_classification_preprocessing_empty_raises(sample_mask):
    from nnmodel.models.ClassificationModel import ClassificationModel

    model = ClassificationModel(device="cpu")
    empty = np.zeros((0, 224, 224), dtype=np.float32)
    with pytest.raises(ValueError):
        model.preprocessing(empty, sample_mask[:0])


def test_classification_preprocessing_shape_mismatch(sample_video, sample_mask):
    from nnmodel.models.ClassificationModel import ClassificationModel

    model = ClassificationModel(device="cpu")
    with pytest.raises(ValueError):
        model.preprocessing(sample_video, sample_mask[:10])


def test_soft_voting_and_phenotype_predict(mock_diagnosis_classifier, sample_roi_53):
    proba, label, name = mock_diagnosis_classifier.predict(sample_roi_53)
    assert name in CLASS_NAMES
    assert name == "Benign"
    assert label == 0
    assert proba.shape == (1, 3)
    assert torch.isclose(proba.sum(), torch.tensor(1.0), atol=1e-4)


def test_predict_malignant(mock_diagnosis_classifier, sample_roi_53):
    roi = mock_diagnosis_classifier.prepare_roi_tensor(sample_roi_53)
    result = mock_diagnosis_classifier.predict_malignant(roi)
    assert result["diagnosis_name"] in (
        "Adrenocortical Carcinoma",
        "Pheochromocytoma",
    )
    assert "diagnosis_proba" in result
    assert result["branch"] == "malignant"


def test_predict_benign_cascade_solid(mock_diagnosis_classifier, sample_roi_53):
    roi = mock_diagnosis_classifier.prepare_roi_tensor(sample_roi_53)
    result = mock_diagnosis_classifier.predict_benign(roi)
    assert result["level1_name"] == "Solid"
    assert result["diagnosis_name"] in ("Adenoma", "Nodular Hyperplasia")
    assert result["branch"] == "benign_solid"
    assert "level1" in result["diagnosis_proba"]
    assert "level2" in result["diagnosis_proba"]


def test_predict_benign_cascade_cystic(mock_diagnosis_classifier, sample_roi_53):
    # Force level1 -> Cystic/Lipomatous
    mock_diagnosis_classifier.benign_l1 = type(mock_diagnosis_classifier.benign_l1)(prefer_class=1)
    roi = mock_diagnosis_classifier.prepare_roi_tensor(sample_roi_53)
    result = mock_diagnosis_classifier.predict_benign(roi)
    assert result["level1_name"] == "Cystic/Lipomatous"
    assert result["diagnosis_name"] in ("Cyst", "Myelolipoma")
    assert result["branch"] == "benign_cystic"


def test_predict_diagnosis_routes(mock_diagnosis_classifier, sample_roi_53):
    roi = mock_diagnosis_classifier.prepare_roi_tensor(sample_roi_53)
    benign = mock_diagnosis_classifier.predict_diagnosis(roi, "Benign")
    malignant = mock_diagnosis_classifier.predict_diagnosis(roi, "Malignant")
    indeterminate = mock_diagnosis_classifier.predict_diagnosis(roi, "Indeterminate")
    assert benign is not None and benign["diagnosis_name"]
    assert malignant is not None and malignant["diagnosis_name"]
    assert indeterminate is None


def test_prepare_roi_tensor_shape(mock_diagnosis_classifier, sample_roi_53):
    tensor = mock_diagnosis_classifier.prepare_roi_tensor(sample_roi_53)
    assert tensor.shape[0] == 1
    assert tensor.shape[1] == 1  # MedicalNet: single CT channel
    assert tensor.shape[2] == 53
    assert tensor.shape[3:] == (112, 112)
