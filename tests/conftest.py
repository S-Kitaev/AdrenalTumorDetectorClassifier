import numpy as np
import pytest
import torch
import torch.nn as nn


@pytest.fixture
def sample_video():
    return np.random.rand(54, 224, 224).astype(np.float32) * 255.0


@pytest.fixture
def sample_mask():
    mask = np.zeros((54, 224, 224), dtype=np.uint8)
    mask[:, 50:170, 50:170] = 255
    return mask


@pytest.fixture
def sample_roi_53():
    return np.random.rand(53, 100, 100).astype(np.float32)


class _TinyBinary(nn.Module):
    def __init__(self, prefer_class: int = 0):
        super().__init__()
        self.prefer_class = prefer_class
        self.fc = nn.Linear(1, 2)
        with torch.no_grad():
            self.fc.weight.zero_()
            self.fc.bias.zero_()
            self.fc.bias[prefer_class] = 10.0

    def forward(self, x):
        # Ignore spatial structure; return biased logits
        batch = x.shape[0]
        ones = torch.ones(batch, 1, device=x.device, dtype=x.dtype)
        return self.fc(ones)


@pytest.fixture
def mock_diagnosis_classifier(monkeypatch):
    """ClassificationModel with phenotype models mocked and diagnosis models as tiny nets."""
    from nnmodel.models.ClassificationModel import ClassificationModel

    clf = ClassificationModel(device="cpu")
    clf.use_amp = False

    class _PhenotypeStub(nn.Module):
        def __init__(self, probs):
            super().__init__()
            self.probs = torch.tensor(probs, dtype=torch.float32)

        def forward(self, x):
            # Convert probs to logits via log
            logits = torch.log(self.probs.clamp(min=1e-6)).unsqueeze(0).expand(x.shape[0], -1)
            return logits

    # Soft-voting of three identical Benign-favoring models
    clf.models = [
        _PhenotypeStub([0.8, 0.1, 0.1]),
        _PhenotypeStub([0.7, 0.2, 0.1]),
        _PhenotypeStub([0.75, 0.15, 0.1]),
    ]

    clf.malignant_model = _TinyBinary(prefer_class=0)  # Adrenocortical Carcinoma
    clf.benign_l1 = _TinyBinary(prefer_class=0)  # Solid
    clf.benign_l2_solid = _TinyBinary(prefer_class=0)  # Adenoma
    clf.benign_l2_cystic = _TinyBinary(prefer_class=1)  # Myelolipoma
    clf.diagnosis_loaded = True
    return clf


@pytest.fixture
def tmp_video_file(tmp_path, sample_video):
    path = tmp_path / "sample.npy"
    np.save(path, sample_video.astype(np.uint8))
    return str(path)
