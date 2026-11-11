import os
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from nnmodel.models.SegmentationModel import SegmentationModel


def test_make_writer_creates_mp4(tmp_path, sample_video):
    out = tmp_path / "out.mp4"
    writer = SegmentationModel.make_writer(sample_video.astype(np.uint8), str(out), fps=5)
    assert writer is not None
    frame = cv2.cvtColor(sample_video[0].astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for _ in range(3):
        writer.write(frame)
    writer.release()
    assert out.exists()
    assert out.stat().st_size > 0


def test_preprocessing_npy_shape(tmp_video_file):
    model = SegmentationModel()
    # Avoid loading YOLO weights for preprocessing-only test
    frames = model.preprocessing(tmp_video_file)
    assert frames is not None
    assert frames.shape == (54, 224, 224)


def test_preprocessing_empty_returns_none(tmp_path):
    model = SegmentationModel()
    bad = tmp_path / "empty.npy"
    np.save(bad, np.zeros((0, 10, 10), dtype=np.uint8))
    result = model.preprocessing(str(bad))
    assert result is None


def test_predict_batch_mocked(sample_video):
    model = SegmentationModel()
    model._model = MagicMock()
    model.use_amp = False

    h, w = sample_video.shape[1], sample_video.shape[2]
    fake_pred = MagicMock()
    fake_pred.boxes = None
    fake_pred.masks = None
    model._model.predict.return_value = [fake_pred] * sample_video.shape[0]

    mask, rois = model.predict(sample_video.astype(np.uint8), conf_threshold=0.5)
    assert mask.shape == sample_video.shape
    assert len(rois) == sample_video.shape[0]
    model._model.predict.assert_called_once()
    args, kwargs = model._model.predict.call_args
    assert isinstance(args[0], list)
    assert len(args[0]) == sample_video.shape[0]


def test_save_videos_mocked(tmp_path, sample_video):
    model = SegmentationModel()
    model._model = MagicMock()
    model.use_amp = False

    fake_pred = MagicMock()
    fake_pred.boxes = None
    fake_pred.masks = None
    model._model.predict.return_value = [fake_pred] * sample_video.shape[0]

    result_dir = tmp_path / "out"
    result_dir.mkdir()
    model.predict(
        sample_video.astype(np.uint8),
        save_detection_video=True,
        save_segmentation_video=True,
        result_dir=str(result_dir),
        video_name="unit",
        fps=5,
    )
    det = result_dir / "unit_detection.mp4"
    seg = result_dir / "unit_mask.mp4"
    assert det.exists() and det.stat().st_size > 0
    assert seg.exists() and seg.stat().st_size > 0
