from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from nnmodel.models.SegmentationModel import SegmentationModel


def test_save_detection_and_mask_videos(tmp_path, sample_video):
    model = SegmentationModel()
    model._model = MagicMock()
    model.use_amp = False

    fake_pred = MagicMock()
    fake_pred.boxes = None
    fake_pred.masks = None
    n = sample_video.shape[0]
    model._model.predict.return_value = [fake_pred] * n

    out_dir = tmp_path / "save"
    out_dir.mkdir()
    mask, rois = model.predict(
        sample_video.astype(np.uint8),
        save_detection_video=True,
        save_segmentation_video=True,
        result_dir=str(out_dir),
        video_name="save_test",
        fps=5,
    )

    assert mask.shape == sample_video.shape
    det = out_dir / "save_test_detection.mp4"
    seg = out_dir / "save_test_mask.mp4"
    assert det.exists() and det.stat().st_size > 0
    assert seg.exists() and seg.stat().st_size > 0
