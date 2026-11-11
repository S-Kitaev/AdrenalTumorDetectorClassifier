import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize
from torchvision import transforms

from nnmodel.BaseNNModel import BaseNNModel
from nnmodel.models.MedicalNetResNet import (
    build_medicalnet_resnet18,
    _normalize_state_dict,
)
from nnmodel.models.Simple3DCNN import Simple3DCNN
from nnmodel.settings import (
    BENIGN_LEVEL1_CLASS_NAMES,
    BENIGN_LEVEL1_PATH,
    BENIGN_LEVEL2_CYSTIC_CLASS_NAMES,
    BENIGN_LEVEL2_CYSTIC_PATH,
    BENIGN_LEVEL2_SOLID_CLASS_NAMES,
    BENIGN_LEVEL2_SOLID_PATH,
    CLASS_NAMES,
    MALIGNANT_CLASS_NAMES,
    MALIGNANT_MODEL_PATH,
    MEDICALNET_WEIGHTS_PATH,
    settings,
)


class ClassificationModel(BaseNNModel):
    """Phenotype ensemble (Simple3DCNN bagging) + diagnosis (MedicalNet 3D ResNet-18)."""

    def __init__(self, model_type: str = "classification", device: Optional[str] = None):
        super().__init__()
        self.model_path = settings[model_type]["all"]
        if self.model_path.endswith(("/", "\\")) or os.path.isdir(self.model_path):
            self.model_dir = self.model_path.rstrip("/\\") or "./model_files"
        else:
            self.model_dir = os.path.dirname(self.model_path) or "./model_files"

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.use_amp = self.device.type == "cuda"

        self.train_mean = 0.0009
        self.train_std = 0.0211
        self.transforms = transforms.Compose([
            transforms.Lambda(lambda x: self._preprocess_volume(x)),
        ])

        self.labels = list(CLASS_NAMES)
        self.models = []
        self.num_models = 3
        self.model_weights = None

        self.malignant_model = None
        self.benign_l1 = None
        self.benign_l2_solid = None
        self.benign_l2_cystic = None
        self.diagnosis_loaded = False

    def _preprocess_volume(self, volume):
        volume = np.array(volume, dtype=np.float32)
        if np.max(volume) > 1.0:
            volume = volume / 255.0
        volume = torch.tensor(volume, dtype=torch.float32)
        volume = (volume - self.train_mean) / self.train_std
        return volume

    def load(self) -> None:
        if os.path.isdir(self.model_dir):
            model_files = []
            for i in range(1, self.num_models + 1):
                model_path = os.path.join(self.model_dir, f"model_{i}_best.pth")
                if os.path.exists(model_path):
                    model_files.append(model_path)
                else:
                    print(f"[ClassificationModel] Warning: model not found: {model_path}")

            if not model_files:
                all_pths = [
                    f for f in os.listdir(self.model_dir)
                    if f.endswith(".pth") and f.startswith("model_") and "best" in f
                    and "old" not in f and "malignant" not in f and "benign" not in f
                ]
                model_files = [os.path.join(self.model_dir, f) for f in sorted(all_pths)[: self.num_models]]
        else:
            model_files = [self.model_path] if os.path.exists(self.model_path) else []

        if not model_files:
            raise FileNotFoundError(
                f"[ClassificationModel] No phenotype models found in {self.model_dir}"
            )

        for i, model_path in enumerate(model_files[: self.num_models]):
            model = Simple3DCNN(num_classes=len(CLASS_NAMES))
            try:
                try:
                    state_dict = torch.load(
                        model_path, map_location=self.device, weights_only=False
                    )
                except TypeError:
                    state_dict = torch.load(model_path, map_location=self.device)
                if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                    model.load_state_dict(state_dict["model_state_dict"])
                else:
                    model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                self.models.append(model)
                print(f"[ClassificationModel] Phenotype model {i + 1} loaded from {model_path}")
            except Exception as e:
                print(f"[ClassificationModel] Failed to load {model_path}: {e}")

        if not self.models:
            raise RuntimeError("[ClassificationModel] Failed to load any phenotype model")

        self.model_weights = None

    def _load_r3d_model(self, weights_path: str, num_classes: int) -> nn.Module:
        """Backward-compatible alias for diagnosis model loading. """
        return self._load_diagnosis_model(weights_path, num_classes)

    def _load_diagnosis_model(self, weights_path: str, num_classes: int) -> nn.Module:
        """
        Build MedicalNet ResNet-18, initialize backbone from Tencent MedicalNet,
        then load task-specific fine-tuned weights.
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"[ClassificationModel] Diagnosis weights not found: {weights_path}"
            )

        medicalnet_path = MEDICALNET_WEIGHTS_PATH if os.path.exists(MEDICALNET_WEIGHTS_PATH) else None
        if medicalnet_path is None:
            print(
                f"[ClassificationModel] Warning: MedicalNet weights not found at "
                f"{MEDICALNET_WEIGHTS_PATH}; building ResNet-18 without MedicalNet init"
            )

        model = build_medicalnet_resnet18(
            num_classes=num_classes,
            medicalnet_path=medicalnet_path,
            map_location=self.device,
        )

        try:
            ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(weights_path, map_location=self.device)

        state_dict = _normalize_state_dict(ckpt)
        model_dict = model.state_dict()
        filtered = {
            k: v
            for k, v in state_dict.items()
            if k in model_dict and tuple(v.shape) == tuple(model_dict[k].shape)
        }
        if not filtered:
            raise RuntimeError(
                f"[ClassificationModel] No matching tensors in diagnosis checkpoint: {weights_path}"
            )
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
        print(
            f"[ClassificationModel] Diagnosis checkpoint loaded "
            f"({len(filtered)}/{len(model_dict)} tensors) from {weights_path}"
        )

        model.to(self.device)
        model.eval()
        return model

    def load_diagnosis_models(self) -> None:
        """Load second-stage MedicalNet ResNet-18 diagnosis models (once per CTModel lifetime)."""
        self.malignant_model = self._load_diagnosis_model(MALIGNANT_MODEL_PATH, num_classes=2)
        self.benign_l1 = self._load_diagnosis_model(BENIGN_LEVEL1_PATH, num_classes=2)
        self.benign_l2_solid = self._load_diagnosis_model(BENIGN_LEVEL2_SOLID_PATH, num_classes=2)
        self.benign_l2_cystic = self._load_diagnosis_model(BENIGN_LEVEL2_CYSTIC_PATH, num_classes=2)
        self.diagnosis_loaded = True
        print("[ClassificationModel] Diagnosis models (MedicalNet ResNet-18) loaded")

    def preprocessing(
        self,
        np_video: np.ndarray = None,
        np_mask: np.ndarray = None,
        target_size: tuple = (100, 100),
    ) -> np.ndarray:
        if np_mask is None:
            raise ValueError("Mask is not set")
        if np_video is None:
            raise ValueError("Video is not set")
        if np_mask.shape != np_video.shape:
            raise ValueError(
                f"Mask/video shape mismatch: {np_mask.shape} vs {np_video.shape}"
            )
        if np_video.size == 0 or np_video.shape[0] == 0:
            raise ValueError("Empty video array")

        videos_mult_mask = np_video.astype(np.float64) * (
            np_mask.astype(np.float64) / (255.0 if np_mask.max() > 1 else 1.0)
        )

        if videos_mult_mask.shape[1:] != target_size:
            data_resized = np.zeros(
                (videos_mult_mask.shape[0], target_size[0], target_size[1]),
                dtype=np.float64,
            )
            for i in range(videos_mult_mask.shape[0]):
                data_resized[i] = resize(
                    videos_mult_mask[i],
                    target_size,
                    preserve_range=True,
                    anti_aliasing=True,
                )
            videos_mult_mask = data_resized

        expected_depth = 53
        if videos_mult_mask.shape[0] != expected_depth:
            if videos_mult_mask.shape[0] > expected_depth:
                start_idx = (videos_mult_mask.shape[0] - expected_depth) // 2
                videos_mult_mask = videos_mult_mask[start_idx : start_idx + expected_depth]
            else:
                last_frame = videos_mult_mask[-1:]
                padding = np.tile(
                    last_frame, (expected_depth - videos_mult_mask.shape[0], 1, 1)
                )
                videos_mult_mask = np.concatenate([videos_mult_mask, padding], axis=0)

        return videos_mult_mask

    def prepare_roi_tensor(
        self,
        volume: np.ndarray,
        spatial_size: tuple = (112, 112),
    ) -> torch.Tensor:
        """Convert grayscale ROI (T,H,W) to MedicalNet input (1, 1, T, H, W)."""
        if volume is None or volume.size == 0:
            raise ValueError("Empty ROI volume")

        vol = np.asarray(volume, dtype=np.float32)
        if vol.ndim != 3:
            raise ValueError(f"Expected ROI (T,H,W), got shape {vol.shape}")

        if vol.shape[1:] != spatial_size:
            resized = np.zeros((vol.shape[0], spatial_size[0], spatial_size[1]), dtype=np.float32)
            for i in range(vol.shape[0]):
                resized[i] = resize(
                    vol[i], spatial_size, preserve_range=True, anti_aliasing=True
                ).astype(np.float32)
            vol = resized

        if vol.max() > 1.0:
            vol = vol / 255.0

        tensor = torch.from_numpy(vol).float()  # T,H,W
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # 1,1,T,H,W (MedicalNet: 1 channel)
        return tensor.to(self.device)

    def _soft_voting(self, probabilities) -> np.ndarray:
        probabilities = np.array(probabilities, dtype=np.float32)
        return np.mean(probabilities, axis=0)

    def _forward_logits(self, model: nn.Module, roi_tensor: torch.Tensor) -> torch.Tensor:
        if self.use_amp:
            with torch.cuda.amp.autocast():
                return model(roi_tensor)
        return model(roi_tensor)

    def predict(self, np_videos_mult_mask: np.ndarray = None) -> Tuple[torch.Tensor, int, str]:
        if np_videos_mult_mask is None or np_videos_mult_mask.size == 0:
            raise ValueError("Empty input for phenotype prediction")
        if not self.models:
            raise RuntimeError("Phenotype models are not loaded")

        with torch.no_grad():
            transformed = self.transforms(np_videos_mult_mask)
            model_input = transformed.unsqueeze(0).unsqueeze(0).to(self.device)

            all_probs = []
            for model in self.models:
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        output = model(model_input)
                else:
                    output = model(model_input)
                probs = torch.softmax(output.float(), dim=1).cpu().numpy()
                all_probs.append(probs[0])

            avg_probs = self._soft_voting(all_probs)
            predicted_proba = torch.from_numpy(np.array(avg_probs, dtype=np.float32)).unsqueeze(0)
            predicted_label_idx = int(torch.argmax(predicted_proba, dim=1).item())

            print(
                f"[ClassificationModel] Phenotype probs: "
                f"{CLASS_NAMES[0]}={avg_probs[0]:.4f}, "
                f"{CLASS_NAMES[1]}={avg_probs[1]:.4f}, "
                f"{CLASS_NAMES[2]}={avg_probs[2]:.4f}"
            )

        return predicted_proba, predicted_label_idx, self.labels[predicted_label_idx]

    def _predict_binary(
        self,
        model: nn.Module,
        roi_tensor: torch.Tensor,
        class_names: list,
    ) -> Dict[str, Any]:
        if model is None:
            raise RuntimeError("Diagnosis model is not loaded")
        with torch.no_grad():
            logits = self._forward_logits(model, roi_tensor)
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()[0]
        label_idx = int(np.argmax(probs))
        return {
            "proba": {name: float(probs[i]) for i, name in enumerate(class_names)},
            "label": label_idx,
            "name": class_names[label_idx],
            "probs_vector": probs,
        }

    def predict_malignant(self, roi_tensor: torch.Tensor) -> Dict[str, Any]:
        result = self._predict_binary(
            self.malignant_model, roi_tensor, MALIGNANT_CLASS_NAMES
        )
        return {
            "diagnosis_name": result["name"],
            "diagnosis_label": result["label"],
            "diagnosis_proba": result["proba"],
            "branch": "malignant",
        }

    def predict_benign(self, roi_tensor: torch.Tensor) -> Dict[str, Any]:
        level1 = self._predict_binary(
            self.benign_l1, roi_tensor, BENIGN_LEVEL1_CLASS_NAMES
        )
        if level1["label"] == 0:  # Solid
            level2 = self._predict_binary(
                self.benign_l2_solid, roi_tensor, BENIGN_LEVEL2_SOLID_CLASS_NAMES
            )
            branch = "benign_solid"
        else:  # Cystic/Lipomatous
            level2 = self._predict_binary(
                self.benign_l2_cystic, roi_tensor, BENIGN_LEVEL2_CYSTIC_CLASS_NAMES
            )
            branch = "benign_cystic"

        return {
            "diagnosis_name": level2["name"],
            "diagnosis_label": level2["label"],
            "diagnosis_proba": {
                "level1": level1["proba"],
                "level2": level2["proba"],
            },
            "level1_name": level1["name"],
            "branch": branch,
        }

    def predict_diagnosis(
        self,
        roi_tensor: torch.Tensor,
        phenotype: str,
    ) -> Optional[Dict[str, Any]]:
        if phenotype == CLASS_NAMES[2]:  # Malignant
            return self.predict_malignant(roi_tensor)
        if phenotype == CLASS_NAMES[0]:  # Benign
            return self.predict_benign(roi_tensor)
        return None  # Indeterminate

    def get_model_info(self) -> dict:
        return {
            "num_models": len(self.models),
            "weights": self.model_weights.tolist() if self.model_weights is not None else None,
            "labels": self.labels,
            "diagnosis_loaded": self.diagnosis_loaded,
            "device": str(self.device),
        }
