"""Create MedicalNet-style diagnosis stubs (random init, optional MedicalNet backbone).

For real MedicalNet weights put resnet_18_23dataset.pth into model_files/, then:
  python scripts/create_diagnosis_stubs.py --with-medicalnet
"""

from pathlib import Path
import argparse

import torch

from nnmodel.models.MedicalNetResNet import build_medicalnet_resnet18
from nnmodel.settings import (
    BENIGN_LEVEL1_PATH,
    BENIGN_LEVEL2_CYSTIC_PATH,
    BENIGN_LEVEL2_SOLID_PATH,
    MALIGNANT_MODEL_PATH,
    MEDICALNET_WEIGHTS_PATH,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-medicalnet",
        action="store_true",
        help="Initialize stubs from MEDICALNET_WEIGHTS_PATH if file exists",
    )
    args = parser.parse_args()

    medicalnet = None
    if args.with_medicalnet and Path(MEDICALNET_WEIGHTS_PATH).exists():
        medicalnet = MEDICALNET_WEIGHTS_PATH
        print(f"Using MedicalNet backbone: {medicalnet}")
    elif args.with_medicalnet:
        print(f"MedicalNet file missing: {MEDICALNET_WEIGHTS_PATH}")

    for path in (
        MALIGNANT_MODEL_PATH,
        BENIGN_LEVEL1_PATH,
        BENIGN_LEVEL2_SOLID_PATH,
        BENIGN_LEVEL2_CYSTIC_PATH,
    ):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        model = build_medicalnet_resnet18(num_classes=2, medicalnet_path=medicalnet)
        torch.save(model.state_dict(), p)
        print(f"Saved stub: {p} ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
