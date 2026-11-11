"""
MedicalNet-compatible 3D ResNet-18 for classification.

Architecture follows Tencent MedicalNet / Med3D (resnet_18, shortcut A),
with AdaptiveAvgPool3d + Linear head instead of the segmentation head.
Pretrained weights: resnet_18_23dataset.pth (or resnet_18.pth).
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False,
    )


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.zeros(
        out.size(0),
        planes - out.size(1),
        out.size(2),
        out.size(3),
        out.size(4),
        device=out.device,
        dtype=out.dtype,
    )
    return torch.cat([out, zero_pads], dim=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class MedicalNetResNet18(nn.Module):
    """3D ResNet-18 (MedicalNet backbone) + classification head."""

    def __init__(self, num_classes: int = 2, shortcut_type: str = "A"):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(
            1, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, shortcut_type)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            BasicBlock, 256, 2, shortcut_type, stride=1, dilation=2
        )
        self.layer4 = self._make_layer(
            BasicBlock, 512, 2, shortcut_type, stride=1, dilation=4
        )
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = [
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (N, 1, D, H, W)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def _normalize_state_dict(raw) -> dict:
    """Unwrap MedicalNet / DataParallel / training checkpoints to a flat state_dict."""
    if isinstance(raw, dict):
        if "state_dict" in raw and isinstance(raw["state_dict"], dict):
            raw = raw["state_dict"]
        elif "model_state_dict" in raw and isinstance(raw["model_state_dict"], dict):
            raw = raw["model_state_dict"]

    if not isinstance(raw, dict):
        raise ValueError(f"Unsupported checkpoint type: {type(raw)}")

    out = {}
    for k, v in raw.items():
        key = k
        if key.startswith("module."):
            key = key[len("module.") :]
        # MedicalNet segmentation head is unused for classification
        if key.startswith("conv_seg"):
            continue
        out[key] = v
    return out


def load_medicalnet_backbone(
    model: MedicalNetResNet18,
    weights_path: str,
    map_location="cpu",
) -> int:
    """
    Load Tencent MedicalNet pretrained weights into the backbone.

    Returns number of parameter tensors successfully copied.
    """
    try:
        ckpt = torch.load(weights_path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(weights_path, map_location=map_location)

    pretrain = _normalize_state_dict(ckpt)
    model_dict = model.state_dict()
    filtered = {
        k: v
        for k, v in pretrain.items()
        if k in model_dict and tuple(v.shape) == tuple(model_dict[k].shape)
    }
    model_dict.update(filtered)
    model.load_state_dict(model_dict)
    return len(filtered)


def build_medicalnet_resnet18(
    num_classes: int,
    medicalnet_path: Optional[str] = None,
    map_location="cpu",
) -> MedicalNetResNet18:
    model = MedicalNetResNet18(num_classes=num_classes, shortcut_type="A")
    if medicalnet_path:
        n = load_medicalnet_backbone(model, medicalnet_path, map_location=map_location)
        print(f"[MedicalNet] Loaded {n} tensors from {medicalnet_path}")
    return model
