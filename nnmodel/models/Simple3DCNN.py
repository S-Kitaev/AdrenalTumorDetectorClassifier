import torch
import torch.nn as nn

class Simple3DCNN(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(Simple3DCNN, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=2),

            torch.nn.Conv3d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=2),

            torch.nn.Conv3d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=2),
        )

        with torch.no_grad():
            sample = torch.randn(1, 1, 53, 100, 100)
            features = self.features(sample)
            linear_input_size = features.view(features.size(0), -1).size(1)

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(linear_input_size, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x