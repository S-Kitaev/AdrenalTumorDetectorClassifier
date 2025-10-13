import torch
import torch.nn as nn

class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(Simple3DCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
        )

        # Автоматический расчет размера входа для линейного слоя
        with torch.no_grad():
            sample = torch.randn(1, 1, 54, 224, 224)
            features = self.features(sample)
            linear_input_size = features.view(features.size(0), -1).size(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x