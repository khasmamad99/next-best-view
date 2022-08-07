import torch
import torch.nn as nn


class VoxNet(nn.Module):
    """From "VoxNet: A 3D Convolutional Neural Network for Real-Time Object Recognition,"
    https://www.ri.cmu.edu/pub_files/2015/9/voxnet_maturana_scherer_iros15.pdf.
    """

    def __init__(self, num_classes):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=5, stride=2),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout()
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=6912, out_features=128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=128, out_features=num_classes)
        )


    def forward(self, x):
        """
        x : torch.Tensor, (B, 1, 32, 32, 32)

        Returns: torch.Tensor, (B, num_classes)
        """
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x
        