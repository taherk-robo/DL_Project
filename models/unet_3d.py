# # models/unet_3d.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class UNet3D(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1, init_features=32):
#         super(UNet3D, self).__init__()
#         features = init_features

#         # Encoder
#         self.encoder1 = UNet3D._block(in_channels, features, name="enc1")
#         self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
#         self.encoder2 = UNet3D._block(features, features * 2, name="enc2")
#         self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
#         self.encoder3 = UNet3D._block(features * 2, features * 4, name="enc3")
#         self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
#         self.encoder4 = UNet3D._block(features * 4, features * 8, name="enc4")
#         self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

#         # Bottleneck
#         self.bottleneck = UNet3D._block(features * 8, features * 16, name="bottleneck")

#         # Decoder
#         self.upconv4 = nn.ConvTranspose3d(
#             features * 16, features * 8, kernel_size=2, stride=2)
#         self.decoder4 = UNet3D._block(
#             features * 16, features * 8, name="dec4")
#         self.upconv3 = nn.ConvTranspose3d(
#             features * 8, features * 4, kernel_size=2, stride=2)
#         self.decoder3 = UNet3D._block(
#             features * 8, features * 4, name="dec3")
#         self.upconv2 = nn.ConvTranspose3d(
#             features * 4, features * 2, kernel_size=2, stride=2)
#         self.decoder2 = UNet3D._block(
#             features * 4, features * 2, name="dec2")
#         self.upconv1 = nn.ConvTranspose3d(
#             features * 2, features, kernel_size=2, stride=2)
#         self.decoder1 = UNet3D._block(
#             features * 2, features, name="dec1")

#         self.conv = nn.Conv3d(
#             in_channels=features, out_channels=out_channels, kernel_size=1)

#     def forward(self, x):
#         # Encoder
#         enc1 = self.encoder1(x)
#         enc2 = self.encoder2(self.pool1(enc1))
#         enc3 = self.encoder3(self.pool2(enc2))
#         enc4 = self.encoder4(self.pool3(enc3))

#         # Bottleneck
#         bottleneck = self.bottleneck(self.pool4(enc4))

#         # Decoder
#         dec4 = self.upconv4(bottleneck)
#         dec4 = torch.cat((dec4, enc4), dim=1)
#         dec4 = self.decoder4(dec4)
#         dec3 = self.upconv3(dec4)
#         dec3 = torch.cat((dec3, enc3), dim=1)
#         dec3 = self.decoder3(dec3)
#         dec2 = self.upconv2(dec3)
#         dec2 = torch.cat((dec2, enc2), dim=1)
#         dec2 = self.decoder2(dec2)
#         dec1 = self.upconv1(dec2)
#         dec1 = torch.cat((dec1, enc1), dim=1)
#         dec1 = self.decoder1(dec1)

#         return self.conv(dec1)

#     @staticmethod
#     def _block(in_channels, features, name):
#         return nn.Sequential(
#             nn.Conv3d(
#                 in_channels=in_channels,
#                 out_channels=features,
#                 kernel_size=3,
#                 padding=1,
#                 bias=False),
#             nn.BatchNorm3d(num_features=features),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(
#                 in_channels=features,
#                 out_channels=features,
#                 kernel_size=3,
#                 padding=1,
#                 bias=False),
#             nn.BatchNorm3d(num_features=features),
#             nn.ReLU(inplace=True))


import torch
import torch.nn as nn
import torch.nn.functional as F

class PointCloudAutoencoder(nn.Module):
    def __init__(self):
        super(PointCloudAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Reconstruct x, y, z
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Example usage of PointCloudAutoencoder
if __name__ == "__main__":
    # Test with a random tensor input
    model = PointCloudAutoencoder()
    test_input = torch.randn((1, 3))  # Input shape: (batch_size, num_features)
    output = model(test_input)
    print(output)
