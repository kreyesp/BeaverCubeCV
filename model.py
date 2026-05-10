import torch
import torch.nn as nn


class SmallCloudNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=4):
        super().__init__()

        def block(cin, cout):
            # return nn.Sequential(
            #     nn.Conv2d(cin, cout, 3, padding=1),
            #     nn.GroupNorm(8, cout),
            #     nn.ReLU(),
            #     nn.Conv2d(cout, cout, 3, padding=1),
            #     nn.GroupNorm(8, cout),
            #     nn.ReLU(),
            # )
            return nn.Sequential(
                    nn.Conv2d(cin, cout, 3, padding=1),
                    nn.GroupNorm(16, cout),   # bumped from 8 to 16 groups
                    nn.ReLU(),
                    nn.Conv2d(cout, cout, 3, padding=1),
                    nn.GroupNorm(16, cout),
                    nn.ReLU(),
                )

        # self.enc1 = block(in_ch, 32)
        # self.enc2 = block(32, 64)
        # self.enc3 = block(64, 128)
        # self.pool = nn.MaxPool2d(2)

        # self.up2  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        # self.dec2 = block(128, 64)

        # self.up1  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        # self.dec1 = block(64, 32)

        # self.head = nn.Conv2d(32, num_classes, 1)
        self.enc1 = block(in_ch, 64)
        self.enc2 = block(64, 128)
        self.enc3 = block(128, 256)
        self.pool = nn.MaxPool2d(2)       


        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = block(256, 128)

        self.up1  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = block(128, 64)

        self.head = nn.Conv2d(64, num_classes, 1)

    @staticmethod
    def _match(up, skip):
        """Crop skip connection to match upsampled tensor if sizes differ."""
        if up.shape != skip.shape:
            skip = skip[:, :, :up.shape[2], :up.shape[3]]
        return skip

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        u2 = self.up2(e3)
        d2 = self.dec2(torch.cat([u2, self._match(u2, e2)], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, self._match(u1, e1)], dim=1))

        return self.head(d1)