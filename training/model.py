import torch
import torch.nn as nn

# Parameters
CNN_W = 16
FEAT_EXT_W = 48
FEAT_EXT_W2 = 48
USE_IMAGES = True
USE_FEATURES = True


class Model(nn.Module):
    def __init__(self, p_dropout=0):
        super(Model, self).__init__()

        self.cnn_layers = nn.Sequential(

            nn.Conv2d(1, CNN_W, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_dropout),
            nn.Conv2d(CNN_W, CNN_W, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(CNN_W, CNN_W * 2, kernel_size=3, stride=2, padding=1),
            nn.Dropout(p=p_dropout),

            nn.Conv2d(CNN_W * 2, CNN_W * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_dropout),
            nn.Conv2d(CNN_W * 2, CNN_W * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(CNN_W * 2, CNN_W * 4, kernel_size=3, stride=2, padding=1),
            nn.Dropout(p=p_dropout),

            nn.Conv2d(CNN_W * 4, CNN_W * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_dropout),
            nn.Conv2d(CNN_W * 4, CNN_W * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(CNN_W * 4, CNN_W * 8, kernel_size=3, stride=2, padding=1),
            nn.Dropout(p=p_dropout),
        )

        self.feature_extractor = nn.Sequential(
            nn.Linear(CNN_W * 8 * 8 * 8, FEAT_EXT_W)
        )

        self.classifier_images_and_features = nn.Sequential(
            nn.Dropout(p=p_dropout),
            nn.Linear(FEAT_EXT_W + 24, FEAT_EXT_W2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_dropout),
            nn.Linear(FEAT_EXT_W2, 1)
        )

        self.classifier_images = nn.Sequential(
            nn.Dropout(p=p_dropout),
            nn.Linear(FEAT_EXT_W, 1)
        )

        self.classifier_features = nn.Sequential(
            nn.Linear(24, 12),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_dropout),
            nn.Linear(12, 1)
        )

    # Defining the forward pass
    def forward(self, x, y):
        if USE_IMAGES:
            x = self.cnn_layers(x)
            x = torch.flatten(x, 1)
            x = self.feature_extractor(x)
            if USE_FEATURES:
                z = torch.cat([x, y], 1)
                return self.classifier_images_and_features(z)
            else:
                return self.classifier_images(x)
        elif USE_FEATURES:
            return self.classifier_features(y)
