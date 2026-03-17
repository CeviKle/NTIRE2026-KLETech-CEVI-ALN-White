import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import vgg16


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()

        vgg = vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        features = list(vgg.features)[:23]

        self.features = nn.ModuleList(features).eval()

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        results = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in {3, 8, 15, 22}:
                results.append(x)
        return results


class PerceptualLossModule:
    def __init__(self, device):
        self.model = Vgg16().to(device)
        self.criterion = nn.L1Loss().to(device)

    def compute_content_loss(self, pred, target):
        pred_feats = self.model(pred)
        target_feats = self.model(target)

        loss = 0
        for pf, tf in zip(pred_feats, target_feats):
            loss += self.criterion(pf, tf.detach())

        return loss / len(pred_feats)
