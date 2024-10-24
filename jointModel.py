import torch
import torch.nn as nn


class LowRankLinear(nn.Module):
    def __init__(self, dim, rank):
        super().__init__()
        self.lowRankLinear = nn.Linear(dim, rank)

    def forward(self, x):
        return self.lowRankLinear(x)


class Classifier(nn.Module):
    def __init__(self, rank, n_class=14):
        super(Classifier, self).__init__()

        self.dense_1 = nn.Linear(rank, n_class, bias=True)

    def forward(self, x):
        x = self.dense_1(x)
        return x


class JointLowRankModel(nn.Module):
    def __init__(self, base, linearLayer, classifier, trainLinearLayer=True, train_all=True):
        super(JointLowRankModel, self).__init__()
        self.base = base
        self.linearLayer = linearLayer
        self.classifier = classifier

        if not train_all:
            for param in self.base.parameters():
                param.requires_grad = False

        if not trainLinearLayer:
            for param in self.linearLayer.parameters():
                param.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def forward(self, x):
        x = self.base(x)
        x = self.linearLayer(x)
        x = self.classifier(x)
        return x
