import timm
import torch.nn as nn


class Moding_dif_models(nn.Module):
    def __init__(self, model_name, embed_dim, num_class):
        super(Moding_dif_models, self).__init__()
        print(model_name)
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, x):
        features = self.model(x)
        logits = self.fc(features)
        return logits
