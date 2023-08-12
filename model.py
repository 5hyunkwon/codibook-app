import torch
import torch.nn as nn
import torchvision


class CodiClassifier:
    def __init__(self, args):
        self.tv_ver = torchvision.__version__
        self.model_name = args['model_name']
        self.num_classes = args['num_classes']
        self.model = self._load_model()

    def _load_model(self):
        base_model = torch.hub.load(f'pytorch/vision:v{self.tv_ver}', self.model_name, pretrained=True)
        num_features = base_model.fc.in_features
        base_model.fc = nn.Linear(num_features, self.num_classes)
        return base_model

    def get_model(self):
        return self.model
