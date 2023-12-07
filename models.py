import torch
import torchvision
import torch.nn as nn


model_config = {
        "inception_v3": {
            "feature_type": "image",
            "target_resolution": (299, 299),
            "flatten": False,
            },
        
        "wideresnet50": {
            "feature_type": "image",
            "target_resolution": (224, 224),
            "flatten": False,
            },
        
        "resnet50": {
            "feature_type": "image",
            "target_resolution": (224, 224),
            "flatten": False,
            },
        
        "resnet34": {
            "feature_type": "image",
            "target_resolution": None,
            "flatten": False,
            },

        "vgg": {
            "feature_type": "image",
            "target_resolution": (178, 178),
            "flatten": False,
            },
        }



class VGG(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG, self).__init__()
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M', 1024, 1024, 1024, 'M']
        self.features = self._make_layers()
        self.classifier = nn.Linear(4096, num_classes)
        
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self):
        layers = []
        in_channels = 3
        for x in self.cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



def get_model(model, n_classes):
    if model == "resnet50":
        network = torchvision.models.resnet50(weights=None)
        d = network.fc.in_features
        network.fc = nn.Linear(d, n_classes)
    elif model == "resnet34":
        network = torchvision.models.resnet34(weights=None)
        d = network.fc.in_features
        network.fc = nn.Linear(d, n_classes)
    elif model == "wideresnet50":
        network = torchvision.models.wide_resnet50_2(weights=None)
        d = network.fc.in_features
        network.fc = nn.Linear(d, n_classes)
    elif model == "vgg":
        network = VGG()
    else:
        raise NotImplementedError
    return network

