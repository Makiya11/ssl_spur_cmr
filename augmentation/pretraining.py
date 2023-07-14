import numpy as np
from torchvision import transforms 

class SSLTrainTransform2D(object):
    def __init__(self, transform_params): 
        self.transform =  transforms.Compose(
            [
                transforms.ToPILImage(mode='L'),
                transforms.RandomResizedCrop(size=transform_params["side_size"]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0, contrast=[0.7, 1.3], saturation=0, hue=0),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5)),
            ]
        )

    def __call__(self, x):
        xi = self.transform(x)
        xj = self.transform(x)
        return xi, xj


class SSLTestTransform2D(object):
    def __init__(self, transform_params): 
       
        self.transform =  transforms.Compose(
            [
                transforms.ToPILImage(mode='L'),
                transforms.Resize(size=transform_params["side_size"]),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5)),
            ]
        )

    def __call__(self, x):
        xi = self.transform(x)
        xj = self.transform(x)
        return xi, xj
