import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, arch, weight_path, num_classes, feature_extract=False):
        super(CNN, self).__init__()
        self.arch = arch
        self.num_classes = num_classes
        self.feature_extract = feature_extract
        self.conv0 = nn.Conv2d(1, 3, 1)
        self.encoder = self.get_encoder()
    
    def get_encoder(self):
        if "VGG16" in self.arch:
            encoder = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
            self.set_parameter_requires_grad(encoder)
            num_ftrs = encoder.classifier[6].in_features
            encoder.classifier[6] = nn.Linear(num_ftrs, self.num_classes)
        elif "DenseNet121" in self.arch:
            encoder = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
            self.set_parameter_requires_grad(encoder)
            num_ftrs = encoder.classifier.in_features
            encoder.classifier = nn.Linear(num_ftrs, self.num_classes)
        else:
            raise Exception('Check DL model')
        
        return encoder

    def set_parameter_requires_grad(self, model):
        """Freeze all layers"""
        if self.feature_extract:
            for param in model.parameters():
                param.requires_grad = False
       
    def forward(self, i):
        out = self.conv0(i)
        out = self.encoder(out)
        return out 


class EDES_CNN(nn.Module):
    def __init__(self, arch, weight_path, num_classes, feature_extract=False):
        super(EDES_CNN, self).__init__()
        self.arch = arch
        self.weight_path = weight_path
        self.num_classes = num_classes
        self.feature_extract = feature_extract
        self.conv0 = nn.Conv2d(1, 3, 1)
        self.encoder = self.get_encoder()
        self.classifier = nn.Sequential(
            nn.Linear(2048*2, num_classes),
            nn.Softmax(dim=1)
        )
        
    def get_encoder(self):
        if "VGG16" in self.arch:
            if 'imagenet' == self.weight_path:
                encoder = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
            else:
                encoder = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=False)
            num_ftrs = encoder.classifier[6].in_features
            encoder.classifier[6] = nn.Linear(num_ftrs, 2048)
            self.set_parameter_requires_grad(encoder)

        elif "DenseNet121" in self.arch:
            if 'imagenet' == self.weight_path:
                encoder = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
            else:
                encoder = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=False)    
            num_ftrs = encoder.classifier.in_features
            encoder.classifier = nn.Linear(num_ftrs, 2048)
            self.set_parameter_requires_grad(encoder)
            
        else:
            raise Exception('Check DL model')    
        
        return encoder

    def set_parameter_requires_grad(self, model):
        """Freeze all layers"""
        if self.feature_extract:
            for param in model.parameters():
                param.requires_grad = False
       
    def forward(self, i):
        out1 = self.conv0(i[:,:,0])
        out1 = self.encoder(out1)

        out2 = self.conv0(i[:,:,1])
        out2 = self.encoder(out2)

        combined = torch.cat((out1, out2), dim=1)
        out = self.classifier(combined)
        return out 

if __name__ == "__main__":
    model = EDES_CNN('DenseNet121', 'imagenet', 2, False)

    


    