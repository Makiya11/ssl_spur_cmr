
from torchvision import transforms
import torchvision.transforms.functional as TF
import random

class EDESTrainTransforms2D:
    def __init__(self, transform_params):
        # Resize
        self.resize = transforms.Resize(size=(int(transform_params['side_size']),
                                              int(transform_params['side_size'])))
        self.transform_params = transform_params
        self.toPIL = transforms.ToPILImage(mode='L')
        self.toTen = transforms.ToTensor()
        self.norm = transforms.Normalize(mean=(0.5), std=(0.5))

    def transform(self, es, ed):
        
        es = self.toPIL(es)
        ed = self.toPIL(ed)

        # resize
        es = self.resize(es)
        ed = self.resize(ed)
                        
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            es, output_size=(self.transform_params['side_size'], 
                             self.transform_params['side_size']))
        
        es = TF.crop(es, i, j, h, w)
        ed = TF.crop(ed, i, j, h, w)
        
        # Random horizontal flipping
        if random.random() > 0.5:
            es = TF.hflip(es)
            ed = TF.hflip(ed)

        # Random vertical flipping
        if random.random() > 0.5:
            es = TF.vflip(es)
            ed = TF.vflip(ed)
        
        # Random rotation
        rand_degree = random.randint(-20, 20)
        es = TF.rotate(img=es, angle=rand_degree)
        ed = TF.rotate(img=ed, angle=rand_degree)

        es = self.toTen(es)
        ed = self.toTen(ed)

        es = self.norm(es)
        ed = self.norm(ed)
        return es, ed
    
    def __call__(self, es, ed):
        return self.transform(es, ed)

class EDESTestTransforms2D:
    def __init__(self, transform_params):
        # Resize
        self.resize = transforms.Resize(size=(transform_params['side_size'],
                                              transform_params['side_size']))
        self.toPIL = transforms.ToPILImage(mode='L')
        self.toTen = transforms.ToTensor()
        self.norm = transforms.Normalize(mean=(0.5), std=(0.5))

    def transform(self, es, ed):
        
        es = self.toPIL(es)
        ed = self.toPIL(ed)
        
        # resize
        es = self.resize(es)        
        ed = self.resize(ed)

        es = self.toTen(es)
        ed = self.toTen(ed)

        es = self.norm(es)
        ed = self.norm(ed)
        return es, ed
    
    def __call__(self, es, ed):
        return self.transform(es, ed)
    
    
    
class TrainTransforms2D:
    def __init__(self, transform_params):
        # Resize
        self.resize = transforms.Resize(size=(int(transform_params['side_size']),
                                              int(transform_params['side_size'])))
        self.transform_params = transform_params
        self.toPIL = transforms.ToPILImage(mode='L')
        self.toTen = transforms.ToTensor()
        self.norm = transforms.Normalize(mean=(0.5), std=(0.5))

    def transform(self, x):
        
        x = self.toPIL(x)
        x = self.resize(x)   
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            x, output_size=(self.transform_params['side_size'], 
                             self.transform_params['side_size']))
        x = TF.crop(x, i, j, h, w)
        
        # Random horizontal flipping
        if random.random() > 0.5:
            x = TF.hflip(x)

        # Random vertical flipping
        if random.random() > 0.5:
            x = TF.vflip(x)
        
        rand_degree = random.randint(-20, 20)
        x = TF.rotate(img=x, angle=rand_degree)
        
        x = self.toTen(x)
        x = self.norm(x)
        return x
    
    def __call__(self, x):
        return self.transform(x)

class TestTransforms2D:
    def __init__(self, transform_params):
        self.resize = transforms.Resize(size=(transform_params['side_size'],
                                              transform_params['side_size']))
        self.toPIL = transforms.ToPILImage(mode='L')
        self.toTen = transforms.ToTensor()
        self.norm = transforms.Normalize(mean=(0.5), std=(0.5))

    def transform(self, x):
        x = self.toPIL(x)
        x = self.resize(x)        
        x = self.toTen(x)
        x = self.norm(x)
        return x
    
    def __call__(self, x):
        return self.transform(x)