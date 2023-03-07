from torchvision import transforms as T
from torchvision.transforms.transforms import Normalize, ToTensor
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

def train_transformer(size=(224,224)):
    return T.Compose([
        T.Resize((224,224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize
    ])

def test_transformer(size=(224,224)):
    return T.Compose([
        T.Resize(size),
        T.ToTensor(),
        normalize
    ])

