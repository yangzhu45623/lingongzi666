import os,glob
from PIL.Image import Image
import torch
from utils.util import load_img
from torch.utils.data import Dataset
from datasets.transformer import test_transformer
class PatientDataset(Dataset):
    def __init__(self,root,img_num,transformer=None,pad_type='zero',data_type='cat',args=None):
        super(PatientDataset).__init__()
        self.root = root
        self.transformer = transformer
        self.img_num = img_num//args.step
        self.pad_type = pad_type
        self.data_type = data_type
        self.args = args

        if self.transformer == None:
            self.transformer = test_transformer()

        self.static_data()
    
    def __getitem__(self, index):
        patient_dir = self.datas[index]
        images = []
        paths = list(os.listdir(patient_dir))
        paths.sort(key = lambda x:float(float(x.split('-')[-1][:-4])))
        for i,name in enumerate(list(os.listdir(patient_dir)[::self.args.step])):
            if i == self.img_num:
                break
            path = os.path.join(patient_dir,name)
            img = load_img(path)
            img = self.transformer(img)
            images.append(img)
        for _ in range(len(images),self.img_num):
            if self.pad_type == 'zero':
                images.append(torch.zeros_like(images[0]))
            elif self.pad_type == 'gauss':
                images.append(torch.randn_like(images[0]))
            else:
                raise Exception("Please input right pad_type")

        if self.data_type == '3d':                                  # 如果数据的类型是3d就返回3d的数据类型
            return torch.stack(images,dim=1),self.labels[index]
        else:                                                       # 否则返回concatenate的方法
            return torch.cat(images,dim=0),self.labels[index]
        
    
    def __len__(self) -> int:
        return len(self.datas)
    
    def static_data(self):
        self.datas = []
        self.labels = []
        self.class_names = list(os.listdir(self.root))
        self.class_names.sort()
        for i,class_name in enumerate(self.class_names):
            patient_names = os.listdir(os.path.join(self.root,class_name))
            for patient_name in patient_names:
                self.datas.append(os.path.join(self.root,class_name,patient_name))
                self.labels.append(i)

class PatientDatasetPath(PatientDataset):
    def __init__(self, root, img_num, transformer=None, pad_type='zero', data_type='cat',args=None):
        super().__init__(root, img_num, transformer=transformer, pad_type=pad_type, data_type=data_type,args=args)
    
    def __getitem__(self, index):
        image,label = super().__getitem__(index)
        return image,label,self.datas[index]

class ImageDataset(Dataset):
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm',
                      '.tif', '.tiff', '.webp', '.JPEG')
    def __init__(self,root,transformer=None,*args,**kwags):
        super(ImageDataset).__init__()
        self.root = root
        self.transformer = transformer
        if self.transformer == None:
            self.transformer = test_transformer()

        self.static_data()
    def __getitem__(self, index):
        path = self.datas[index]
        img = load_img(path)
        img = self.transformer(img)

        return img,self.labels[index]
    
    def __len__(self) -> int:
        return len(self.datas)
    
    def static_data(self):
        self.datas = []
        self.labels = []
        self.class_names = list(os.listdir(self.root))
        self.class_names.sort()
        for i,class_name in enumerate(self.class_names):
            images = glob.glob(os.path.join(self.root,class_name,"**",'*.png'),recursive=True)
            images = list(filter(lambda x: os.path.splitext(x)[-1].lower() in self.IMG_EXTENSIONS,images))
            self.datas.extend(images)
            self.labels.extend([i]*len(images))
    
class ImageDatasetPath(ImageDataset):
    def __init__(self, root, transformer=None, *args, **kwags):
        super().__init__(root, transformer=transformer, *args, **kwags)
    
    def __getitem__(self, index):
        img,label = super().__getitem__(index)
        return img,label,self.datas[index]

def get_dataset(train_type,with_path=None):
    if with_path:
        if train_type == 'patient':
            return PatientDatasetPath
        else:
            return ImageDatasetPath
    else:
        if train_type == 'patient':
            return PatientDataset
        elif train_type == 'image':
            return ImageDataset