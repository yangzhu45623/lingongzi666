import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from pytorch_grad_cam import (GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM)
from run.trainer import BaseTrainer
from utils.util import get_patient2class
from utils.util import parse_args
from datasets.dataset import get_dataset
from datasets.transformer import test_transformer


class CamTrainer(BaseTrainer):
    def __init__(self, args, train_dataset, val_dataset):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.best_acc1 = 0

        self.build_model()

    def cam(self,dset_mode,target_layer,id_to_class):
        if dset_mode == 'train':
            dataset = self.train_dataset
        else:
            dataset = self.val_dataset
            
        test_model_path = self.args.test_model_path.format(self.args.train_type,self.args.model)
        self.load_model(test_model_path)                  # 加载模型
        self.save_dir,test_model = os.path.split(test_model_path)
        test_model = os.path.splitext(test_model)[0]
        
        self.model.eval()
        self.model.to(self.device)

        cam = GradCAM(model=self.model,target_layer=target_layer,use_cuda=torch.cuda.is_available(),reshape_transform=None)
        for input_tensor,_,path in dataset:
            print(path)
            src = Image.open(path, "r").convert("RGB")          # 读取原始的数据
            patient,gt_class = get_patient2class(path)  # 这个是这个图片对应的类名

            input_tensor = input_tensor.unsqueeze(dim=0)
            output = self.model(input_tensor.to(self.device))
            pred_label = torch.argmax(output, dim=-1).item()
            pred_class = id_to_class[pred_label]

            grayscale_cam = cam(input_tensor=input_tensor,
                                target_category=None,
                                eigen_smooth=True,
                                aug_smooth=True)

            if np.alltrue(np.isnan(grayscale_cam)):     # 去掉画grad-cam图有问题的样本（出现nan值）
                continue
            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0, :]
            grayscale_cam = np.array(Image.fromarray(grayscale_cam).resize(src.size))

            fig, ax = plt.subplots(1, 2)
            plt.rcParams['figure.figsize'] = (16, 8)  # 设置figure_size尺寸

            ax[0].imshow(np.array(src))
            ax[0].set_title("src image, gt_label:{}".format(gt_class))

            ax[1].imshow(np.array(src))
            ax[1].imshow(grayscale_cam, alpha=0.5)
            ax[1].set_title("heatmap, pred class:{}".format(pred_class))
            _dir = os.path.join(self.args.datadir+'_cam',self.args.model,gt_class,patient)
            os.makedirs(_dir,exist_ok=True)
            plt.savefig(os.path.join(_dir,os.path.basename(path)))
            plt.close()


if __name__ == '__main__':

    methods = {"gradcam": GradCAM,
            "scorecam": ScoreCAM,
            "gradcam++": GradCAMPlusPlus,
            "ablationcam": AblationCAM,
            "xgradcam": XGradCAM,
            "eigencam": EigenCAM,
            }
    
    target_layers = {
        'efficientnet_b0':'conv_head',
        'inception_resnet_v2':'conv2d_7b',
        'resnet50':'layer4.2',
        'tf_efficientnet_b0':'conv_head',
        'xception':'act4'
    }
    
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    val_transform = test_transformer()
    Dataset = get_dataset(args.train_type,True)
    train_dataset = Dataset(os.path.join(args.datadir,'train'),img_num=args.img_num,transformer=val_transform,data_type=args.data_type,args=args)
    val_dataset = Dataset(os.path.join(args.datadir,'test'),img_num=args.img_num,transformer=val_transform,data_type=args.data_type,args=args)

    id_to_class = {}
    for id,name in enumerate(train_dataset.class_names):
        id_to_class[id] = name

    trainer = CamTrainer(args,train_dataset,val_dataset)
    trainer.cam('train',target_layers[args.model],id_to_class)
    trainer.cam('val',target_layers[args.model],id_to_class)

    
    