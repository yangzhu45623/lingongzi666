import argparse
import os
import json
import timm
import numpy as np
from PIL import Image
import yaml
from models import densenet3d,resnet3d,resnext3d

def load_config(path):
    with open(path,'r',encoding='utf-8') as f:
        config = yaml.load(f,yaml.FullLoader)
    return config

def merge_config2args(args,config):
    for k,v in config.items():
        if isinstance(v,dict):
            merge_config2args(args,v)
        else:
            setattr(args,k,v)

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')                    # 是train还是evaluate

    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')                    # 设置种子

    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')                                      # 设置GPU

    parser.add_argument('-cp','--configpath', default=r'./config.yaml', type=str,
                        help='config path.') 
    
    args =  parser.parse_args()
    config = load_config(args.configpath)
    merge_config2args(args,config)
    
    if args.train_type == 'patient':                                                # 如果以病人为单位
        args.data_type = '3d' if args.model[-2:] == '3d' else 'cat'                 # 如果模型使用的是3d模型，则数据集加载的数据类型一定是3d的，否则为cat的
    else:                                                                           # 以图片为单位
        args.img_num = 1
        args.data_type = None
    args.outdir = os.path.join(args.outdir,args.train_type,args.model) # 保存log的路径

    return args

def create_models(args):
    if args.model== 'resnet3d':
        model = resnet3d.generate_model(
            50,
            n_input_channels=3,
            n_classes=args.num_classes
        )
    elif args.model == 'densenet3d':
        model = densenet3d.generate_model(
            169,
            n_input_channels=3,
            num_classes=args.num_classes
        )
    elif args.model == 'resnext3d':
        resnext3d.generate_model(
            50,
            n_input_channels=3,
            n_classes=args.num_classes
        )
    else:
        model = timm.create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            drop_rate=args.drop,
            drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
            drop_path_rate=args.drop_path,
            drop_block_rate=args.drop_block,
            global_pool=args.gp,
            bn_tf=args.bn_tf,
            bn_momentum=args.bn_momentum,
            bn_eps=args.bn_eps,
            scriptable=args.torchscript,
            checkpoint_path='',
            in_chans = args.img_num*3)              # 出入数据的通道，一张图片是rgb 3个通道，这里就是args.img_num * 3
    return model

def load_img(path):
    return Image.open(path).convert('RGB')

def save_recorder(path,data):
    with open(path+'.json','w') as f:
        json.dump(data,f)

def load_recorder(path):
    with open(path,'r') as f:
        content = json.load(f)
    return content

def get_patient2class(path):
    path = os.path.split(path)[0]
    _class,patient = os.path.split(path)
    return patient,os.path.basename(_class)