import os
import json
import matplotlib.pyplot as plt

# 在paths中将想花在一起的dice或者acc路径输进去
paths = {
    'UNet':r"C:\Users\Administrator\Desktop\Pytorch-UNet-master\UNet_task1_checkpoints\dice_acc.json",                                                     #
    'deeplabv3_resnet50':r'C:\Users\Administrator\Desktop\Pytorch-UNet-master\deeplabv3_resnet50_task1_checkpoints\dice_acc.json'
}

data_type = "acc"              # 选择画dice或者画acc的曲线图
save_dir = "./"               # 生成的图保存的路径
name = 'acc1'                   # 生成的图保存的名字

def load_data(path):
    with open(path,'r') as f:
        return json.load(f)[data_type]

def draw():
    datas = {}
    for key in paths:
        datas[key] = load_data(paths[key])
    
    plt.figure(figsize=(10,5),dpi=100)                                              # 通过调10和5来调图的长宽比例
    length = 0
    for key,value in datas.items():
        plt.plot(list(range(len(value))),value,linewidth=2,label=key)               # 可以通过调linewidth来调线的宽度
        length = len(value)
    plt.xticks(range(length))
    plt.legend(loc='lower right')
    os.makedirs(save_dir,exist_ok=True)
    plt.savefig(os.path.join(save_dir,name+'.png'),bbox_inches='tight')
    plt.savefig(os.path.join(save_dir,name+'.pdf'),bbox_inches='tight')
    plt.close()

draw()