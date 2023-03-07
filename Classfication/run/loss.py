import matplotlib.pyplot as plt
import os
import json

loss_path = {

    "Efficientnet_b0": r"C:\Users\Administrator\Desktop\classification\output\waterT1C-post\image\efficientnet_b0\val-model_best-patient-cm.json",
    "Inception_resnet_v2": r"C:\Users\Administrator\Desktop\classification\output\waterT1C-post\image\inception_resnet_v2\val-model_best-patient-cm.json",
    "Resnet50": r"C:\Users\Administrator\Desktop\classification\output\waterT1C-post\image\resnet50\val-model_best-patient-cm.json",
    "Xception": r"C:\Users\Administrator\Desktop\classification\output\waterT1C-post\image\xception\val-model_best-patient-cm.json"
}


loss_type = "train"                  # 选择train或者val,分别表示训练集和测试集上的loss
save_dir = r"C:\Users\Administrator\Desktop\classification\output\task1\IC3\image"      # 生成的文件保存的路径
linewidth = 1                                   # 线的粗细
title = 'train loss'                                 # 图的标题
xlabel,xlabel_size = "Epochs",15                # x轴的标签和大小
ylabel,ylabel_size = "loss",15                # y轴的标签和大小

def load_data(path):
    with open(path,'r') as f:
        datas = json.load(f)[loss_type]
    return datas

def plot(datas,save_dir):
    plt.figure(figsize=(8,5),dpi=300)           # figsize用来设置图的比例
    for key,value in datas.items():
        plt.plot(list(range(len(value))),value,linewidth=linewidth,label=key)
    plt.legend(loc='upper right')               # 这是设置标注的位置，可以选择 "lower right"，"upper right","upper left","lower right"
    plt.xlabel(xlabel, size=xlabel_size)
    plt.ylabel(ylabel, size=ylabel_size)
    plt.title(title)
    plt.savefig(os.path.join(save_dir,'%s.png'%loss_type), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir,'%s.pdf'%loss_type), bbox_inches='tight')

datas = {}
for key in loss_path:
    datas[key] = load_data(loss_path[key])

plot(datas,save_dir)
