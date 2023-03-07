import matplotlib.pyplot as plt
import os
import json

acc_path = {

    "Efficientnet_b0": r"C:\Users\Administrator\Desktop\classification\output\waterT1C-post\image\efficientnet_b0\acc_efficientnet_b0.json",
    "Inception_resnet_v2": r"C:\Users\Administrator\Desktop\classification\output\waterT1C-post\image\inception_resnet_v2\acc_inception_resnet_v2.json",
    "Resnet50": r"C:\Users\Administrator\Desktop\classification\output\waterT1C-post\image\resnet50\acc_resnet50.json",
    "Xception": r"C:\Users\Administrator\Desktop\classification\output\waterT1C-post\image\xception\acc_xception.json"
}


acc_type = "patient_val"                    # 选择image_val或者patient_val，有些文件中可能只有image_val或者只有patient_val，使用之前，进对应的json文件看看
save_dir = r"C:\Users\Administrator\Desktop\classification\output\waterT1C-post\image"  # 生成的文件保存的路径
linewidth = 1                                   # 线的粗细
title = 'Patient'                                 # 图的标题
xlabel,xlabel_size = "Epochs",15                # x轴的标签和大小
ylabel,ylabel_size = "Accuracy",15                # y轴的标签和大小

def load_data(path):
    with open(path,'r') as f:
        datas = json.load(f)[acc_type]
    return datas

def plot(datas,save_dir):
    plt.figure(figsize=(10,5),dpi=300)           # figsize用来设置图的比例
    for key,value in datas.items():
        plt.plot(list(range(len(value))),value,linewidth=linewidth,label=key)
    plt.legend(loc='lower right')               # 这是设置标注的位置，可以选择 "lower right"，"upper right","upper left","lower right"
    plt.xlabel(xlabel, size=xlabel_size)
    plt.ylabel(ylabel, size=ylabel_size)
    plt.title(title)
    plt.savefig(os.path.join(save_dir,'%s.png'%acc_type), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir,'%s.pdf'%acc_type), bbox_inches='tight')

datas = {}
for key in acc_path:
    datas[key] = load_data(acc_path[key])

plot(datas,save_dir)
