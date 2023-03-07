import numpy as np
import matplotlib.pyplot as plt
import torch
import json
from sklearn.metrics import roc_curve, auc

def get_roc(score_list,label_list,num_class):
    score_array = np.array(score_list)
    # 将label转换成onehot形式
    label_tensor = torch.tensor(label_list)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], num_class)
    label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    label_onehot = np.array(label_onehot)

    fpr, tpr, _ = roc_curve(label_onehot.ravel(), score_array.ravel())
    roc_auc = auc(fpr,tpr)
    return fpr,tpr,roc_auc

def plot_roc(total_rocs,save_dir,save_name):
    plt.figure()

    for i,name in enumerate(total_rocs.keys()):
        fpr,tpr,auc_value = total_rocs[name]
        plt.plot(fpr.tolist(),tpr.tolist(),color=colors[i],lw=lw,label='ROC curve for %s (area:%0.2f)'%(name,auc_value))

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig("{}/{}_roc.png".format(save_dir,save_name),bbox_inches='tight')
    plt.savefig("{}/{}_roc.pdf".format(save_dir,save_name),bbox_inches='tight')
    plt.close()

lw = 2                                  # 线条大小
colors = ['red','blue', "darkorange","green"]                 # 使用的线条颜色
title = 'ROC for patient'
paths = {                               # 加载ROC曲线的路径
        "Efficientnet_b0":r'C:\Users\Administrator\Desktop\classification\output\waterT1C-pre\image\efficientnet_b0\val-model_best-patient-roc.json',
        "Inception_resnet_v2":r'C:\Users\Administrator\Desktop\classification\output\waterT1C-pre\image\inception_resnet_v2\val-model_best-patient-roc.json',
        "Resnet50":r'C:\Users\Administrator\Desktop\classification\output\waterT1C-pre\image\resnet50\val-model_best-patient-roc.json',
        "Xception":r'C:\Users\Administrator\Desktop\classification\output\waterT1C-pre\image\xception\val-model_best-patient-roc.json',
        }
save_dir = r'C:\Users\Administrator\Desktop\classification\output\waterT1C-pre\image'     # 保存ROC的路径
save_name = 'waterT1C-pre_ROC4-2_patient'                                                                                               # 保存生成文件的名字

total_rocs = {}
for key,path in paths.items():
    with open(path,'r') as f:
        content = json.load(f)
    score_list = content['score-list']
    label_list = content['label-list']
    num_class = len(score_list[0])
    total_rocs[key] = get_roc(score_list,label_list,num_class)
plot_roc(total_rocs,save_dir,save_name)
