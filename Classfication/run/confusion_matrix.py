import numpy as np
import json
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def get_cm(path):
    with open(path,'r') as f:
        content = json.load(f)
    y_true = np.array(content['y-true'])
    y_pred = np.array(content['y-pred'])

    cm = confusion_matrix(y_true,y_pred)
    return cm

def plot_confusion_matrix(cm,
                          save_dir,
                          save_name,
                          normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.3f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "black")

    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.3f}; misclass={:0.3f}'.format(accuracy, misclass))
    plt.savefig("{}/{}_confusion_matrix.png".format(save_dir,save_name))
    plt.savefig("{}/{}_confusion_matrix.pdf".format(save_dir,save_name))
    plt.close()






title = 'Xception'                                 # 图的title，你自己定
cmap = plt.get_cmap('Blues')                    # 使用的颜色色系，具体怎么改，查百度
target_names = ['High risk','Low risk']                    # 分类的类别

paths = {                               # 
        "image-val":r'C:\Users\Administrator\Desktop\classification\output\waterT1C-post\image\xception\val-model_best-image-cm.json',
        "patient-val":r'C:\Users\Administrator\Desktop\classification\output\waterT1C-post\image\xception\val-model_best-patient-cm.json'
        }
save_dir = r'C:\Users\Administrator\Desktop\classification\output\waterT1C-post\image\xception'     #               保存的路径，已经要输入有效路径

for key,path in paths.items():
    cm = get_cm(path)
    plot_confusion_matrix(cm,save_dir,key)
