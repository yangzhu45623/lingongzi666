import os,shutil
import random

random.seed(0)

# root = r'C:\Users\Administrator\Documents\vscode\project1\capture\capture image-CCRT'
root = r'C:\Users\Administrator\Documents\vscode\project2\capture'
classes = os.listdir(root)

os.makedirs(os.path.join(root,'train'),exist_ok=True)
os.makedirs(os.path.join(root,'test'),exist_ok=True)

for _class in classes:
    patients = list(os.listdir(os.path.join(root,_class)))
    random.shuffle(patients)
    train_patients = patients[:int(len(patients)*0.8)]
    test_patients = patients[int(len(patients)*0.8):]

    for patient in train_patients:
        shutil.copytree(os.path.join(root,_class,patient),os.path.join(root,'train',_class,patient))
    
    for patient in test_patients:
        shutil.copytree(os.path.join(root,_class,patient),os.path.join(root,'test',_class,patient))