import time
import json
from numpy.core.fromnumeric import mean
import torch
import os,shutil
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.backends import cudnn
from utils.util import create_models,save_recorder,get_patient2class
from utils.meter import AverageMeter,ProgressMeter

cudnn.benchmark = True
def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_patient_prib(_list,threhold):
    _class = 1
    prib = np.array(_list).mean(axis=0)

    if prib.shape[-1] != 2:             # 对于不是二分类的任务，则不进行阈值处理
        _class = prib.argmax()
        return prib.tolist(),int(_class)

    if prib[0] > threhold:
        _class = 0
    prib[0] = 1.0 + (prib[0]-1)*0.5/(1-threhold)
    prib[1] = 1-prib[0]
    return prib.tolist(),_class

class BaseTrainer():
    def __init__(self,args,train_dataset,val_dataset):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.best_acc1 = 0

        self.save_dir = self.args.outdir
        os.makedirs(self.save_dir,exist_ok=True)
        
        self.init_recorder()
        self.build_loader()
        self.build_model()
    
    def init_recorder(self):
        self.total_acc = {
            'train':[],
            'image_val':[],
            'patient_val':[]
        }
        self.total_loss = {
            'train':[],
            'val':[]
        }
    
    def save_recorder(self):
        save_recorder(os.path.join(self.save_dir,'acc_%s'%self.args.model),self.total_acc)
        save_recorder(os.path.join(self.save_dir,'loss_%s'%self.args.model),self.total_loss)

    def build_loader(self):
        self.train_loader = DataLoader(self.train_dataset,batch_size=self.args.batch_size,shuffle=True,num_workers=self.args.num_workers,pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset,batch_size=self.args.batch_size,shuffle=False,num_workers=self.args.num_workers, pin_memory=True)
        
        self.class_names = self.train_dataset.class_names
        self.class_name2label = {}
        for i,name in enumerate(self.class_names):
            self.class_name2label[name] = i

    def build_model(self):
        self.model = create_models(self.args).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(),self.args.lr,momentum=self.args.momentum,weight_decay=self.args.wd)
    
    def accuracy(self,output,target,topk=(1,)):
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def validate(self,loader):
        self.model.eval()
        total,correct,losses = 0,0,[]
        label_list,score_list = [],[]           # 存储画auc用到的数据

        y_true,y_pred = [],[]                   # 存储画混淆矩阵用到的数据

        with torch.no_grad():
            for i, (images, target,_) in enumerate(loader):
                images = images.to(self.device)
                target = target.to(self.device)

                output = self.model(images)
                loss = self.criterion(output, target)
                losses.append(loss.item())

                scores = F.softmax(output,dim=-1)
                score_list.extend(scores.detach().cpu().numpy().tolist())
                label_list.extend(target.cpu().numpy().tolist())
                
                pred = torch.argmax(output,1)
                correct += (pred==target).sum().float()
                total+=target.size(0)

                y_true.extend(target.cpu().numpy().tolist())
                y_pred.extend(pred.cpu().numpy().tolist())

            acc = (correct/total).cpu().detach().item()
            acc *= 100

            patient_metric = {
                'scores':score_list,
                'labels':score_list,
                'y_true':y_true,
                'y_pred':y_pred
            }
        return acc,mean(losses),patient_metric

    def validate_patient(self,loader):
        """这个函数计算当以图片为单位训练的时候，以病人为单位的精度"""
        self.model.eval()
        total,correct,losses = 0,0,[]
        patient2class = {}                                                          # 记录每个病人对应的class
        patient2preds = {}                                                          # 记录每个病人预测的class
        patient2predclass = {}                                                      # 记录每个病人的图片预测的结果
        img_prib = {}                                                               # 记录每个图片对应不同类别的概率

        scores_patient,labels_patient = [],[]                                               # 用于画病人roc曲线
        y_true_patient,y_pred_patient = [],[]                                               # 用于画病人混淆矩阵

        scores_image,labels_image = [],[]                                                   # 用于画图片roc曲线
        y_true_image,y_pred_image = [],[]                                                   # 用于画图片混淆矩阵

        with torch.no_grad():
            for i,(images,target,paths) in enumerate(loader):
                images = images.to(self.device)
                target = target.to(self.device)

                output = self.model(images)
                loss = self.criterion(output, target)
                losses.append(loss.item())
                pred = torch.argmax(output,1)
                prib = F.softmax(output,1)

                scores_image.extend(prib.detach().cpu().numpy().tolist())
                labels_image.extend(target.cpu().numpy().tolist())
                y_true_image.extend(target.cpu().numpy().tolist())
                y_pred_image.extend(pred.cpu().numpy().tolist())

                correct += (pred==target).sum().float()
                total+=target.size(0)

                for i,path in enumerate(paths):                                          # 提取出来每个病人真实的类别
                    patient,_class = get_patient2class(path)                             # 通过path提取该图片的类别和病人
                    patient2class[patient] = self.class_name2label[_class]               # 记录病人的真实标注
                    img_prib[path] = [round(item,4) for item in prib[i].cpu().tolist()]  # 记录这张图片分成不同类别的概率
                    temp = patient2preds.setdefault(patient,[])
                    temp.append(prib[i].cpu().numpy().tolist())

            """计算image的acc"""
            imgacc = float((correct/total).cpu().detach().data.numpy())
            imgacc *= 100

            """计算patient的acc"""
            correct,total = 0,0
            for key,val in patient2preds.items():
                score,pred = get_patient_prib(val,self.args.alpha)
                patient2predclass[key] = pred
                correct += int(pred == patient2class[key]) 
                total+=1

                labels_patient.append(patient2class[key])
                scores_patient.append(score)

                y_true_patient.append(patient2class[key])
                y_pred_patient.append(pred)
            patient_acc = 100*(correct/total)

            patient_metric = {
                'scores':scores_patient,
                'labels':labels_patient,
                'y_true':y_true_patient,
                'y_pred':y_pred_patient
            }

            image_metric = {
                'scores':scores_image,
                'labels':labels_image,
                'y_true':y_true_image,
                'y_pred':y_pred_image
            }
            return imgacc,mean(losses),patient_acc,patient2predclass,patient2preds,img_prib,patient_metric,image_metric

    def save_model(self,is_best):
        state = {
            'epoch':self.epoch+1,
            'arch': self.args.model,
            'state_dict': self.model.state_dict(),
            'best_acc1': self.best_acc1,
            'optimizer': self.optimizer.state_dict(),
        }

        filename = os.path.join(self.save_dir,'%s_epoch%d'%(self.args.model,self.epoch))
        torch.save(state,filename)
        if is_best:
            shutil.copyfile(filename, '{}/model_best.pth'.format(self.save_dir))
    
    def load_model(self,pth):
        filename = os.path.join(pth)
        self.model.load_state_dict(torch.load(filename)['state_dict'])

class TrainTrainer(BaseTrainer):
    def __init__(self, args, train_dataset, val_dataset):
        super().__init__(args, train_dataset, val_dataset)

    def train(self):
        """before train"""
        for self.epoch in range(0,self.args.epochs):
            meters = self.before_epoch()
            epoch_loss = self.train_epoch(*meters)
            self.after_epoch(epoch_loss)
        
        self.save_recorder()

    def before_epoch(self):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        progress = ProgressMeter(len(self.train_loader),[batch_time, data_time, losses, top1],prefix="Epoch: [{}]".format(self.epoch))
        adjust_learning_rate(self.optimizer,self.epoch,self.args)
        self.model.train()
        return batch_time,data_time,losses,top1,progress
        
    def train_epoch(self,*argss):
        batch_time,data_time,losses,top1,progress = argss
        epoch_loss = []
        end = time.time()
        
        for i,(images,target,path) in enumerate(self.train_loader):
            data_time.update(time.time() - end)
            images = images.to(self.device)
            target = target.to(self.device)
            output = self.model(images)
            loss = self.criterion(output,target)
            epoch_loss.append(loss.item())
            acc1, = self.accuracy(output,target)
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            if i % self.args.print_freq == 0:
                progress.display(i)
        return mean(epoch_loss)
            
    def after_epoch(self,epoch_loss):
        self.total_loss['train'].append(epoch_loss)
        if self.args.train_type == 'patient':
            res = self.validate(self.val_loader)
            acc1,val_loss = res[0],res[1]
            self.total_acc['patient_val'].append(acc1)
            self.total_loss['val'].append(val_loss)
            print('Validation Patient Accuracy: %.2f'%acc1)
        else:
            res = self.validate_patient(self.val_loader)
            img_acc,val_loss,patient_acc = res[0],res[1],res[2]
            self.total_acc['image_val'].append(img_acc)
            self.total_loss['val'].append(val_loss)
            self.total_acc['patient_val'].append(patient_acc)
            print('Validation Image Accuracy: %.2f, Validation Patient Accuracy: %.2f'%(img_acc,patient_acc))
            acc1 = patient_acc
        is_best = acc1 > self.best_acc1
        self.best_acc1 = max(acc1, self.best_acc1)

        self.save_model(is_best)

class TestTrainer(BaseTrainer):
    def __init__(self, args, train_dataset, val_dataset):
        super().__init__(args, train_dataset, val_dataset)

    def test(self,dset_mode):
        if dset_mode == 'train':
            loader = self.train_loader
        else:
            loader = self.val_loader
        
        test_model_path = self.args.test_model_path.format(self.args.train_type,self.args.model)
        self.load_model(test_model_path)                  # 加载模型
        self.save_dir,test_model = os.path.split(test_model_path)
        test_model = os.path.splitext(test_model)[0]
        if self.args.train_type == 'patient':
            acc1,_,patient_metric = self.validate(loader)
            print('%s, Patient Accuracy: %.2f'%(dset_mode,acc1))
        else:
            acc1,_,patient_acc,patient2predclass,patient2preds,img_prib,patient_metric,image_metric = self.validate_patient(loader)
            print('%s Image Accuracy: %.2f, Patient Accuracy: %.2f'%(dset_mode,acc1,patient_acc))

            with open(os.path.join(self.save_dir,'%s-%s-patient_pred.json'%(dset_mode,test_model)),'w') as f:
                json.dump(patient2predclass,f)
            with open(os.path.join(self.save_dir,'%s-%s-patient_img_pred.json'%(dset_mode,test_model)),'w') as f:
                json.dump(patient2preds,f)
            
            with open(os.path.join(self.save_dir,'%s-%s-patient_img_prib.json'%(dset_mode,test_model)),'w') as f:
                json.dump(img_prib,f)
            
                """保存image画roc用到的数据"""
            with open(os.path.join(self.save_dir,'%s-%s-image-roc.json'%(dset_mode,test_model)),'w') as f:
                json.dump({
                    'score-list':image_metric['scores'],
                    'label-list':image_metric['labels']
                },f)
            with open(os.path.join(self.save_dir,'%s-%s-image-cm.json'%(dset_mode,test_model)),'w') as f:
                json.dump({
                    'y-pred':image_metric['y_pred'],
                    'y-true':image_metric['y_true']
                },f)
        """保存patient 画 roc用到的数据"""
        with open(os.path.join(self.save_dir,'%s-%s-patient-roc.json'%(dset_mode,test_model)),'w') as f:
            json.dump({
                'score-list':patient_metric['scores'],
                'label-list':patient_metric['labels']
            },f)
        with open(os.path.join(self.save_dir,'%s-%s-patient-cm.json'%(dset_mode,test_model)),'w') as f:
            json.dump({
                'y-pred':patient_metric['y_pred'],
                'y-true':patient_metric['y_true']
            },f)
        

