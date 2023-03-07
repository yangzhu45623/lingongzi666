import argparse
import os
import json
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
from torch import optim
from eval import eval_net
from unet import UNet
from dataset import NewDataset
from torch.utils.data import DataLoader, random_split

def train_net(net, mode, device, epochs=10, batch_size=1, lr=0.001, val_percent=0.2, img_scale=0.5):
    dir_checkpoint = '{}_checkpoints/'.format(mode)
    dataset = NewDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

    criterion = nn.BCEWithLogitsLoss()
    total_acc = [0]
    total_dice = [0]
    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        # with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
        for iter, batch in enumerate(train_loader):
            imgs = batch['image']
            true_masks = batch['mask']

            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32
            true_masks = true_masks.to(device=device, dtype=mask_type)

            masks_pred = net(imgs)
            if mode.startswith("deeplabv3_resnet50"):
                masks_pred = masks_pred["out"]

            loss = criterion(masks_pred, true_masks)
            epoch_loss += loss.item()

            if iter % 10 == 0:
                print("Epoch:{} / Iter:{}, Loss:{}".format(epoch, iter, round(loss.item(), 4)))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()

        dice_coef, acc = eval_net(net, val_loader, device, mode)
        total_acc.append(float(acc))
        total_dice.append(float(dice_coef))
        scheduler.step(dice_coef)

        print('Validation Dice Coeff: {}, Acc: {}'.format(round(dice_coef, 4), round(acc, 4)))

        os.makedirs(dir_checkpoint, exist_ok=True)
        torch.save(net.state_dict(), dir_checkpoint + f'epoch{epoch + 1}.pth')
    
        dice_acc = {
            'dice':total_dice,
            'acc':total_acc
        }
        
        with open(os.path.join(dir_checkpoint,'dice_acc.json'),'w') as f:
            json.dump(dice_acc,f)
    plt.figure(figsize=(20, 10), dpi=240)
    plt.plot(list(range(len(total_dice))), total_dice,  linewidth=2, label="dice")
    plt.plot(list(range(len(total_acc))), total_acc, '--', linewidth=2, label="acc")
    plt.xticks(range(len(total_acc)))
    plt.legend(loc='lower right')
    plt.savefig("train_{}_result.png".format(mode))
    plt.show()
    plt.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=20,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=8,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')

    return parser.parse_args()


if __name__ == '__main__':
    """对应课题2的"""
    #dir_img = r'C:\Users\Administrator\Desktop\task 2\train\imgs'
    #dir_mask = r'C:\Users\Administrator\Desktop\task 2\train\masks'

    """对应课题1的"""
    dir_img = r'C:\Users\Administrator\Desktop\课题1：全图需要分割（二分类）\train\imgs'
    dir_mask = r'C:\Users\Administrator\Desktop\课题1：全图需要分割（二分类）\train\masks'

    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    networks = {"UNet_task1": UNet(),
                "deeplabv3_resnet50_task1": models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=1)
                }

    for mode, net in networks.items():
        net.to(device=device)
        train_net(net=net, mode=mode, epochs=args.epochs, batch_size=args.batchsize,
                  lr=args.lr, device=device, img_scale=args.scale, val_percent=0.2)
