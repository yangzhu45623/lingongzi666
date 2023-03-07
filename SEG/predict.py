import argparse
import os
import copy
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import glob
import numpy as np
import torch
from PIL import Image
from unet import UNet
from dataset import NewDataset

def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5, mode="deeplabv3_resnet50"):
    net.eval()
    img = NewDataset.preprocess(full_img, scale_factor)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        if mode.startswith("deeplabv3_resnet50"):
            output = output["out"]
        probs = torch.sigmoid(output)
        probs = probs.squeeze(0).squeeze(0)

        pred_mask = np.uint8((probs > 0.5).cpu().numpy())
    return pred_mask

def crop_image(mask):
    try:
        coor = np.nonzero(mask)
        xmin = coor[0][0]
        xmax = coor[0][-1]
        coor[1].sort() # 直接改变原数组，没有返回值
        ymin = coor[1][0]
        ymax = coor[1][-1]
    except IndexError:
        return None,None,None,None

    return xmin,ymin,xmax,ymax


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='checkpoints/CP_epoch20.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=True)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()


if __name__ == "__main__":
    """
    ！！！！ 注意
    运行之前，先把 test_dir下面所有的unet_seg文件夹和deeplab3_seg文件夹删除，不然会报错
    ！！！！！

    """
    args = get_args()
    # 下面这个是对课题2的
    # test_dir = r"C:\Users\Administrator\Desktop\task 2\test\imgs"
    # paths = glob.glob(os.path.join(test_dir,"*","*","*.png"))
    # 下面这个是对课题1的
    test_dir = r"C:\Users\Administrator\Desktop\课题1：全图需要分割（二分类）\test"
    paths = glob.glob(os.path.join(test_dir,"*","*","*","*","*.png"))
    print(paths)

    net1 = UNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net1.to(device=device)
    net1.load_state_dict(torch.load("UNet_checkpoints/epoch10.pth", map_location=device))

    import torchvision.models as models

    net2 = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=1)
    net2.to(device=device)
    net2.load_state_dict(torch.load("deeplabv3_resnet50_checkpoints/epoch10.pth", map_location=device))
    with torch.no_grad():
        for i, p in enumerate(paths):
            src = Image.open(p).convert('RGB')

            mask_unet = predict_img(net=net1,
                                    full_img=src,
                                    scale_factor=args.scale,
                                    out_threshold=args.mask_threshold,
                                    device=device,
                                    mode="UNet")

            mask_deeplab = predict_img(net=net2,
                                       full_img=src,
                                       scale_factor=args.scale,
                                       out_threshold=args.mask_threshold,
                                       device=device,
                                       mode="deeplabv3_resnet50")
            img = np.array(src)

            ## resize到跟原图尺寸一致
            mask_unet = np.array(Image.fromarray(mask_unet).resize(img.shape[:2]))
            mask_deeplab = np.array(Image.fromarray(mask_deeplab).resize(img.shape[:2]))

            # for unet
            res_unet = copy.deepcopy(img)

            """如果是想裁剪，就用下面的代码，否则把下面代码注释掉"""
            xmin,ymin,xmax,ymax = crop_image(mask_unet)
            if xmin is None: continue
            res_unet = res_unet[xmin:xmax, ymin:ymax]
            """如果是想带黑色背景的结果，就用下面的代码"""
            # res_unet[mask_unet==0] = 0

            save_path = p.replace('imgs','unet_seg')
            if not os.path.exists(os.path.split(save_path)[0]):
                os.makedirs(os.path.split(save_path)[0],exist_ok=True)
            Image.fromarray(res_unet).save(save_path)


            # for deeplab
            res_deeplab = copy.deepcopy(img)
            """如果是想裁剪，就用下面的代码，否则把下面代码注释掉"""
            xmin,ymin,xmax,ymax = crop_image(mask_deeplab)
            res_deeplab = res_deeplab[xmin:xmax, ymin:ymax]

            """如果是想带黑色背景的结果，就用下面的代码"""
            # res_deeplab[mask_deeplab==0] = 0

            save_path = p.replace('imgs','deeplab3_seg')
            if not os.path.exists(os.path.split(save_path)[0]):
                os.makedirs(os.path.split(save_path)[0],exist_ok=True)
            Image.fromarray(res_deeplab).save(save_path)

            print('Done! %s'%p)
