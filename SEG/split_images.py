import glob, os, random
import shutil

if __name__ == "__main__":

    images = glob.glob("data/train/imgs/*.jpg")
    masks = [p.replace("imgs", "masks") for p in images]
    img_mask = list(zip(images, masks))
    random.shuffle(img_mask)

    ratio = 0.8
    num_train = int(len(images) * ratio)
    train_images = img_mask[:num_train]
    test_images = img_mask[num_train:]

    for p_img, p_mask in test_images:
        new_img_name = p_img.replace("train", "test")
        new_mask_name = p_mask.replace("train", "test")
        shutil.move(p_img, new_img_name)
        shutil.move(p_mask, new_mask_name)
