import enum
import pickle
from sys import path
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import argparse


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        self.len = len(self.path)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = Image.open(self.path[idx])
        data = np.asarray(data.convert('RGB'))

        return self.transform(data)


def num_to_color(num):
    if isinstance(num, list):
        num = num[0]

    if num == 0:
        color = (200, 200, 200)
    elif num == 1:
        color = (0, 65, 255)
    elif num == 2:
        color = (255, 40, 0)
    elif num == 3:
        color = (53, 161, 107)
    elif num == 4:
        color = (250, 245, 0)
    elif num == 5:
        color = (102, 204, 255)
    return color


def patch_to_wsi(pred, width, height, resized_size, size, stride, output_path, output_name, cnt):
    pred_color = list(map(num_to_color, pred))
    row_max = int((width - size[0]) / stride + 1)
    column_max = int((height - size[1]) / stride + 1)
    canvas = Image.new('RGB', (int(resized_size[0] * row_max * (stride / size[0])), int(
        resized_size[1] * column_max * (stride / size[1]))), (255, 255, 255))

    for column in range(column_max):
        for row in range(row_max):
            # img = Image.open(input_path + input_name + str(cnt).zfill(10) +
            #                  ".png", 'r').resize((resized_size[0], resized_size[1]))
            x = np.zeros((resized_size[0], resized_size[1], 3))
            x[:, :] = pred_color[cnt]
            img = Image.fromarray(x.astype(np.uint8))
            canvas.paste(
                img, (row * resized_size[0], column * resized_size[1]))
            cnt = cnt + 1

    return canvas

    # # save
    # canvas.save(output_path + output_name + '.png', 'PNG', quality=100)


def prediction(model, path_list):
    model.eval()

    dataset = Dataset(path_list)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=256,
        shuffle=False,  num_workers=2)

    pred = []
    for data in tqdm(dataloader, leave=False):
        data = data.cuda()
        y = model(data)
        pred.extend(y.argmax(1).cpu().detach().numpy())

    return np.array(pred)


def main(model, image_name, width, height, path_list, output_path):
    pred = prediction(model=model, path_list=path_list)
    img = patch_to_wsi(
        pred=pred, width=width, height=height,
        resized_size=(32, 32),
        size=(256, 256),
        stride=256,
        output_path=output_path,
        output_name=image_name,
        cnt=0)
    return img


def make_qualitative_result(wsi_name, save_path):

    num_classes = 3
    level = 2

    # load model
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.cuda()
    model.load_state_dict(torch.load(
        model_path, map_location=torch.device('cuda:0')))

    for image_name in tqdm(wsi_name):
        img = openslide.OpenSlide(dataset_path+'origin/'+image_name+'.ndpi')
        (width, height) = img.level_dimensions[level]
        path_list = glob(dataset_path+'test/'+image_name+'/*')
        path_list.sort()
        image = main(model=model,
                     image_name=image_name,
                     width=width, height=height,
                     path_list=path_list,
                     output_path=save_path)

        # overlaid
        # image = Image.open(save_path+image_name+'.png')
        image_gt = Image.open(dataset_path+'overlaid_[0, 1, 2]/rgb/' +
                              image_name+'_overlaid.tif')

        WIDTH = image.size[0]
        HEIGHT = image.size[1]

        for x in range(WIDTH):
            for y in range(HEIGHT):
                if image_gt.getpixel((x, y)) == (0, 0, 0):
                    image.putpixel((x, y), (0, 0, 0))
                elif image_gt.getpixel((x, y)) == (255, 255, 255):
                    image.putpixel((x, y), (255, 255, 255))

        # image.save(save_path+image_name+'_result.png',
        #            'PNG', quality=100, optimize=True)

        image_gt = np.asarray(image_gt.convert('RGB'))
        image = np.asarray(image.convert('RGB'))

        fig = plt.figure(figsize=[10, 5])
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(image_gt)
        ax.axis("off")
        ax = fig.add_subplot(1, 2, 2)
        ax.imshow(image)
        ax.axis("off")
        plt.savefig(save_path+image_name+'_result.png',
                    bbox_inches='tight')
        plt.close()

    ####### sumalize ############
    path_list = glob(save_path+'/*')
    path_list.sort()
    fig = plt.figure()
    for i, path in enumerate(path_list[:10]):
        ax = fig.add_subplot(10, 1, i+1)
        img = Image.open(path)
        img = img.convert("RGB")
        ax.imshow(img)
        ax.axis("off")
    plt.savefig(save_path+'/0_sumalize.png',  bbox_inches="tight", dpi=500)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_path', default='result_ws03/wsi-fpl/lr_0.0003-seed_0-eta_5/',
                        type=str, help='paths saving ther results')
    args = parser.parse_args()

    dataset_path = '../dataset/chemotherapy/202203_chemotherapy/'
    model_path = args.save_path + 'best_model.pth'
    save_path = args.save_path + 'qualitative/'
    print(dataset_path)
    print(model_path)
    print(save_path)

    # load file
    with open(dataset_path+'all/train_name.pkl', "rb") as tf:
        test_wsi_name = pickle.load(tf)[:10]
    save_path += 'train/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    make_qualitative_result(
        wsi_name=test_wsi_name,
        save_path=save_path)
