
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import time
import gc
from sklearn.metrics import confusion_matrix
import torchvision.transforms as transforms
from utils import Dataset, fix_seed, make_folder, get_rampup_weight, cal_OP_PC_mIoU, save_confusion_matrix
from losses import PiModelLoss, ProportionLoss, VATLoss
from PIL import Image

log = logging.getLogger(__name__)




class DatasetBag(torch.utils.data.Dataset):
    def __init__(self, name, path, cwd):
        self.name_list = name
        self.path_list = path
        self.cwd = cwd
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        self.len = len(self.name_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        name = self.name_list[idx]

        data_list = []
        for path in self.path_list[name]:
            data = Image.open(self.cwd+'/'+path)
            data = np.asarray(data.convert('RGB'))
            data_list.append(data)
        data_list = np.array(data_list)

        (b, w, h, c) = data_list.shape
        data = torch.zeros((b, c, w, h))
        for i in range(b):
            data[i] = self.transform(data_list[i])

        return name, data


@ hydra.main(config_path='../config', config_name='config-proportion-loss')
def main(cfg: DictConfig) -> None:

    # file name
    cwd = hydra.utils.get_original_cwd()

    dataset_path = cwd + '/../dataset/chemotherapy_new/'

    with open(dataset_path+'wsi_name.pkl', "rb") as tf:
        wsi_name = pickle.load(tf)
    with open(dataset_path+'patch_path.pkl', "rb") as tf:
        patch_path = pickle.load(tf)

    dataset = DatasetBag(
        name=wsi_name,
        path=patch_path,
        cwd=cwd)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1,
        shuffle=False,  num_workers=cfg.num_workers)

    # define model
    model = resnet18(weights='ResNet18_Weights.DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, cfg.dataset.num_classes)
    model = model.to(cfg.device)
    model.load_state_dict(torch.load(
        cwd+'/model_backup/full_supervised.pth', map_location=torch.device('cuda:0')))

    model.eval()
    proportion_dict = {}
    with torch.no_grad():
        for name, data in tqdm(loader, leave=False):
            pred = []
            data = data[0]
            name = name[0]
            if (data.size(0) % cfg.batch_size) == 0:
                J = int((data.size(0)//cfg.batch_size))
            else:
                J = int((data.size(0)//cfg.batch_size)+1)

            for j in range(J):
                if j+1 != J:
                    data_j = data[j*cfg.batch_size: (j+1)*cfg.batch_size]
                else:
                    data_j = data[j*cfg.batch_size:]

                data_j = data_j.to(cfg.device)
                y = model(data_j)
                pred.extend(y.argmax(1).cpu().detach().numpy())
            
            proportion = np.eye(3)[pred].mean(axis=0)
            print(name, proportion)
            proportion_dict[name] = proportion
            
    with open(cwd+'/proportion_dict_new_data.pkl', "wb") as tf:
        pickle.dump(proportion_dict, tf)
        



if __name__ == '__main__':
    main()
    with open('proportion_dict_new_data.pkl', "rb") as tf:
        proportion_dict = pickle.load(tf)
    print(proportion_dict)


# from glob import glob
# import os
# from PIL import Image
# import numpy as np
# import pickle
# from tqdm import tqdm
# import matplotlib.pyplot as plt


# def get_folder(c):
#     f = glob('../dataset/chemotherapy_new/%d/*' % c)
#     return set(list([os.path.split(x)[-1][:-4] for x in f]))


# for c in range(3):
#     print(len(get_folder(c)))

# dataset_path = '../dataset/chemotherapy_new/'
# save_path = '../dataset/chemotherapy_new/'
# wsi_name = list(set(get_folder(0)) | set(get_folder(1)) | set(get_folder(2)))

# path_dict = {}
# for wsi in tqdm(wsi_name):
#     bag_path = []
#     for c in range(3):
#         for i in range(10):
#             for p in glob(dataset_path+str(c)+'/'+wsi+'_00'+str(i)+'/*'):
#                 bag_path.append(p)
#     path_dict[wsi] = np.array(bag_path)

# with open(save_path+'wsi_name.pkl', "wb") as tf:
#     pickle.dump(wsi_name, tf)
# with open(save_path+'patch_path.pkl', "wb") as tf:
#     pickle.dump(path_dict, tf)

