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

log = logging.getLogger(__name__)


class DatasetBag(torch.utils.data.Dataset):
    def __init__(self, name, data, label, proportion):
        self.name_list = name
        self.data_list = data
        self.label_list = label
        self.proportion_list = proportion
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        self.len = len(self.name_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        name = self.name_list[idx]
        (b, w, h, c) = self.data_list[name].shape
        data = torch.zeros((b, c, w, h))
        for i in range(b):
            data[i] = self.transform(self.data_list[name][i])

        label = self.label_list[name]
        label = torch.tensor(label).long()

        proportion = self.proportion_list[name]
        proportion = torch.tensor(proportion).float()

        return data, label, proportion


@ hydra.main(config_path='../config', config_name='config-proportion-loss')
def main(cfg: DictConfig) -> None:

    # file name
    cwd = hydra.utils.get_original_cwd()

    dataset_path = cwd + cfg.dataset.dir

    # test
    with open(dataset_path+'test_name.pkl', "rb") as tf:
        test_wsi_name = pickle.load(tf)
    with open(dataset_path+'test_data.pkl', "rb") as tf:
        test_data = pickle.load(tf)
    with open(dataset_path+'test_label.pkl', "rb") as tf:
        test_label = pickle.load(tf)
    with open(dataset_path+'test_proportion.pkl', "rb") as tf:
        test_proportion = pickle.load(tf)

    test_dataset = DatasetBag(
        name=test_wsi_name,
        data=test_data,
        label=test_label,
        proportion=test_proportion)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1,
        shuffle=False,  num_workers=cfg.num_workers)

    # define model
    fix_seed(cfg.seed)
    if cfg.model == 'resnet50':
        model = resnet50(weights='ResNet50_Weights.DEFAULT')
    elif cfg.model == 'resnet18':
        model = resnet18(weights='ResNet18_Weights.DEFAULT')
    else:
        log.info('No model!')
    model.fc = nn.Linear(model.fc.in_features, cfg.dataset.num_classes)
    model = model.to(cfg.device)
    model.load_state_dict(torch.load(
        cwd+'/model_backup/proportion_loss.pth', map_location=torch.device('cuda:0')))

    l1_function = ProportionLoss(metric=cfg.val_metric)

    # validation
    model.eval()
    gt, pred = [], []
    with torch.no_grad():
        for data, label, proportion in tqdm(test_loader, leave=False):
            data, proportion = data[0], proportion[0]
            gt.extend(label[0])

            confidence = []
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
                confidence.extend(
                    F.softmax(y, dim=1).cpu().detach().numpy())

            pred_prop = torch.tensor(np.array(confidence)).mean(dim=0)
            l1_loss = l1_function(pred_prop, proportion)

        acc = np.array(np.array(gt) == np.array(pred)).mean()
        cm = confusion_matrix(y_true=gt, y_pred=pred)
        OP, PC, mIoU = cal_OP_PC_mIoU(cm)
        log.info('test loss: %.4f, OP: %.4f, PC: %.4f, mIoU: %.4f' %
                 (l1_loss, OP, PC, mIoU))

    print(l1_loss)


if __name__ == '__main__':
    main()
