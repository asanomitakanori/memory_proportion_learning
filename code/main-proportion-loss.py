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


class DatasetBagSampling(torch.utils.data.Dataset):
    def __init__(self, name, path, label, proportion, num_sampled_instances, cwd):
        self.name_list = name
        self.path_list = path
        self.label_list = label
        self.proportion_list = proportion
        self.num_sampled_instances = num_sampled_instances
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
        if b > self.num_sampled_instances:
            index = np.arange(b)
            sampled_index = np.random.choice(index, self.num_sampled_instances)

            data = torch.zeros((self.num_sampled_instances, c, w, h))
            label = torch.zeros((self.num_sampled_instances))
            for i, j in enumerate(sampled_index):
                data[i] = self.transform(data_list[j])
                label[i] = self.label_list[name][j]
        else:
            data = torch.zeros((b, c, w, h))
            label = torch.zeros((b))
            for i in range(b):
                data[i] = self.transform(data_list[i])
                label[i] = self.label_list[name][i]

        # label = torch.tensor(label).long()

        proportion = self.proportion_list[name]
        proportion = torch.tensor(proportion).float()

        return data, label, proportion


class DatasetBag(torch.utils.data.Dataset):
    def __init__(self, name, path, label, proportion, cwd):
        self.name_list = name
        self.path_list = path
        self.label_list = label
        self.proportion_list = proportion
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

        label = self.label_list[name]
        label = torch.tensor(label).long()

        proportion = self.proportion_list[name]
        proportion = torch.tensor(proportion).float()

        return data, label, proportion


def load_data(dataset_path, kind):
    if kind == 'train':
        with open(dataset_path+kind+'_index.pkl', "rb") as tf:
            index = pickle.load(tf)
    else:
        index = []
    with open(dataset_path+kind+'_name.pkl', "rb") as tf:
        wsi_name = pickle.load(tf)
    with open(dataset_path+kind+'_path.pkl', "rb") as tf:
        path = pickle.load(tf)
    with open(dataset_path+kind+'_label.pkl', "rb") as tf:
        label = pickle.load(tf)
    with open(dataset_path+kind+'_proportion.pkl', "rb") as tf:
        proportion = pickle.load(tf)
    return index, wsi_name, path, label, proportion


def evaluation(model, loader, cfg):
    model.eval()
    l1_function = ProportionLoss(metric=cfg.val_metric)
    gt, pred = [], []
    with torch.no_grad():
        for data, label, proportion in tqdm(loader, leave=False):
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
                # ここを書き換える
                y = model(data_j)
                pred.extend(y.argmax(1).cpu().detach().numpy())
                confidence.extend(
                    F.softmax(y, dim=1).cpu().detach().numpy())

            pred_prop = torch.tensor(np.array(confidence)).mean(dim=0)
            l1 = l1_function(pred_prop, proportion).item()

        # acc = np.array(np.array(gt) == np.array(pred)).mean()
        cm = confusion_matrix(y_true=gt, y_pred=pred)
        OP, PC, mIoU = cal_OP_PC_mIoU(cm)

    return l1, cm, OP, PC, mIoU


@ hydra.main(config_path='../config', config_name='config-proportion-loss')
def main(cfg: DictConfig) -> None:

    # file name
    cwd = hydra.utils.get_original_cwd()
    result_path = cwd + cfg.result_path
    result_path += 'wsi-proportion-loss/'
    make_folder(result_path)
    result_path += cfg.consistency
    result_path += '-samp_%s' % str(cfg.num_sampled_instances)
    result_path += '-lr_%s' % str(cfg.lr)
    result_path += '-seed_%s' % str(cfg.seed)
    result_path += '/'
    make_folder(result_path)

    fh = logging.FileHandler(result_path+'exec.log')
    log.addHandler(fh)
    log.info(OmegaConf.to_yaml(cfg))
    log.info('cwd:%s' % cwd)

    dataset_path = cwd + cfg.dataset_dir

    train_index, train_wsi_name, train_path, train_label, train_proportion = \
        load_data(dataset_path, 'train')

    _, val_wsi_name, val_path, val_label, val_proportion = \
        load_data(dataset_path, 'val')

    _, test_wsi_name, test_path, test_label, test_proportion = \
        load_data(dataset_path, 'test')

    # define loader
    train_dataset = DatasetBagSampling(
        name=train_wsi_name,
        path=train_path,
        label=train_label,
        proportion=train_proportion,
        num_sampled_instances=cfg.num_sampled_instances,
        cwd=cwd)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1,
        shuffle=True,  num_workers=cfg.num_workers)

    val_dataset = DatasetBag(
        name=val_wsi_name,
        path=val_path,
        label=val_label,
        proportion=val_proportion,
        cwd=cwd)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1,
        shuffle=False,  num_workers=cfg.num_workers)

    test_dataset = DatasetBag(
        name=test_wsi_name,
        path=test_path,
        label=test_label,
        proportion=test_proportion,
        cwd=cwd)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1,
        shuffle=False,  num_workers=cfg.num_workers)

    # define model
    fix_seed(cfg.seed)
    if cfg.model == 'resnet50':
        model = resnet50(weights='ResNet50_Weights.DEFAULT')
    elif cfg.model == 'resnet18':
        model = resnet18(pretrained=True)
        # model = Resnet18(3)
    else:
        log.info('No model!')
    model.fc = nn.Linear(model.fc.in_features, cfg.num_classes)
    model = model.to(cfg.device)

    # define criterion and optimizer
    loss_function = ProportionLoss(metric=cfg.proportion_metric)
    if cfg.consistency == 'none':
        consistency_criterion = None
    elif cfg.consistency == 'vat':
        consistency_criterion = VATLoss()
    elif cfg.consistency == 'pi':
        consistency_criterion = PiModelLoss()
    else:
        raise NameError('Unknown consistency criterion')

    l1_function = ProportionLoss(metric=cfg.val_metric)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    fix_seed(cfg.seed)
    # train_acces, test_acces = [], []
    train_OPs, train_PCs, train_mIoUs = [], [], []
    test_OPs, test_PCs, test_mIoUs = [], [], []
    train_losses, val_losses, test_losses = [], [], []
    best_validation_loss = np.inf
    final_OP, final_PC, final_mIoU = 0, 0, 0
    for epoch in range(cfg.num_epochs):
        s_time = time.time()

        # train
        model.train()
        losses = []
        b_list = [0]
        mb_data, mb_proportion = [], []
        gt, pred = [], []
        for iteration, (data, label, proportion) in enumerate(tqdm(train_loader, leave=False)):
            data = data[0]
            gt.extend(label[0].numpy())
            b_list.append(b_list[-1]+data.size(0))
            mb_data.extend(data)
            mb_proportion.extend(proportion)

            if (iteration+1) % cfg.mini_batch == 0 or (iteration + 1) == len(train_loader):
                mb_data = torch.stack(mb_data)
                mb_proportion = torch.stack(mb_proportion)
                mb_data = mb_data.to(cfg.device)
                mb_proportion = mb_proportion.to(cfg.device)

                if cfg.consistency == "vat":
                    # VAT should be calculated before the forward for cross entropy
                    consistency_loss = consistency_criterion(model, mb_data)
                elif cfg.consistency == "pi":
                    consistency_loss, _ = consistency_criterion(model, mb_data)
                else:
                    consistency_loss = torch.tensor(0.)
                alpha = get_rampup_weight(0.05, iteration, -1)
                consistency_loss = alpha * consistency_loss

                y = model(mb_data)
                pred.extend(y.argmax(1).cpu().detach().numpy())
                confidence = F.softmax(y, dim=1)
                pred_prop = torch.zeros(mb_proportion.size(
                    0), cfg.num_classes).to(cfg.device)
                for n in range(mb_proportion.size(0)):
                    pred_prop[n] = torch.mean(
                        confidence[b_list[n]: b_list[n+1]], dim=0)
                        
                prop_loss = loss_function(pred_prop, mb_proportion)
                loss = prop_loss + consistency_loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                b_list = [0]
                mb_data, mb_proportion = [], []

                losses.append(loss.item())

        train_loss = np.array(losses).mean()
        train_losses.append(train_loss)

        # train_acc = np.array(np.array(gt) == np.array(pred)).mean()
        train_cm = confusion_matrix(y_true=gt, y_pred=pred)
        train_OP, train_PC, train_mIoU = cal_OP_PC_mIoU(train_cm)
        train_OPs.append(train_OP)
        train_PCs.append(train_PC)
        train_mIoUs.append(train_mIoU)

        e_time = time.time()
        log.info('[Epoch: %d/%d (%ds)] train OP: %.4f, PC: %.4f, mIoU: %.4f' %
                 (epoch+1, cfg.num_epochs, e_time-s_time, train_OP, train_PC, train_mIoU))

        # validation
        # s_time = time.time()
        # model.eval()
        # losses, l1_losses = [], []
        # with torch.no_grad():
        #     for data, _, proportion in tqdm(val_loader, leave=False):
        #         data, proportion = data[0], proportion[0]

        #         confidence = []
        #         if (data.size(0) % cfg.batch_size) == 0:
        #             J = int((data.size(0)//cfg.batch_size))
        #         else:
        #             J = int((data.size(0)//cfg.batch_size)+1)

        #         for j in range(J):
        #             if j+1 != J:
        #                 data_j = data[j*cfg.batch_size: (j+1)*cfg.batch_size]
        #             else:
        #                 data_j = data[j*cfg.batch_size:]

        #             data_j = data_j.to(cfg.device)
        #             y = model(data_j)
        #             confidence.extend(
        #                 F.softmax(y, dim=1).cpu().detach().numpy())

        #         pred = torch.tensor(np.array(confidence)).mean(dim=0)
        #         l1_loss = l1_function(pred, proportion)
        #         l1_losses.append(l1_loss.item())

        # val_loss = np.array(l1_losses).mean()
        # val_losses.append(val_loss)
        # e_time = time.time()
        # log.info('[Epoch: %d/%d (%ds)] val_l1_loss: %.4f' %
        #          (epoch+1, cfg.num_epochs, e_time-s_time, val_loss))

        # # test
        # model.eval()
        # gt, pred = [], []
        # with torch.no_grad():
        #     for data, label in tqdm(test_loader, leave=False):
        #         data, label = data.to(cfg.device), label.to(cfg.device)
        #         y = model(data)
        #         gt.extend(label.cpu().detach().numpy())
        #         pred.extend(y.argmax(1).cpu().detach().numpy())
        # # test_acc = (np.array(gt) == np.array(pred)).mean()

        # test_cm = confusion_matrix(y_true=gt, y_pred=pred)
        # test_OP, test_PC, test_mIoU = cal_OP_PC_mIoU(test_cm)
        # test_OPs.append(test_OP)
        # test_PCs.append(test_PC)
        # test_mIoUs.append(test_mIoU)

        # e_time = time.time()

        # log.info('[Epoch: %d/%d (%ds)] test OP: %.4f, PC: %.4f, mIoU: %.4f' %
        #          (epoch+1, cfg.num_epochs, e_time-s_time, test_OP, test_PC, test_mIoU))
        # log.info(
        #     '-----------------------------------------------------------------------')
        
        # validation
        s_time = time.time()
        val_l1, val_cm, val_OP, val_PC, val_mIoU = evaluation(
            model, val_loader, cfg)
        e_time = time.time()
        val_losses.append(val_l1)
        log.info('[Epoch: %d/%d (%ds)] val l1: %.4f, OP: %.4f, PC: %.4f, mIoU: %.4f' %
                 (epoch+1, cfg.num_epochs, e_time-s_time,
                  val_l1, val_OP, val_PC, val_mIoU))
        # test
        s_time = time.time()
        test_l1, test_cm, test_OP, test_PC, test_mIoU = evaluation(
            model, test_loader, cfg)
        test_OPs.append(test_OP)
        test_PCs.append(test_PC)
        test_mIoUs.append(test_mIoU)
        test_losses.append(test_l1)

        e_time = time.time()
        log.info('[Epoch: %d/%d (%ds)] test l1: %.4f, OP: %.4f, PC: %.4f, mIoU: %.4f' %
                 (epoch+1, cfg.num_epochs, e_time-s_time,
                  test_l1, test_OP, test_PC, test_mIoU))
        log.info(
            '-----------------------------------------------------------------------')

        if val_l1 < best_validation_loss:
            torch.save(model.state_dict(), result_path + 'best_model.pth')
            save_confusion_matrix(cm=train_cm, path=result_path+'cm_train.png',
                                  title='epoch: %d, OP: %.4f, PC: %.4f, mIoU: %.4f' % (epoch+1, train_OP, train_PC, train_mIoU))
            save_confusion_matrix(cm=val_cm, path=result_path+'cm_val.png',
                                  title='epoch: %d, l1: %.4f, OP: %.4f, PC: %.4f, mIoU: %.4f' % (epoch+1, val_l1, val_OP, val_PC, val_mIoU))
            save_confusion_matrix(cm=test_cm, path=result_path+'cm_test.png',
                                  title='epoch: %d, OP: %.4f, PC: %.4f, mIoU: %.4f' % (epoch+1, test_OP, test_PC, test_mIoU))

            best_validation_loss = val_l1
            final_OP = test_OP
            final_PC = test_PC
            final_mIoU = test_mIoU

        np.save(result_path+'train_OP', train_OPs)
        np.save(result_path+'train_PC', train_PCs)
        np.save(result_path+'train_mIoU', train_mIoUs)
        np.save(result_path+'test_OP', test_OPs)
        np.save(result_path+'test_PC', test_PCs)
        np.save(result_path+'test_mIoU', test_mIoUs)
        plt.plot(train_OPs, label='train_OP')
        plt.plot(train_PCs, label='train_PC')
        plt.plot(train_mIoUs, label='train_mIoU')
        plt.plot(test_OPs, label='test_OP')
        plt.plot(test_PCs, label='test_PC')
        plt.plot(test_mIoUs, label='test_mIoU')
        plt.legend()
        plt.ylim(0, 1)
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.savefig(result_path+'curve_acc.png')
        plt.close()

        np.save(result_path+'train_loss', train_losses)
        np.save(result_path+'val_loss', val_losses)
        plt.plot(train_losses, label='train_loss')
        plt.plot(val_losses, label='val_loss')
        plt.plot(test_losses, label='test_loss')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(result_path+'curve_loss.png')
        plt.close()

    log.info(OmegaConf.to_yaml(cfg))
    log.info('OP: %.4f, PC: %.4f, mIoU: %.4f' %
             (final_OP, final_PC, final_mIoU))
    log.info('--------------------------------------------------')


if __name__ == '__main__':
    main()
