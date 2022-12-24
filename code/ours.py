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
from sklearn.metrics import confusion_matrix
from utils import Dataset, fix_seed, make_folder, get_rampup_weight, cal_OP_PC_mIoU, save_confusion_matrix
from losses import PiModelLoss, ProportionLoss, VATLoss
from PIL import Image
from Model.model_backward import Resnet18_k8 as Resnet18
from torch.autograd import Variable
from utils import *


log = logging.getLogger(__name__)

@ hydra.main(config_path='../config', config_name='ours_backward_k8')
def main(cfg: DictConfig) -> None:

    # file name
    cwd = hydra.utils.get_original_cwd()
    result_path = cwd + cfg.result_path
    result_path += 'wsi-proportion-loss/'
    make_folder(result_path)
    result_path += cfg.consistency
    result_path += '-minibatch_%s' % str(cfg.mini_batch)
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
        train_dataset, sampler=ImbalancedDatasetSampler(train_wsi_name), batch_size=1,
        shuffle=False,  num_workers=cfg.num_workers)

    val_dataset = DatasetBag(
        name=val_wsi_name,
        path=val_path,
        label=val_label,
        proportion=val_proportion,
        cwd=cwd)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1,
        shuffle=False,  num_workers=4)

    test_dataset = DatasetBag(
        name=test_wsi_name,
        path=test_path,
        label=test_label,
        proportion=test_proportion,
        cwd=cwd)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1,
        shuffle=False,  num_workers=4)

    # define model
    fix_seed(cfg.seed)
    if cfg.model == 'resnet50':
        model = resnet50(weights='ResNet50_Weights.DEFAULT')
    elif cfg.model == 'resnet18':
        # model = resnet18(pretrained=True)
        model = Resnet18(3)
    else:
        log.info('No model!')
    # model.fc = nn.Linear(model.fc.in_features, cfg.num_classes)
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
                
                ft = []
                ft.append(mb_data.cpu())
                init_flag = True
                for i in range(len(model.net)):
                    if init_flag:
                        for j in range(len(model.net) - 1):
                            model.net[j].requires_grad = False
                        model.net[-1].requires_grad = True
                        init_flag = False
                    ft[-1] = Variable(ft[-1], requires_grad=True)
                    ft.append(model(ft[-1].to(cfg.device), i).cpu())
                pred.extend(ft[-1].argmax(1).cpu().detach().numpy())
                confidence = F.softmax(ft[-1], dim=1)
                pred_prop = torch.zeros(mb_proportion.size(
                    0), cfg.num_classes).to(cfg.device)
                for n in range(mb_proportion.size(0)):
                    pred_prop[n] = torch.mean(
                        confidence[b_list[n]: b_list[n+1]], dim=0)

                prop_loss = loss_function(pred_prop, mb_proportion)
                loss = prop_loss + consistency_loss
                loss.backward()
                ft = ft[:-1]
                grad, ft = get_grad(ft)
                optimizer.step()
                model.net[-1].requires_grad= False

                b_list = [0]
                mb_data, mb_proportion = [], []
                losses.append(loss.item())

                for i in reversed(range(len(model.net)-1)):
                    optimizer.zero_grad()
                    for j in range(i):
                        model.net[0].requires_grad = False
                    model.net[i].requires_grad = True
                    output = model(ft[-1].to(cfg.device), i)
                    loss = torch.sum((grad.to(cfg.device) * output), dim=1).mean()
                    loss.backward()
                    assert grad.shape[1] == output.shape[1], 'difference size error'
                    grad, ft = get_grad(ft)
                    optimizer.step()
                    model.net[i].requires_grad = False
                    output = output.detach().cpu()

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
