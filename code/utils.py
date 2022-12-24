import datetime
import pytz
import os
import torch
from matplotlib import pyplot as plt
import glob
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import cv2
from sklearn.manifold import TSNE
import torchvision.transforms as transforms
from PIL import Image
import pickle
import torch.nn.functional as F
from losses import ProportionLoss
from tqdm import tqdm
import json
from hydra.utils import to_absolute_path as abs_path


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/torchsampler/imbalanced.py
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, train_wsi_name, indices=None, num_samples=None, callback_get_label=None):
        with open(abs_path('name_4label.json')) as f:
            self.propotion_cluster = json.load(f)
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(train_wsi_name))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(train_wsi_name, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(train_wsi_name, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, train_wsi_name, idx):
        return self.propotion_cluster[train_wsi_name[idx]]['label']

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

def get_date():
    return datetime.datetime.now(
        pytz.timezone('Asia/Tokyo')).strftime('%Y.%m.%d.%H.%M.%S')


def make_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)  # fix the initial value of the network weight
    torch.cuda.manual_seed(seed)  # for cuda
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True  # choose the determintic algorithm


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        self.len = self.data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = self.data[idx]
        data = self.transform(data)
        label = self.label[idx]
        label = torch.tensor(label).long()
        return data, label

def get_rampup_weight(weight, iteration, rampup):
    alpha = weight * sigmoid_rampup(iteration, rampup)
    return alpha

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def save_confusion_matrix(cm, path, title=''):
    sns.heatmap(cm, annot=True, cmap='Blues_r', fmt="d")
    plt.xlabel('pred')
    plt.ylabel('GT')
    plt.title(title)
    plt.savefig(path)
    plt.close()


def cal_OP_PC_mIoU(cm):
    num_classes = cm.shape[0]

    TP_c = np.zeros(num_classes)
    for i in range(num_classes):
        TP_c[i] = cm[i][i]

    FP_c = np.zeros(num_classes)
    for i in range(num_classes):
        FP_c[i] = cm[i, :].sum()-cm[i][i]

    FN_c = np.zeros(num_classes)
    for i in range(num_classes):
        FN_c[i] = cm[:, i].sum()-cm[i][i]

    OP = TP_c.sum() / (TP_c+FP_c).sum()
    PC = (TP_c/(TP_c+FP_c)).mean()
    mIoU = (TP_c/(TP_c+FP_c+FN_c)).mean()

    return OP, PC, mIoU


def create_video_cm(cwd):
    size = (640, 480)  # サイズ指定
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 保存形式
    save = cv2.VideoWriter(cwd+'cm_uni.mp4', fourcc, 10.0, size)  # 動画を保存する形を作成

    print("saving...")
    for epoch in range(100):
        img_path = glob.glob(cwd+'*True*cm_%s.png' % (epoch+1))[0]
        print(img_path)
        img = cv2.imread(img_path)  # 画像を読み込む
        img = cv2.resize(img, (640, 480))  # 上でサイズを指定していますが、念のため
        save.write(img)  # 保存

    print("saved")
    save.release()  # ファイルを閉じる


def visualize_feature_space(path, label, epoch):
    feature = np.load(path)
    f_embedded = TSNE(n_components=2).fit_transform(feature)
    for i in range(label.max()+1):
        x = f_embedded[label == i][:, 0]
        y = f_embedded[label == i][:, 1]
        plt.scatter(x, y, label=i, alpha=0.3)
    plt.axis('off')
    plt.legend()
    plt.title('epoch: %d' % (epoch))
    plt.savefig(path[:-4]+'.png')
    plt.close()


def create_video_feature_space(cwd):
    size = (640, 480)  # サイズ指定
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 保存形式
    save = cv2.VideoWriter(cwd+'test_feature_space_xx.mp4',
                           fourcc, 1.0, size)  # 動画を保存する形を作成

    print("saving...")
    for epoch in range(10, 101, 10):
        img_path = glob.glob(cwd+'*False*test_feature_%s.png' % (epoch))[0]
        print(img_path)
        img = cv2.imread(img_path)  # 画像を読み込む
        img = cv2.resize(img, (640, 480))  # 上でサイズを指定していますが、念のため
        save.write(img)  # 保存

    print("saved")
    save.release()  # ファイルを閉じる


def show_figure(cwd):

    # path = 'add_proportion_loss/fpl_cifar10_True_0.001_10_1_simple_confidence_100_64_0.01_1'
    # plt.plot(np.load(cwd+path+'/test_acc.npy'),
    #          label='proportion + 0.01 * ce (lr=0.001)')
    # path = 'add_proportion_loss/fpl_cifar10_True_0.0001_10_1_simple_confidence_100_64_0.01_1'
    # plt.plot(np.load(cwd+path+'/test_acc.npy'),
    #          label='proportion + 0.01 * ce (lr=0.0001)')

    # path = 'add_proportion_loss/fpl_cifar10_True_0.001_10_1_simple_confidence_100_64_0.1_1'
    # plt.plot(np.load(cwd+path+'/test_acc.npy'),
    #          label='proportion + 0.1 * ce (lr=0.001)')
    # path = 'add_proportion_loss/fpl_cifar10_True_0.0001_10_1_simple_confidence_100_64_0.1_1'
    # plt.plot(np.load(cwd+path+'/test_acc.npy'),
    #          label='proportion + 0.1 * ce (lr=0.0001)')

    # path = 'add_proportion_loss/fpl_cifar10_True_0.001_10_1_simple_confidence_100_64_1_1'
    # plt.plot(np.load(cwd+path+'/test_acc.npy'),
    #          label='proportion + ce (lr=0.001)')
    # path = 'add_proportion_loss/fpl_cifar10_True_0.0001_10_1_simple_confidence_100_64_1_1'
    # plt.plot(np.load(cwd+path+'/test_acc.npy'),
    #          label='proportion + ce (lr=0.0001)')

    # path = 'add_proportion_loss_mini30/fpl_cifar10_True_0.001_10_1_simple_confidence_100_64_1_0'
    # plt.plot(np.load(cwd+path+'/test_acc.npy'), label='ce (lr=0.001)')

    path = 'add_proportion_loss_mini30/fpl_cifar10_True_0.001_10_1_simple_confidence_100_64_0_1'
    plt.plot(np.load(cwd+path+'/train_acc.npy'), label='proportion (lr=0.001)')
    path = 'add_proportion_loss_mini30/fpl_cifar10_True_0.0001_10_1_simple_confidence_100_64_0_1'
    plt.plot(np.load(cwd+path+'/train_acc.npy'),
             label='proportion (lr=0.0001)')

    # path = 'add_proportion_loss_mini8/fpl_cifar10_True_0.001_10_1_simple_confidence_100_64_0.1_1'
    # plt.plot(np.load(cwd+path+'/label_acc.npy'),
    #          label='proportion + 0.1 * ce  (lr=0.001)')
    # path = 'add_proportion_loss_mini8/fpl_cifar10_True_0.001_10_1_simple_confidence_100_64_0.01_1'
    # plt.plot(np.load(cwd+path+'/label_acc.npy'),
    #          label='proportion + 0.01 * ce  (lr=0.001)')
    # path = 'add_proportion_loss_mini8/fpl_cifar10_True_0.001_10_1_simple_confidence_100_64_1_1'
    # plt.plot(np.load(cwd+path+'/label_acc.npy'),
    #          label='proportion + ce (lr=0.001)')

    # path = 'add_proportion_loss_mini8/fpl_cifar10_True_0.001_10_1_simple_confidence_100_64_1_0'
    # plt.plot(np.load(cwd+path+'/label_acc.npy'), label='ce (lr=0.001)')
    # path = 'debug/fpl_cifar10_True_0.0001_10_1_simple_confidence_100_64_1_0'
    # plt.plot(np.load(cwd+path+'/label_acc.npy'), label='ce (lr=0.0001)')

    plt.legend(bbox_to_anchor=(1, 0), loc='lower right',
               borderaxespad=0, fontsize=7)
    plt.ylim(0, 1)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.savefig(cwd+'add_proportion_loss_mini8/label_acc.png')
    plt.close()


def visualize_theta(cwd, label):
    show_label = 0
    theta_true_list, theta_false_list = [], []
    for epoch in range(100):
        theta = np.load(glob.glob(cwd+'*True*theta_%s.npy' % (epoch+1))[0])
        theta = theta.reshape(-1, theta.shape[-1])
        theta_true = theta[label == show_label][:, show_label]
        theta_false = theta[label != show_label][:, show_label]
        theta_true_list.append(theta_true)
        theta_false_list.append(theta_false)
    theta_true = np.array(theta_true_list).transpose()
    theta_false = np.array(theta_false_list).transpose()

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    im1 = ax1.imshow(theta_true, aspect=1/((theta_true.shape[0]/50)))
    plt.colorbar(im1)
    ax2 = fig.add_subplot(2, 1, 2)
    im2 = ax2.imshow(theta_false, aspect=1/((theta_false.shape[0]/50)))
    plt.colorbar(im2)

    plt.savefig(cwd+'/uni_theta_%d' % show_label)


if __name__ == '__main__':
    cwd = './result/'

    show_figure(cwd)



def get_grad(feature):
    # feature = feature[:-1]
    grad = feature[-1].grad
    feature = feature[:-1]
    return grad, feature
    

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
                y= model(data_j)
                pred.extend(y.argmax(1).cpu().detach().numpy())
                confidence.extend(
                    F.softmax(y, dim=1).cpu().detach().numpy())

            pred_prop = torch.tensor(np.array(confidence)).mean(dim=0)
            l1 = l1_function(pred_prop, proportion).item()

        # acc = np.array(np.array(gt) == np.array(pred)).mean()
        cm = confusion_matrix(y_true=gt, y_pred=pred)
        OP, PC, mIoU = cal_OP_PC_mIoU(cm)

    return l1, cm, OP, PC, mIoU
    