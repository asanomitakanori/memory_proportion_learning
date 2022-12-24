import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def num_to_color(num):
    if isinstance(num, list):
        num = num[0]

    if num == 0:
        color = (200, 200, 200)
    elif num == 1:
        color = (0, 0, 255)
    elif num == 2:
        color = (255, 0, 0)
    return color


result_path = '../result_chemotherapy/mini/wsi-fpl/simple_confidence-lr_0.0003-seed_0-eta_5-n_1'

with open('/workspaces/dataset/chemotherapy/202203_chemotherapy/mini/train_index.pkl', "rb") as tf:
    index = pickle.load(tf)
np.random.seed(0)
np.random.shuffle(index)

with open('/workspaces/dataset/chemotherapy/202203_chemotherapy/mini/train_label.pkl', "rb") as tf:
    label = pickle.load(tf)

N = 100
num_epochs = 100
num_classes = 3

index_0, index_1, index_2 = [], [], []
for i in index:
    if label[i[0]][i[1]] == 0:
        index_0.append(i)
    elif label[i[0]][i[1]] == 1:
        index_1.append(i)
    elif label[i[0]][i[1]] == 2:
        index_2.append(i)

    if len(index_2) == N:
        break

print(len(index_0), len(index_1), len(index_2))
index_0, index_1 = index_0[: N], index_1[: N]
print(len(index_0), len(index_1), len(index_2))


############## theta ###############################
theta_0 = np.zeros((N*num_classes, num_epochs))
theta_1 = np.zeros((N*num_classes, num_epochs))
theta_2 = np.zeros((N*num_classes, num_epochs))
for epoch in range(num_epochs):
    with open(result_path + '/theta/%d.pkl' % (epoch+1), "rb") as tf:
        theta = pickle.load(tf)
    # theta = np.load(theta_path + '/theta/%d.npy' %
    #                 (epoch+1), allow_pickle=True)
    # theta = theta.item()
    for n in range(N):
        # GT: 0
        confidence = theta[index_0[n][0]][index_0[n][1]]
        theta_0[n][epoch] = confidence[0]
        theta_0[N+n][epoch] = confidence[1]
        theta_0[2*N+n][epoch] = confidence[2]

        # GT: 1
        confidence = theta[index_1[n][0]][index_1[n][1]]
        theta_1[n][epoch] = confidence[0]
        theta_1[N+n][epoch] = confidence[1]
        theta_1[2*N+n][epoch] = confidence[2]

        # GT: 2
        confidence = theta[index_2[n][0]][index_2[n][1]]
        theta_2[n][epoch] = confidence[0]
        theta_2[N+n][epoch] = confidence[1]
        theta_2[2*N+n][epoch] = confidence[2]

figsize = (10, 5)

plt.figure(figsize=figsize)
g = plt.subplot()
g.set_title('GT: 0')
img = g.imshow(theta_0, aspect='auto')
plt.tick_params(labelleft=False, left=False)
plt.colorbar(img)
plt.savefig(result_path + '/reward_0.png')
plt.close()

plt.figure(figsize=figsize)
g = plt.subplot()
g.set_title('GT: 1')
img = g.imshow(theta_1, aspect='auto')
plt.tick_params(labelleft=False, left=False)
plt.colorbar(img)
plt.savefig(result_path + '/reward_1.png')
plt.close()

plt.figure(figsize=figsize)
g = plt.subplot()
g.set_title('GT: 2')
img = g.imshow(theta_2, aspect='auto')
plt.tick_params(labelleft=False, left=False)
plt.colorbar(img)
plt.savefig(result_path + '/reward_2.png')
plt.close()


############## loss ###############################
loss_0 = np.zeros((N*num_classes, num_epochs))
loss_1 = np.zeros((N*num_classes, num_epochs))
loss_2 = np.zeros((N*num_classes, num_epochs))
for epoch in range(num_epochs):
    with open(result_path + '/accum_loss/%d.pkl' % (epoch+1), "rb") as tf:
        loss = pickle.load(tf)
    # theta = np.load(theta_path + '/theta/%d.npy' %
    #                 (epoch+1), allow_pickle=True)
    # theta = theta.item()
    for n in range(N):
        # GT: 0
        x = loss[index_0[n][0]][index_0[n][1]]
        loss_0[n][epoch] = x[0]
        loss_0[N+n][epoch] = x[1]
        loss_0[2*N+n][epoch] = x[2]

        # GT: 1
        x = loss[index_1[n][0]][index_1[n][1]]
        loss_1[n][epoch] = x[0]
        loss_1[N+n][epoch] = x[1]
        loss_1[2*N+n][epoch] = x[2]

        # GT: 2
        x = loss[index_2[n][0]][index_2[n][1]]
        loss_2[n][epoch] = x[0]
        loss_2[N+n][epoch] = x[1]
        loss_2[2*N+n][epoch] = x[2]

figsize = (10, 5)

plt.figure(figsize=figsize)
g = plt.subplot()
g.set_title('GT: 0')
img = g.imshow(loss_0, aspect='auto')
plt.tick_params(labelleft=False, left=False)
plt.colorbar(img)
plt.savefig(result_path + '/loss_0.png')
plt.close()

plt.figure(figsize=figsize)
g = plt.subplot()
g.set_title('GT: 1')
img = g.imshow(loss_1, aspect='auto')
plt.tick_params(labelleft=False, left=False)
plt.colorbar(img)
plt.savefig(result_path + '/loss_1.png')
plt.close()

plt.figure(figsize=figsize)
g = plt.subplot()
g.set_title('GT: 2')
img = g.imshow(loss_2, aspect='auto')
plt.tick_params(labelleft=False, left=False)
plt.colorbar(img)
plt.savefig(result_path + '/loss_2.png')
plt.close()


############## pseudo label ###############################
pseudo_label = np.zeros((N*num_classes, num_epochs))
for epoch in range(num_epochs):
    with open(result_path + '/p_label/%d.pkl' % (epoch+1), "rb") as tf:
        p_label = pickle.load(tf)
    # theta = np.load(theta_path + '/theta/%d.npy' %
    #                 (epoch+1), allow_pickle=True)
    # theta = theta.item()
    for n in range(N):
        # GT: 0
        x = p_label[index_0[n][0]][index_0[n][1]]
        pseudo_label[n][epoch] = int(x)

        # GT: 1
        x = p_label[index_1[n][0]][index_1[n][1]]
        pseudo_label[N+n][epoch] = int(x)

        # GT: 2
        x = p_label[index_2[n][0]][index_2[n][1]]
        pseudo_label[2*N+n][epoch] = int(x)

figsize = (10, 5)
cmap = mpl.colors.ListedColormap(
    ['r', 'gold', 'limegreen'])

plt.figure(figsize=figsize)
g = plt.subplot()
img_p_label = np.zeros((N*num_classes, num_epochs, 3))
for i in range(N*num_classes):
    for j in range(num_epochs):
        img_p_label[i][j] = num_to_color(pseudo_label[i][j])
img = g.imshow(img_p_label, aspect='auto', interpolation='nearest')
plt.tick_params(labelleft=False, left=False)
plt.savefig(result_path + '/p_label.png')
plt.close()
