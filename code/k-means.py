import numpy as np
import pickle
import json 


def kmeans(k, X, max_iter=300):
    X_size,n_features = X.shape
    
    # ランダムに重心の初期値を初期化
    centroids  = X[np.random.choice(X_size,k)]
    
    # 前の重心と比較するために、仮に新しい重心を入れておく配列を用意
    new_centroids = np.zeros((k, n_features))
    
    # 各データ所属クラスタ情報を保存する配列を用意
    cluster = np.zeros(X_size)
    
    # ループ上限回数まで繰り返し
    for epoch in range(max_iter):
        
        # 入力データ全てに対して繰り返し
        for i in range(X_size):
            
            # データから各重心までの距離を計算（ルートを取らなくても大小関係は変わらないので省略）
            distances = np.sum((centroids - X[i]) ** 2, axis=1)
            
            # データの所属クラスタを距離の一番近い重心を持つものに更新
            cluster[i] = np.argsort(distances)[0]
            
        # すべてのクラスタに対して重心を再計算
        for j in range(k):
            new_centroids[j] = X[cluster==j].mean(axis=0)
            
        # もしも重心が変わっていなかったら終了
        if np.sum(new_centroids == centroids) == k:
            print("break")
            break
        centroids =  new_centroids
    return cluster


with open('/home/asanomi/デスクトップ/dataset/chemotherapy/202203_chemotherapy/all/train_proportion.pkl', "rb") as tf:
    proportion = pickle.load(tf)
init = True

for key in proportion.keys():
    data = proportion[key]
    if init:
        x = data.reshape(1, 3)
        init = False
    else:
        x = np.concatenate([x, data.reshape(1, 3)], axis=0)
# print(x)
y = kmeans(4, x)
print(len(y))


with open('/home/asanomi/デスクトップ/dataset/chemotherapy/202203_chemotherapy/all/train_name.pkl', "rb") as tf:
    name = pickle.load(tf)
print(name)

init = True
for index, na in enumerate(name):
    if init:
        x =[(na, {'label': int(y[index])})]
        init = False
    else:
        x.append((na, {'label': int(y[index])}))
# print(list(x))
# x = list(x)
x = dict(x)

with open('/home/asanomi/デスクトップ/matsuo+ours/name_label.json', 'w') as f:
    json.dump(x, f, indent=4)
