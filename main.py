import glob

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

PICTURE_PATH = "./data/orl_faces"

all_data_set = []  # 原始总数据集，二维矩阵n*m，n个样例，m个属性
all_data_label = []  # 总数据对应的类标签

def get_picture():
    label = 1
    # 读取所有图片并一维化
    while (label <= 20):
        for name in glob.glob(PICTURE_PATH + "\\s" + str(label) + "\\*.pgm"):
            img = Image.open(name)
            # img.getdata()
            # np.array(img).reshape(1, 92*112)
            all_data_set.append(list(img.getdata()))
            all_data_label.append(label)
        label += 1

get_picture()

x = np.array(all_data_set)
y = np.array(all_data_label)

x = MinMaxScaler().fit_transform(x)

# step2:划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# step3:构建支持向量机模型
clf = SVC()

# step4:训练模型
clf.fit(x_train, y_train)

print('[测试集精度: ]', accuracy_score(y_test, clf.predict(x_test)))

# step5:进行PCA降维，抽取10列特征
pca = PCA(n_components=10)

# step6:使用降维后的数据训练模型
pca.fit(x_train)

# step7:转化数据集,进行降维
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

# step8:构建模型
clf = SVC()
clf.fit(x_train_pca, y_train)
print('[PCA降维后精度: ]', accuracy_score(y_test, clf.predict(x_test_pca)))
