import glob

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

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

n_components = []  # 用于保存降维特征数
scores = []  # 用于保存不同特征维度下的得分

# step3:迭代特征数量,判断不同特征维度下的得分
for n_component in range(10, 200, 10):
    # step4:进行PCA降维，抽取50列特征
    pca = PCA(n_components=n_component, svd_solver='full')

    # step5:使用降维后的数据训练模型
    pca.fit(x_train)

    # step6:转化数据集,进行降维
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)

    # step7:构建模型
    clf = SVC()
    clf.fit(x_train_pca, y_train)
    score = accuracy_score(y_test, clf.predict(x_test_pca))
    print('[PCA降维后精度: ]', score)

    # step8:将特征数、得分追加到列表用于下面画图
    n_components.append(n_component)
    scores.append(score)

# step9:绘制图像
plt.plot(n_components, scores)
plt.show()
